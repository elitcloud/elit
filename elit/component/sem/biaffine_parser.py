# coding: utf-8

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import os
from abc import ABC
from typing import List

import mxnet as mx
import numpy as np
from mxnet import nd, ndarray, autograd
from mxnet.gluon import nn
from mxnet.gluon.loss import SoftmaxCrossEntropyLoss, SigmoidBinaryCrossEntropyLoss

from elit.component.sem.data import ParserVocabulary
from elit.component.dep.common.utils import orthonormal_VanillaLSTMBuilder, bilinear, reshape_fortran, leaky_relu, \
    biLSTM, \
    orthonormal_initializer, flatten_numpy, embedding_from_numpy, mxnet_prefer_gpu, parameter_from_numpy, \
    parameter_init, freeze, rel_argmax, arc_argmax


class RNN(nn.Block):

    def __init__(self,
                 word_embs,
                 tag_embs,
                 word_dims,
                 tag_dims,
                 dropout_dim,
                 lstm_layers,
                 lstm_hiddens,
                 dropout_lstm_input,
                 dropout_lstm_hidden,
                 bert=0,
                 debug=False
                 ):
        super().__init__()
        self.tag_embs = embedding_from_numpy(tag_embs) if tag_embs is not None else None
        self.dropout_lstm_hidden = dropout_lstm_hidden

        def _emb_mask_generator(seq_len, batch_size):
            """word and tag dropout (drop whole word and tag)

            Parameters
            ----------
            seq_len : int
                length of sequence
            batch_size : int
                batch size

            Returns
            -------
            np.ndarray
                dropout mask for word and tag
            """
            wm, tm = nd.zeros((seq_len, batch_size, 1)), nd.zeros((seq_len, batch_size, 1))
            for i in range(seq_len):
                word_mask = np.random.binomial(1, 1. - dropout_dim, batch_size).astype(np.float32)
                if self.tag_embs:
                    tag_mask = np.random.binomial(1, 1. - dropout_dim, batch_size).astype(np.float32)
                    scale = 3. / (2. * word_mask + tag_mask + 1e-12)
                    tag_mask *= scale
                    word_mask *= scale
                    tag_mask = nd.array(tag_mask)
                    tm[i, :, 0] = tag_mask
                word_mask = nd.array(word_mask)
                wm[i, :, 0] = word_mask
            return wm, tm

        self.generate_emb_mask = _emb_mask_generator
        self.b_lstm = nn.Sequential()
        self.dropout_lstm_input = dropout_lstm_input
        self.f_lstm = nn.Sequential()
        self.pret_word_embs = embedding_from_numpy(word_embs, trainable=False) if isinstance(word_embs,
                                                                                             np.ndarray) else word_embs

        self.f_lstm.add(orthonormal_VanillaLSTMBuilder(1, word_dims + tag_dims + bert, lstm_hiddens, dropout_lstm_input,
                                                       dropout_lstm_hidden, debug))
        self.b_lstm.add(orthonormal_VanillaLSTMBuilder(1, word_dims + tag_dims + bert, lstm_hiddens, dropout_lstm_input,
                                                       dropout_lstm_hidden, debug))
        for i in range(lstm_layers - 1):
            self.f_lstm.add(orthonormal_VanillaLSTMBuilder(1, 2 * lstm_hiddens, lstm_hiddens, dropout_lstm_input,
                                                           dropout_lstm_hidden, debug))
            self.b_lstm.add(orthonormal_VanillaLSTMBuilder(1, 2 * lstm_hiddens, lstm_hiddens, dropout_lstm_input,
                                                           dropout_lstm_hidden, debug))

    def feature_detect(self, tag_inputs, word_inputs, bert):
        is_train = autograd.is_training()
        batch_size = word_inputs.shape[1]
        seq_len = word_inputs.shape[0]
        # unked_words = np.where(word_inputs < self._vocab.words_in_train, word_inputs, self._vocab.UNK)
        if self.pret_word_embs is not None:
            word_embs = self.pret_word_embs(nd.array(word_inputs))
            if bert is not None:
                word_embs = nd.concat(word_embs, nd.array(bert), dim=2)
        else:
            word_embs = nd.array(bert)
        tag_embs = self.tag_embs(nd.array(tag_inputs)) if self.tag_embs is not None else None
        # Dropout
        if is_train:
            wm, tm = self.generate_emb_mask(seq_len, batch_size)
            if self.tag_embs is not None:
                emb_inputs = nd.concat(nd.multiply(wm, word_embs), nd.multiply(tm, tag_embs), dim=2)
            else:
                emb_inputs = nd.multiply(wm, word_embs)
        else:
            if self.tag_embs is not None:
                emb_inputs = nd.concat(word_embs, tag_embs, dim=2)  # seq_len x batch_size
            else:
                emb_inputs = word_embs
        top_recur = biLSTM(self.f_lstm, self.b_lstm, emb_inputs, batch_size,
                           dropout_x=self.dropout_lstm_input if is_train else 0)
        return top_recur

    def forward(self, tag_inputs, word_inputs, bert):
        return self.feature_detect(tag_inputs, word_inputs, bert)


class MLP(nn.Block):
    def __init__(self, lstm_hiddens, mlp_arc_size, mlp_rel_size, dropout_mlp, debug=False):
        super().__init__()
        self.dropout_mlp = dropout_mlp
        mlp_size = mlp_arc_size + mlp_rel_size
        W = orthonormal_initializer(mlp_size, 2 * lstm_hiddens, debug)
        self.mlp_dep_W = parameter_from_numpy(self, 'mlp_dep_W', W)
        self.mlp_dep_b = parameter_init(self, 'mlp_dep_b', (mlp_size,), mx.init.Zero())
        self.mlp_head_W = parameter_from_numpy(self, 'mlp_head_W', W)
        self.mlp_head_b = parameter_init(self, 'mlp_head_b', (mlp_size,), mx.init.Zero())
        self.mlp_rel_size = mlp_rel_size
        self.mlp_arc_size = mlp_arc_size

    def forward(self, top_recur):
        return self.mlp(top_recur)

    def mlp(self, top_recur):
        is_train = autograd.is_training()
        if is_train:
            top_recur = nd.Dropout(data=top_recur, axes=[0], p=self.dropout_mlp)
        W_dep, b_dep = self.mlp_dep_W.data(), self.mlp_dep_b.data()
        W_head, b_head = self.mlp_head_W.data(), self.mlp_head_b.data()
        dep, head = leaky_relu(nd.dot(top_recur, W_dep.T) + b_dep), leaky_relu(nd.dot(top_recur, W_head.T) + b_head)
        if is_train:
            dep, head = nd.Dropout(data=dep, axes=[0], p=self.dropout_mlp), nd.Dropout(data=head, axes=[0],
                                                                                       p=self.dropout_mlp)
        dep, head = nd.transpose(dep, axes=[2, 0, 1]), nd.transpose(head, axes=[2, 0, 1])
        dep_arc, dep_rel = dep[:self.mlp_arc_size], dep[self.mlp_arc_size:]
        head_arc, head_rel = head[:self.mlp_arc_size], head[self.mlp_arc_size:]
        return dep_arc, dep_rel, head_arc, head_rel


class BiAffine(nn.Block):
    def __init__(self, vocab,
                 mlp_arc_size,
                 mlp_rel_size):
        super(BiAffine, self).__init__()
        self._vocab = vocab
        self.binary_ce_loss = SigmoidBinaryCrossEntropyLoss(batch_axis=-1)
        self.rel_W = parameter_init(self, 'rel_W', (vocab.rel_size * (mlp_rel_size + 1), mlp_rel_size + 1),
                                    init=mx.init.Zero())
        self.arc_W = parameter_init(self, 'arc_W', (mlp_arc_size, mlp_arc_size + 1), init=mx.init.Zero())
        self.softmax_loss = SoftmaxCrossEntropyLoss(axis=0, batch_axis=-1)
        self.mlp_arc_size = mlp_arc_size
        self.mlp_rel_size = mlp_rel_size
        # self.initialize()

    def forward(self, dep_arc, dep_rel, head_arc, head_rel, word_inputs, arc_targets=None, rel_targets=None,
                blend=None):
        # seq_len x batch_size
        mask = np.greater(word_inputs, self._vocab.ROOT).astype(np.float32)
        return self.biaffine(dep_arc, dep_rel, head_arc, head_rel, mask, arc_targets, rel_targets, blend)

    def biaffine(self, dep_arc, dep_rel, head_arc, head_rel, mask, arc_targets, rel_targets, blend):
        is_train = autograd.is_training()
        batch_size = mask.shape[1]
        seq_len = mask.shape[0]
        W_arc = self.arc_W.data()
        arc_logits = bilinear(dep_arc, W_arc, head_arc, self.mlp_arc_size, seq_len, batch_size,
                              num_outputs=1,
                              bias_x=True, bias_y=False)  # type: nd.NDArray
        if blend is not None:
            arc_logits = arc_logits + blend
        # (#head x #dep) x batch_size
        flat_arc_logits = reshape_fortran(arc_logits, (seq_len, seq_len * batch_size))
        # (#head ) x (#dep x batch_size)
        arc_preds = nd.greater(arc_logits, 0)  # sigmoid y > 0.5 when x > 0
        if is_train or arc_targets is not None:
            arc_correct = arc_preds.asnumpy() * arc_targets
            arc_accuracy = np.sum(arc_correct) / np.sum(arc_targets * mask)
            # targets_1D = flatten_numpy(arc_targets)
            # losses = self.softmax_loss(flat_arc_logits, nd.array(targets_1D))
            flat_arc_targets = reshape_fortran(arc_targets, (seq_len, seq_len * batch_size))
            losses = self.binary_ce_loss(flat_arc_logits, nd.array(flat_arc_targets))
            if is_train or arc_targets is not None:
                mask_1D_tensor = nd.array(flatten_numpy(mask))
            arc_loss = nd.sum(losses * mask_1D_tensor) / mask_1D_tensor.sum()
            # return arc_accuracy, 0, 0, arc_loss
        W_rel = self.rel_W.data()
        rel_logits = bilinear(dep_rel, W_rel, head_rel, self.mlp_rel_size, seq_len, batch_size,
                              num_outputs=self._vocab.rel_size, bias_x=True, bias_y=True)  # type: nd.NDArray
        # #head x rel_size x #dep x batch_size
        flat_rel_logits = reshape_fortran(rel_logits.transpose([1, 0, 2, 3]),
                                          (self._vocab.rel_size, seq_len * seq_len * batch_size))
        # rel_size x (#head x #dep x batch_size)
        if is_train or arc_targets is not None:
            mask_rel = reshape_fortran(nd.array(mask * arc_targets),
                                       (1, seq_len * seq_len * batch_size))  # type: nd.NDArray
            flat_rel_preds = flat_rel_logits.argmax(0)
            flat_rel_target = nd.array(reshape_fortran(rel_targets, (1, seq_len * seq_len * batch_size))).squeeze(
                axis=0)
            rel_correct = nd.equal(flat_rel_preds, flat_rel_target).asnumpy()
            rel_correct = rel_correct * flatten_numpy(arc_targets * mask)
            rel_accuracy = np.sum(rel_correct) / np.sum(arc_targets * mask)
            losses = self.softmax_loss(flat_rel_logits, flat_rel_target)
            rel_loss = nd.sum(losses * mask_rel) / mask_rel.sum()
        if is_train or arc_targets is not None:
            loss = arc_loss + rel_loss
        if is_train:
            return arc_accuracy, rel_accuracy, loss
        outputs = []
        rel_preds = rel_logits.transpose([1, 0, 2, 3]).argmax(0)
        arc_preds = arc_preds.transpose([2, 0, 1])
        rel_preds = rel_preds.transpose([2, 0, 1])
        for msk, arc_pred, rel_pred in zip(np.transpose(mask), arc_preds, rel_preds):
            # parse sentences one by one
            msk[0] = 1.
            sent_len = int(np.sum(msk))
            arc_pred = arc_pred[:sent_len, :sent_len]
            outputs.append((arc_pred[:sent_len, :sent_len], arc_pred * rel_pred[:sent_len, :sent_len]))
        return outputs


class BiAffineDep(BiAffine):

    def __init__(self, vocab, mlp_arc_size, mlp_rel_size, interpolation=0.5):
        super().__init__(vocab, mlp_arc_size, mlp_rel_size)
        self.interpolation = interpolation

    def biaffine(self, dep_arc, dep_rel, head_arc, head_rel, mask, arc_targets, rel_targets, blend):
        is_train = autograd.is_training()
        batch_size = mask.shape[1]
        seq_len = mask.shape[0]
        num_tokens = int(np.sum(mask))  # non padding, non root token number
        W_arc = self.arc_W.data()
        arc_logits = bilinear(dep_arc, W_arc, head_arc, self.mlp_arc_size, seq_len, batch_size, num_outputs=1,
                              bias_x=True, bias_y=False)
        # return 0, 0, 0, arc_logits.sum()
        # (#head x #dep) x batch_size

        flat_arc_logits = reshape_fortran(arc_logits, (seq_len, seq_len * batch_size))
        # (#head ) x (#dep x batch_size)

        arc_preds = arc_logits.argmax(0)
        if len(arc_preds.shape) == 1:  # dynet did unnecessary jobs
            arc_preds = np.expand_dims(arc_preds, axis=1)
        # seq_len x batch_size

        if is_train or arc_targets is not None:
            mask_1D = flatten_numpy(mask)
            # mask_1D_tensor = nd.inputTensor(mask_1D, batched=True)
            mask_1D_tensor = nd.array(mask_1D)
            correct = np.equal(arc_preds.asnumpy(), arc_targets)
            arc_correct = correct.astype(np.float32) * mask
            arc_accuracy = np.sum(arc_correct) / num_tokens
            targets_1D = flatten_numpy(arc_targets)
            losses = self.softmax_loss(flat_arc_logits, nd.array(targets_1D))
            arc_loss = nd.sum(losses * mask_1D_tensor) / num_tokens

        if not is_train:
            arc_probs = np.transpose(
                np.reshape(nd.softmax(flat_arc_logits, axis=0).asnumpy(), (seq_len, seq_len, batch_size), 'F'))
        # #batch_size x #dep x #head

        W_rel = self.rel_W.data()
        # dep_rel = nd.concat([dep_rel, nd.inputTensor(np.ones((1, seq_len),dtype=np.float32))])
        # head_rel = nd.concat([head_rel, nd.inputTensor(np.ones((1, seq_len), dtype=np.float32))])
        rel_logits = bilinear(dep_rel, W_rel, head_rel, self.mlp_rel_size, seq_len, batch_size,
                              num_outputs=self._vocab.rel_size, bias_x=True, bias_y=True)
        # (#head x rel_size x #dep) x batch_size

        flat_rel_logits = reshape_fortran(rel_logits, (seq_len, self._vocab.rel_size, seq_len * batch_size))
        # (#head x rel_size) x (#dep x batch_size)

        _target_vec = nd.array(targets_1D if is_train else flatten_numpy(arc_preds.asnumpy())).reshape(
            seq_len * batch_size, 1)
        _target_mat = _target_vec * nd.ones((1, self._vocab.rel_size))

        partial_rel_logits = nd.pick(flat_rel_logits, _target_mat.T, axis=0)
        # (rel_size) x (#dep x batch_size)

        if is_train or arc_targets is not None:
            rel_preds = partial_rel_logits.argmax(0)
            targets_1D = flatten_numpy(rel_targets)
            rel_correct = np.equal(rel_preds.asnumpy(), targets_1D).astype(np.float32) * mask_1D
            rel_accuracy = np.sum(rel_correct) / num_tokens
            losses = self.softmax_loss(partial_rel_logits, nd.array(targets_1D))
            rel_loss = nd.sum(losses * mask_1D_tensor) / num_tokens

        if not is_train:
            rel_probs = np.transpose(np.reshape(nd.softmax(flat_rel_logits.transpose([1, 0, 2]), axis=0).asnumpy(),
                                                (self._vocab.rel_size, seq_len, seq_len, batch_size), 'F'))
        # batch_size x #dep x #head x #nclasses

        if is_train or arc_targets is not None:
            # loss = 2 * ((1 - self.interpolation) * arc_loss + self.interpolation * rel_loss)
            loss = arc_loss + rel_loss

        if is_train:
            return arc_accuracy, rel_accuracy, loss

        outputs = []

        for msk, arc_prob, rel_prob in zip(np.transpose(mask), arc_probs, rel_probs):
            # parse sentences one by one
            msk[0] = 1.
            sent_len = int(np.sum(msk))
            arc_pred = arc_argmax(arc_prob, sent_len, msk)
            rel_prob = rel_prob[np.arange(len(arc_pred)), arc_pred]
            rel_pred = rel_argmax(rel_prob, sent_len)
            outputs.append((arc_pred[1:sent_len], rel_pred[1:sent_len]))

        return outputs


class UnlabeledBiAffine(nn.Block):
    def __init__(self, mlp_arc_size):
        super().__init__()
        self.binary_ce_loss = SigmoidBinaryCrossEntropyLoss(batch_axis=-1)
        self.arc_W = parameter_init(self, 'arc_W', (mlp_arc_size, mlp_arc_size + 1), init=mx.init.Zero())
        self.mlp_arc_size = mlp_arc_size
        # self.initialize()

    def forward(self, dep_arc, head_arc, mask, arc_targets=None):
        return self.biaffine(dep_arc, head_arc, mask, arc_targets)

    def biaffine(self, dep_arc, head_arc, mask, arc_targets):
        is_train = autograd.is_training()
        batch_size = mask.shape[1]
        seq_len = mask.shape[0]
        W_arc = self.arc_W.data()
        arc_logits = bilinear(dep_arc, W_arc, head_arc, self.mlp_arc_size, seq_len, batch_size,
                              num_outputs=1,
                              bias_x=True, bias_y=False)  # type: nd.NDArray
        # #head x #dep x batch_size
        if not is_train:
            return arc_logits
        # (#head x #dep) x batch_size
        flat_arc_logits = reshape_fortran(arc_logits, (seq_len, seq_len * batch_size))
        # (#head ) x (#dep x batch_size)
        flat_arc_targets = reshape_fortran(arc_targets, (seq_len, seq_len * batch_size))
        losses = self.binary_ce_loss(flat_arc_logits, nd.array(flat_arc_targets))
        mask_1D_tensor = nd.array(flatten_numpy(mask))
        arc_loss = nd.sum(losses * mask_1D_tensor) / mask_1D_tensor.sum()

        return arc_logits, arc_loss


class DoubleBiAffine(BiAffine):
    def __init__(self, vocab, lstm_hiddens, mlp_arc_size, mlp_rel_size, dropout_mlp, debug=False):
        super().__init__(vocab, lstm_hiddens, mlp_arc_size, mlp_rel_size, dropout_mlp, debug)
        self.first_biaffine = UnlabeledBiAffine(mlp_arc_size)

    def forward(self, top_recur, word_inputs, arc_targets=None, rel_targets=None):
        dep_arc, dep_rel, head_arc, head_rel = self.mlp(top_recur)
        is_train = autograd.is_training()
        # seq_len x batch_size
        mask = np.greater(word_inputs, self._vocab.ROOT).astype(np.float32)
        if is_train:
            arc_logits, arc_loss = self.first_biaffine(dep_arc, head_arc, mask, arc_targets)
        else:
            arc_logits = self.first_biaffine(dep_arc, head_arc, mask, arc_targets)

    def mlp(self, top_recur):
        is_train = autograd.is_training()
        if is_train:
            top_recur = nd.Dropout(data=top_recur, axes=[0], p=self.dropout_mlp)
        W_dep, b_dep = self.mlp_dep_W.data(), self.mlp_dep_b.data()
        W_head, b_head = self.mlp_head_W.data(), self.mlp_head_b.data()
        dep, head = leaky_relu(nd.dot(top_recur, W_dep.T) + b_dep), leaky_relu(nd.dot(top_recur, W_head.T) + b_head)
        if is_train:
            dep, head = nd.Dropout(data=dep, axes=[0], p=self.dropout_mlp), nd.Dropout(data=head, axes=[0],
                                                                                       p=self.dropout_mlp)
        dep, head = nd.transpose(dep, axes=[2, 0, 1]), nd.transpose(head, axes=[2, 0, 1])
        dep_arc, dep_rel = dep[:self.mlp_arc_size], dep[self.mlp_arc_size:]
        head_arc, head_rel = head[:self.mlp_arc_size], head[self.mlp_arc_size:]
        return dep_arc, dep_rel, head_arc, head_rel


class BasicParser(nn.Block, ABC):

    def __init__(self, vocab,
                 word_dims,
                 tag_dims,
                 dropout_dim,
                 lstm_layers,
                 lstm_hiddens,
                 dropout_lstm_input,
                 dropout_lstm_hidden,
                 mlp_arc_size,
                 mlp_rel_size,
                 dropout_mlp,
                 debug=False
                 ):
        super().__init__()

    def save_parameters(self, filename):
        """Save model

        Parameters
        ----------
        filename : str
            path to model file
        """

        params = self._collect_params_with_prefix()
        params.pop('rnn.pret_word_embs.weight', None)
        arg_dict = {key: val._reduce() for key, val in params.items()}
        ndarray.save(filename, arg_dict)

    def save(self, save_path):
        """Save model

        Parameters
        ----------
        filename : str
            path to model file
        """
        self.save_parameters(save_path)

    def load(self, load_path, ctx=None):
        """Load model

        Parameters
        ----------
        load_path : str
            path to model file
        """
        if not ctx:
            ctx = mxnet_prefer_gpu()
        self.load_parameters(load_path, allow_missing=True, ctx=ctx)


class BiaffineParser(BasicParser):
    def __init__(self, vocab: ParserVocabulary,
                 word_dims,
                 tag_dims,
                 dropout_dim,
                 lstm_layers,
                 lstm_hiddens,
                 dropout_lstm_input,
                 dropout_lstm_hidden,
                 mlp_arc_size,
                 mlp_rel_size,
                 dropout_mlp,
                 bert=0,
                 interpolation=0.5,
                 debug=False
                 ):
        """A MXNet replicate of biaffine parser, see following paper
        Dozat, T., & Manning, C. D. (2016). Deep biaffine attention for neural dependency parsing. arXiv:1611.01734.
        It's a re-implementation of DyNet version https://github.com/jcyk/Dynet-Biaffine-dependency-parser

        Parameters
        ----------
        vocab : ParserVocabulary
            built from a data set
        word_dims : int
            word vector dimension
        tag_dims : int
            tag vector dimension
        dropout_dim : float
            keep rate of word dropout (drop out entire embedding)
        lstm_layers : int
            number of lstm layers
        lstm_hiddens : int
            size of lstm hidden states
        dropout_lstm_input : float
            dropout on x in variational RNN
        dropout_lstm_hidden : float
            dropout on h in variational RNN
        mlp_arc_size : int
            output size of MLP for arc feature extraction
        mlp_rel_size : int
            output size of MLP for rel feature extraction
        dropout_mlp : float
            dropout on the output of LSTM
        debug : bool
            debug mode
        """
        super().__init__(vocab, word_dims, tag_dims, dropout_dim, lstm_layers, lstm_hiddens, dropout_lstm_input,
                         dropout_lstm_hidden, mlp_arc_size, mlp_rel_size, dropout_mlp, debug)
        self.create_graph(dropout_dim, dropout_lstm_hidden, dropout_lstm_input, dropout_mlp, lstm_hiddens, lstm_layers,
                          mlp_arc_size, mlp_rel_size, tag_dims, vocab, word_dims, bert, interpolation, debug)

    def create_graph(self, dropout_dim, dropout_lstm_hidden, dropout_lstm_input, dropout_mlp, lstm_hiddens, lstm_layers,
                     mlp_arc_size, mlp_rel_size, tag_dims, vocab, word_dims, bert, interpolation, debug):
        self.rnn = RNN(vocab.get_pret_embs(word_dims) if word_dims > 0 else None,
                       vocab.get_tag_embs(tag_dims) if tag_dims else None, word_dims,
                       tag_dims, dropout_dim, lstm_layers, lstm_hiddens, dropout_lstm_input,
                       dropout_lstm_hidden, bert, debug)
        self.mlp = MLP(lstm_hiddens, mlp_arc_size, mlp_rel_size, dropout_mlp, debug)
        self.biaffine = BiAffine(vocab, mlp_arc_size, mlp_rel_size)

    def forward(self, word_inputs, bert, tag_inputs, arc_targets=None, rel_targets=None):
        """Run decoding

        Parameters
        ----------
        word_inputs : mxnet.ndarray.NDArray
            word indices of seq_len x batch_size
        tag_inputs : mxnet.ndarray.NDArray
            tag indices of seq_len x batch_size
        arc_targets : mxnet.ndarray.NDArray
            gold arc indices of head x dep x batch
        rel_targets : mxnet.ndarray.NDArray
            gold rel indices of head x dep x batch
        Returns
        -------
        tuple
            (arc_accuracy, rel_accuracy, overall_accuracy, loss) when training, else if given gold target
        then return arc_accuracy, rel_accuracy, overall_accuracy, outputs, otherwise return outputs, where outputs is a
        list of (arcs, rels).
        """
        top_recur = self.rnn(tag_inputs, word_inputs, bert)
        dep_arc, dep_rel, head_arc, head_rel = self.mlp(top_recur)
        return self.biaffine(dep_arc, dep_rel, head_arc, head_rel, word_inputs,
                             arc_targets, rel_targets)


class BiaffineDepParser(BiaffineParser):

    def __init__(self, vocab: ParserVocabulary, word_dims, tag_dims, dropout_dim, lstm_layers, lstm_hiddens,
                 dropout_lstm_input, dropout_lstm_hidden, mlp_arc_size, mlp_rel_size, dropout_mlp, bert=0,
                 interpolation=0.5, debug=False):
        super().__init__(vocab, word_dims, tag_dims, dropout_dim, lstm_layers, lstm_hiddens, dropout_lstm_input,
                         dropout_lstm_hidden, mlp_arc_size, mlp_rel_size, dropout_mlp, bert, interpolation, debug)

    def create_graph(self, dropout_dim, dropout_lstm_hidden, dropout_lstm_input, dropout_mlp, lstm_hiddens, lstm_layers,
                     mlp_arc_size, mlp_rel_size, tag_dims, vocab, word_dims, bert, interpolation, debug):
        self.rnn = RNN(vocab.get_pret_embs(word_dims) if word_dims else None,
                       vocab.get_tag_embs(tag_dims) if tag_dims else None, word_dims,
                       tag_dims, dropout_dim, lstm_layers, lstm_hiddens, dropout_lstm_input,
                       dropout_lstm_hidden, bert, debug)
        self.mlp = MLP(lstm_hiddens, mlp_arc_size, mlp_rel_size, dropout_mlp, debug)
        self.biaffine = BiAffineDep(vocab, mlp_arc_size, mlp_rel_size, interpolation)


class SharedRNNParser(BasicParser):

    def __init__(self, vocab: List[ParserVocabulary], word_dims, tag_dims, dropout_dim, lstm_layers, lstm_hiddens,
                 dropout_lstm_input, dropout_lstm_hidden, mlp_arc_size, mlp_rel_size, dropout_mlp, transfer=False,
                 bert=0, debug=False):
        super().__init__(vocab, word_dims, tag_dims, dropout_dim, lstm_layers, lstm_hiddens, dropout_lstm_input,
                         dropout_lstm_hidden, mlp_arc_size, mlp_rel_size, dropout_mlp, debug)
        self.rnn = RNN(vocab[0].get_pret_embs(word_dims), vocab[0].get_tag_embs(tag_dims), word_dims,
                       tag_dims, dropout_dim, lstm_layers, lstm_hiddens, dropout_lstm_input,
                       dropout_lstm_hidden, bert, debug)
        self.mlps = nn.Sequential()
        self.decoders = nn.Sequential()
        self.create_graph(vocab, lstm_hiddens, mlp_arc_size, mlp_rel_size, dropout_mlp, bert, debug)
        self.transfer = transfer

    def dump(self, path: str):
        self.rnn.save_parameters(os.path.join(path, 'rnn.bin'))
        for i, (mlp, decoder) in enumerate(zip(self.mlps, self.decoders)):
            mlp.save_parameters(os.path.join(path, 'mlp{}.bin'.format(i)))
            decoder.save_parameters(os.path.join(path, 'decoder{}.bin'.format(i)))

    def fill(self, path):
        rnn_path = os.path.join(path, 'rnn.bin')
        if os.path.isfile(rnn_path):
            # print('load rnn')
            self.rnn.load_parameters(rnn_path, ctx=mxnet_prefer_gpu())
            freeze(self.rnn)

        for i, (mlp, decoder) in enumerate(zip(self.mlps, self.decoders)):
            mlp_path = os.path.join(path, 'mlp{}.bin'.format(i))
            if os.path.isfile(mlp_path):
                # print('load mlp')
                mlp.load_parameters(mlp_path, ctx=mxnet_prefer_gpu())
                freeze(mlp)

            decoder_path = os.path.join(path, 'decoder{}.bin'.format(i))
            if os.path.isfile(decoder_path):
                # print('load decoder')
                decoder.load_parameters(decoder_path, ctx=mxnet_prefer_gpu())
                freeze(decoder)

    def create_graph(self, vocab, lstm_hiddens, mlp_arc_size, mlp_rel_size, dropout_mlp, bert, debug):
        for voc in vocab:
            self.mlps.add(MLP(lstm_hiddens, mlp_arc_size, mlp_rel_size, dropout_mlp, debug))
            self.decoders.add(BiAffine(voc, mlp_arc_size, mlp_rel_size))

    def forward(self, word_inputs, bert, tag_inputs, arc_targets=None, rel_targets=None):
        is_train = arc_targets is not None
        top_recur = self.rnn(tag_inputs, word_inputs, bert)
        if is_train:
            if self.transfer:  # only train on last dataset
                dep_arc, dep_rel, head_arc, head_rel = self.mlps[-1](top_recur)
                arc_accuracy, rel_accuracy, loss = self.decoders[-1](dep_arc, dep_rel, head_arc, head_rel, word_inputs,
                                                                     arc_targets[-1], rel_targets[-1])
                return -1, rel_accuracy, loss
            else:
                total_loss = []
                LF = []
                for mlp, decoder, a, r in zip(self.mlps, self.decoders, arc_targets, rel_targets):
                    dep_arc, dep_rel, head_arc, head_rel = mlp(top_recur)
                    arc_accuracy, rel_accuracy, loss = decoder(dep_arc, dep_rel, head_arc, head_rel, word_inputs, a, r)
                    total_loss.append(loss)
                    LF.append(rel_accuracy)
                return -1, sum(LF) / len(LF), nd.stack(*total_loss).mean()
        else:
            return [decoder(*mlp(top_recur), word_inputs) for mlp, decoder in zip(self.mlps, self.decoders)]


class SharedPrivateRNNParser(BasicParser):

    def __init__(self, vocab: List[ParserVocabulary], word_dims, tag_dims, dropout_dim, lstm_layers, lstm_hiddens,
                 dropout_lstm_input, dropout_lstm_hidden, mlp_arc_size, mlp_rel_size, dropout_mlp, debug=False):
        super().__init__(vocab, word_dims, tag_dims, dropout_dim, lstm_layers, lstm_hiddens, dropout_lstm_input,
                         dropout_lstm_hidden, mlp_arc_size, mlp_rel_size, dropout_mlp, debug)
        self.rnn = RNN(vocab[0].get_pret_embs(word_dims), vocab[0].get_tag_embs(tag_dims), word_dims,
                       tag_dims, dropout_dim, lstm_layers, lstm_hiddens, dropout_lstm_input,
                       dropout_lstm_hidden, debug)
        self.private_rnns = nn.Sequential()
        self.decoders = nn.Sequential()

        for voc in vocab:
            self.private_rnns.add(RNN(self.rnn.pret_word_embs, voc.get_tag_embs(tag_dims), word_dims,
                                      tag_dims, dropout_dim, lstm_layers, lstm_hiddens, dropout_lstm_input,
                                      dropout_lstm_hidden, debug))
            self.decoders.add(BiAffine(voc, lstm_hiddens * 2, mlp_arc_size, mlp_rel_size, dropout_mlp, debug))
        # self.proj_recur = nn.Dense(lstm_hiddens * 2, weight_initializer=mx.init.Xavier(), flatten=False)

    def forward(self, word_inputs, tag_inputs, arc_targets=None, rel_targets=None):
        is_train = arc_targets is not None
        shared_recur = self.rnn(tag_inputs, word_inputs)
        # private_recur = nd.concat(*[rnn(tag_inputs, word_inputs) for rnn in self.private_rnns], dim=2)
        # recur = nd.concat(shared_recur, private_recur, dim=2)
        # top_recur = self.proj_recur(recur)
        top_recurs = [nd.concat(shared_recur, rnn(tag_inputs, word_inputs), dim=2) for rnn in self.private_rnns]
        if is_train:
            total_loss = []
            LF = []
            for top_recur, decoder, a, r in zip(top_recurs, self.decoders, arc_targets, rel_targets):
                arc_accuracy, rel_accuracy, loss = decoder(top_recur, word_inputs, a, r)
                total_loss.append(loss)
                LF.append(rel_accuracy)
            return -1, sum(LF) / len(LF), nd.stack(*total_loss).mean()
        else:
            return [decoder(top_recur, word_inputs) for top_recur, decoder in zip(top_recurs, self.decoders)]


class BlendParser(SharedRNNParser):
    def __init__(self, vocab: List[ParserVocabulary], word_dims, tag_dims, dropout_dim, lstm_layers, lstm_hiddens,
                 dropout_lstm_input, dropout_lstm_hidden, mlp_arc_size, mlp_rel_size, dropout_mlp, debug=False):
        super().__init__(vocab, word_dims, tag_dims, dropout_dim, lstm_layers, lstm_hiddens, dropout_lstm_input,
                         dropout_lstm_hidden, mlp_arc_size, mlp_rel_size, dropout_mlp, debug)
        self.shared_arc_W = parameter_init(self, 'arc_W', (mlp_arc_size, mlp_arc_size), init=mx.init.Xavier())

    def forward(self, word_inputs, tag_inputs, arc_targets=None, rel_targets=None):
        is_train = arc_targets is not None
        top_recur = self.rnn(tag_inputs, word_inputs)
        if is_train:
            total_loss = []
            LF = []
            for decoder, a, r in zip(self.decoders, arc_targets, rel_targets):
                arc_accuracy, rel_accuracy, loss = decoder(top_recur, word_inputs, a, r, self.shared_arc_W)
                total_loss.append(loss)
                LF.append(rel_accuracy)
            return -1, sum(LF) / len(LF), nd.stack(*total_loss).mean()
        else:
            return [decoder(top_recur, word_inputs, None, None, self.shared_arc_W) for decoder in self.decoders]


class RefineParser(SharedRNNParser):
    def __init__(self, vocab: List[ParserVocabulary], word_dims, tag_dims, dropout_dim, lstm_layers, lstm_hiddens,
                 dropout_lstm_input, dropout_lstm_hidden, mlp_arc_size, mlp_rel_size, dropout_mlp, transfer=False,
                 bert=0, debug=False):
        super().__init__(vocab, word_dims, tag_dims, dropout_dim, lstm_layers, lstm_hiddens, dropout_lstm_input,
                         dropout_lstm_hidden, mlp_arc_size, mlp_rel_size, dropout_mlp, transfer, bert, debug)
        self.weights = parameter_init(self, 'score_weight', (len(vocab), len(vocab)), mx.init.One())
        self.arc_biaffines = nn.Sequential()
        for voc in vocab:
            self.arc_biaffines.add(UnlabeledBiAffine(mlp_arc_size))

    def dump(self, path: str):
        super().dump(path)
        for i, second_decoder in enumerate(self.arc_biaffines):
            second_decoder.save_parameters(os.path.join(path, 'second_decoder{}.bin'.format(i)))

    def fill(self, path):
        super().fill(path)
        for i, second_decoder in enumerate(self.arc_biaffines):
            sd_path = os.path.join(path, 'second_decoder{}.bin'.format(i))
            if os.path.isfile(sd_path):
                second_decoder.load_parameters(sd_path, ctx=mxnet_prefer_gpu())
                freeze(second_decoder)

    def forward(self, word_inputs, bert, tag_inputs, arc_targets=None, rel_targets=None):
        is_train = arc_targets is not None
        top_recur = self.rnn(tag_inputs, word_inputs, bert)
        mask = np.greater(word_inputs, self.decoders[0]._vocab.ROOT).astype(np.float32)

        if is_train:
            total_loss = []
            LF = []
            arc_score = []
            representations = []
            for mlp, unbiaffine, decoder, a, r in zip(self.mlps, self.arc_biaffines, self.decoders, arc_targets,
                                                      rel_targets):
                dep_arc, dep_rel, head_arc, head_rel = mlp(top_recur)
                arc_logits, arc_loss = unbiaffine(dep_arc, head_arc, mask, a)
                total_loss.append(arc_loss)
                arc_score.append(arc_logits)
                representations.append((dep_arc, dep_rel, head_arc, head_rel))

            scores = nd.stack(*arc_score)

            for (dep_arc, dep_rel, head_arc, head_rel), w, decoder, a, r in zip(representations, self.weights.data(),
                                                                                self.decoders, arc_targets,
                                                                                rel_targets):
                blend = nd.dot(w, scores).squeeze() / w.sum()
                arc_accuracy, rel_accuracy, loss = decoder(dep_arc, dep_rel, head_arc, head_rel, word_inputs, a, r,
                                                           blend)
                total_loss.append(loss)
                LF.append(rel_accuracy)

            if self.transfer:
                total_loss = [total_loss[len(self.mlps) - 1], total_loss[-1]]
                LF = LF[:-1]
            return -1, sum(LF) / len(LF), nd.stack(*total_loss).mean()
        else:
            arc_score = []
            representations = []
            outputs = []
            for mlp, unbiaffine, decoder in zip(self.mlps, self.arc_biaffines, self.decoders):
                dep_arc, dep_rel, head_arc, head_rel = mlp(top_recur)
                arc_logits = unbiaffine(dep_arc, head_arc, mask)
                arc_score.append(arc_logits)
                representations.append((dep_arc, dep_rel, head_arc, head_rel))

            scores = nd.stack(*arc_score)

            for (dep_arc, dep_rel, head_arc, head_rel), w, decoder in zip(representations, self.weights.data(),
                                                                          self.decoders):
                blend = nd.dot(w, scores).squeeze() / w.sum()
                outputs.append(decoder(dep_arc, dep_rel, head_arc, head_rel, word_inputs, None, None, blend))
            return outputs


class StairParser(SharedRNNParser):
    def __init__(self, vocab: List[ParserVocabulary], word_dims, tag_dims, dropout_dim, lstm_layers, lstm_hiddens,
                 dropout_lstm_input, dropout_lstm_hidden, mlp_arc_size, mlp_rel_size, dropout_mlp, debug=False):
        super().__init__(vocab, word_dims, tag_dims, dropout_dim, lstm_layers, lstm_hiddens, dropout_lstm_input,
                         dropout_lstm_hidden, mlp_arc_size, mlp_rel_size, dropout_mlp, debug)

    def create_graph(self, vocab, lstm_hiddens, mlp_arc_size, mlp_rel_size, dropout_mlp, debug):
        for idx, voc in enumerate(vocab):
            self.mlps.add(MLP(lstm_hiddens, mlp_arc_size, mlp_rel_size, dropout_mlp, debug))
            if idx != len(vocab) - 1:
                self.decoders.add(BiAffine(voc, mlp_arc_size, mlp_rel_size))
            else:
                self.decoders.add(BiAffine(voc, mlp_arc_size * len(vocab), mlp_rel_size))

    def forward(self, word_inputs, tag_inputs, arc_targets=None, rel_targets=None):
        is_train = arc_targets is not None
        top_recur = self.rnn(tag_inputs, word_inputs)
        das, has = [], []
        if is_train:
            total_loss = []
            LF = []
            for mlp, decoder, a, r in zip(self.mlps, self.decoders, arc_targets, rel_targets):
                dep_arc, dep_rel, head_arc, head_rel = mlp(top_recur)
                das.append(dep_arc)
                has.append(head_arc)
                if len(das) == len(self.decoders):
                    dep_arc = nd.concat(*das, dim=0)
                    head_arc = nd.concat(*has, dim=0)
                arc_accuracy, rel_accuracy, loss = decoder(dep_arc, dep_rel, head_arc, head_rel, word_inputs, a, r)
                total_loss.append(loss)
                LF.append(rel_accuracy)
            return -1, sum(LF) / len(LF), nd.stack(*total_loss).mean()
        else:
            outputs = []
            for mlp, decoder in zip(self.mlps, self.decoders):
                dep_arc, dep_rel, head_arc, head_rel = mlp(top_recur)
                das.append(dep_arc)
                has.append(head_arc)
                if len(das) == len(self.decoders):
                    dep_arc = nd.concat(*das, dim=0)
                    head_arc = nd.concat(*has, dim=0)
                outputs.append(decoder(dep_arc, dep_rel, head_arc, head_rel, word_inputs))
            return outputs
