# -*- coding: UTF-8 -*-
# Adopted from https://github.com/jcyk/Dynet-Biaffine-dependency-parser
# With some modifications, added the char lstm layer
import os
import sys

from elit.dep.common.utils import stdchannel_redirected
import numpy as np
from elit.dep.parser.common.utils import biLSTM, leaky_relu, bilinear, orthonormal_initializer, arc_argmax, rel_argmax, \
    orthonormal_VanillaLSTMBuilder, attention, LSTM, one_hot

with stdchannel_redirected(sys.stderr, os.devnull):
    import dynet as dy


class BiaffineParser(object):
    def __init__(self, vocab,
                 char_dims,
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
        pc = dy.ParameterCollection()
        self.pret_word_embs = pc.lookup_parameters_from_numpy(
            vocab.get_pret_embs(word_dims)) if vocab.has_pret_embs() else None
        if self.pret_word_embs:
            # now create a subset of parameters, which will be saved and loaded, so that pre-trained embeddings are
            # excluded
            pc = pc.add_subcollection('savable')
        self._vocab = vocab
        self.word_embs = pc.lookup_parameters_from_numpy(vocab.get_word_embs(word_dims))
        self.word_dims = word_dims
        if char_dims > 0:
            self.char_embs = pc.lookup_parameters_from_numpy(vocab.get_char_embs(char_dims))
            self.char_lstm = orthonormal_VanillaLSTMBuilder(1, char_dims, word_dims // 2, pc, debug)
            self.char_w = pc.add_parameters((word_dims // 2,), init=dy.ConstInitializer(0.))
        else:
            self.char_lstm = None
        self.tag_embs = pc.lookup_parameters_from_numpy(vocab.get_tag_embs(tag_dims))

        self.LSTM_builders = []
        f = orthonormal_VanillaLSTMBuilder(1, word_dims + tag_dims, lstm_hiddens, pc, debug)
        b = orthonormal_VanillaLSTMBuilder(1, word_dims + tag_dims, lstm_hiddens, pc, debug)
        self.LSTM_builders.append((f, b))
        for i in range(lstm_layers - 1):
            f = orthonormal_VanillaLSTMBuilder(1, 2 * lstm_hiddens, lstm_hiddens, pc, debug)
            b = orthonormal_VanillaLSTMBuilder(1, 2 * lstm_hiddens, lstm_hiddens, pc, debug)
            self.LSTM_builders.append((f, b))
        self.dropout_lstm_input = dropout_lstm_input
        self.dropout_lstm_hidden = dropout_lstm_hidden

        mlp_size = mlp_arc_size + mlp_rel_size
        W = orthonormal_initializer(mlp_size, 2 * lstm_hiddens, debug)
        self.mlp_dep_W = pc.parameters_from_numpy(W)
        self.mlp_head_W = pc.parameters_from_numpy(W)
        self.mlp_dep_b = pc.add_parameters((mlp_size,), init=dy.ConstInitializer(0.))
        self.mlp_head_b = pc.add_parameters((mlp_size,), init=dy.ConstInitializer(0.))
        self.mlp_arc_size = mlp_arc_size
        self.mlp_rel_size = mlp_rel_size
        self.dropout_mlp = dropout_mlp

        self.arc_W = pc.add_parameters((mlp_arc_size, mlp_arc_size + 1), init=dy.ConstInitializer(0.))
        self.rel_W = pc.add_parameters((vocab.rel_size * (mlp_rel_size + 1), mlp_rel_size + 1),
                                       init=dy.ConstInitializer(0.))

        self._pc = pc

        def _emb_mask_generator(seq_len, batch_size):
            ret = []
            for i in range(seq_len):
                word_mask = np.random.binomial(1, 1. - dropout_dim, batch_size).astype(np.float32)
                tag_mask = np.random.binomial(1, 1. - dropout_dim, batch_size).astype(np.float32)
                scale = 3. / (2. * word_mask + tag_mask + 1e-12)
                word_mask *= scale
                tag_mask *= scale
                word_mask = dy.inputTensor(word_mask, batched=True)
                tag_mask = dy.inputTensor(tag_mask, batched=True)
                ret.append((word_mask, tag_mask))
            return ret

        self.generate_emb_mask = _emb_mask_generator

    @property
    def parameter_collection(self):
        return self._pc

    def run(self, char_vocab, cased_word_inputs, word_inputs, tag_inputs, arc_targets=None, rel_targets=None,
            is_train=True):
        """
        Train or test
        :param char_vocab:
        :param cased_word_inputs: seq_len x batch_size
        :param word_inputs: seq_len x batch_size
        :param tag_inputs: seq_len x batch_size
        :param arc_targets: seq_len x batch_size
        :param rel_targets: seq_len x batch_size
        :param is_train: is training or test
        :return:
        """

        def flatten_numpy(ndarray):
            """
            Flatten nd-array to 1-d column vector
            :param ndarray:
            :return:
            """
            return np.reshape(ndarray, (-1,), 'F')

        batch_size = word_inputs.shape[1]
        seq_len = word_inputs.shape[0]
        mask = np.greater(word_inputs, self._vocab.ROOT).astype(np.float32)
        num_tokens = int(np.sum(mask))  # non padding, non root token number

        if is_train or arc_targets is not None:
            mask_1D = flatten_numpy(mask)
            mask_1D_tensor = dy.inputTensor(mask_1D, batched=True)
            #  if batched=True, the last dimension is used as a batch dimension if arr is a list of numpy ndarrays

        if self.char_lstm:
            # Subword model
            char_w = dy.parameter(self.char_w)

            def LSTM_attention(lstm, inputs, dropout_x=0., dropout_h=0.):
                ss = LSTM(lstm, inputs, None, dropout_x, dropout_h)
                hs = [s.h()[0] for s in ss]
                return dy.concatenate([attention(hs, char_w), ss[-1].s()[0]])

            subword_embs = []
            for char_ids in char_vocab:
                char_inputs = [dy.lookup(self.char_embs, char) for char in char_ids]
                subword_embs.append(
                    LSTM_attention(self.char_lstm, char_inputs, self.dropout_lstm_input if is_train else 0.,
                                   self.dropout_lstm_hidden if is_train else 0.))
            subword_embs = dy.concatenate_cols(subword_embs)

            word_embs = [dy.lookup_batch(self.word_embs, np.where(w < self._vocab.words_in_train, w, self._vocab.UNK)) +
                         subword_embs * dy.inputTensor(one_hot(cw, len(char_vocab)).T, batched=True) +
                         0 if self.pret_word_embs is None else dy.lookup_batch(self.pret_word_embs, w, update=False)
                         for cw, w in zip(cased_word_inputs, word_inputs)]
        else:
            word_embs = [dy.lookup_batch(self.word_embs, np.where(w < self._vocab.words_in_train, w, self._vocab.UNK))
                         + 0 if self.pret_word_embs is None else dy.lookup_batch(self.pret_word_embs, w, update=False)
                         for w
                         in word_inputs]

        tag_embs = [dy.lookup_batch(self.tag_embs, pos) for pos in tag_inputs]

        # Dropout
        if is_train:
            emb_masks = self.generate_emb_mask(seq_len, batch_size)
            emb_inputs = [dy.concatenate([dy.cmult(w, wm), dy.cmult(pos, posm)]) for w, pos, (wm, posm) in
                          zip(word_embs, tag_embs, emb_masks)]
        else:
            emb_inputs = [dy.concatenate([w, pos]) for w, pos in zip(word_embs, tag_embs)]  # seq_len x batch_size

        top_recur = dy.concatenate_cols(
            biLSTM(self.LSTM_builders, emb_inputs, batch_size, self.dropout_lstm_input if is_train else 0.,
                   self.dropout_lstm_hidden if is_train else 0.))
        if is_train:
            top_recur = dy.dropout_dim(top_recur, 1, self.dropout_mlp)

        W_dep, b_dep = dy.parameter(self.mlp_dep_W), dy.parameter(self.mlp_dep_b)
        W_head, b_head = dy.parameter(self.mlp_head_W), dy.parameter(self.mlp_head_b)
        dep, head = leaky_relu(dy.affine_transform([b_dep, W_dep, top_recur])), leaky_relu(
            dy.affine_transform([b_head, W_head, top_recur]))
        if is_train:
            dep, head = dy.dropout_dim(dep, 1, self.dropout_mlp), dy.dropout_dim(head, 1, self.dropout_mlp)

        dep_arc, dep_rel = dep[:self.mlp_arc_size], dep[self.mlp_arc_size:]
        head_arc, head_rel = head[:self.mlp_arc_size], head[self.mlp_arc_size:]

        W_arc = dy.parameter(self.arc_W)
        arc_logits = bilinear(dep_arc, W_arc, head_arc, self.mlp_arc_size, seq_len, batch_size, num_outputs=1,
                              bias_x=True, bias_y=False)
        # (#head x #dep) x batch_size

        flat_arc_logits = dy.reshape(arc_logits, (seq_len,), seq_len * batch_size)
        # (#head ) x (#dep x batch_size)

        arc_preds = arc_logits.npvalue().argmax(0)
        if len(arc_preds.shape) == 1:  # dynet did unnecessary jobs
            arc_preds = np.expand_dims(arc_preds, axis=1)
        # seq_len x batch_size

        if is_train or arc_targets is not None:
            arc_correct = np.equal(arc_preds, arc_targets).astype(np.float32) * mask
            arc_accuracy = np.sum(arc_correct) / num_tokens
            targets_1D = flatten_numpy(arc_targets)
            losses = dy.pickneglogsoftmax_batch(flat_arc_logits, targets_1D)
            arc_loss = dy.sum_batches(losses * mask_1D_tensor) / num_tokens

        if not is_train:
            arc_probs = np.transpose(
                np.reshape(dy.softmax(flat_arc_logits).npvalue(), (seq_len, seq_len, batch_size), 'F'))
        # #batch_size x #dep x #head

        W_rel = dy.parameter(self.rel_W)
        # dep_rel = dy.concatenate([dep_rel, dy.inputTensor(np.ones((1, seq_len),dtype=np.float32))])
        # head_rel = dy.concatenate([head_rel, dy.inputTensor(np.ones((1, seq_len), dtype=np.float32))])
        rel_logits = bilinear(dep_rel, W_rel, head_rel, self.mlp_rel_size, seq_len, batch_size,
                              num_outputs=self._vocab.rel_size, bias_x=True, bias_y=True)
        # (#head x rel_size x #dep) x batch_size

        flat_rel_logits = dy.reshape(rel_logits, (seq_len, self._vocab.rel_size), seq_len * batch_size)
        # (#head x rel_size) x (#dep x batch_size)

        partial_rel_logits = dy.pick_batch(flat_rel_logits, targets_1D if is_train else flatten_numpy(arc_preds))
        # (rel_size) x (#dep x batch_size)

        if is_train or arc_targets is not None:
            rel_preds = partial_rel_logits.npvalue().argmax(0)
            targets_1D = flatten_numpy(rel_targets)
            rel_correct = np.equal(rel_preds, targets_1D).astype(np.float32) * mask_1D
            rel_accuracy = np.sum(rel_correct) / num_tokens
            losses = dy.pickneglogsoftmax_batch(partial_rel_logits, targets_1D)
            rel_loss = dy.sum_batches(losses * mask_1D_tensor) / num_tokens

        if not is_train:
            rel_probs = np.transpose(np.reshape(dy.softmax(dy.transpose(flat_rel_logits)).npvalue(),
                                                (self._vocab.rel_size, seq_len, seq_len, batch_size), 'F'))
        # batch_size x #dep x #head x #nclasses

        if is_train or arc_targets is not None:
            loss = arc_loss + rel_loss
            correct = rel_correct * flatten_numpy(arc_correct)
            overall_accuracy = np.sum(correct) / num_tokens

        if is_train:
            return arc_accuracy * 100., rel_accuracy * 100., overall_accuracy * 100., loss

        outputs = []

        for msk, arc_prob, rel_prob in zip(np.transpose(mask), arc_probs, rel_probs):
            # parse sentences one by one
            msk[0] = 1.
            sent_len = int(np.sum(msk))
            arc_pred = arc_argmax(arc_prob, sent_len, msk)
            rel_prob = rel_prob[np.arange(len(arc_pred)), arc_pred]
            rel_pred = rel_argmax(rel_prob, sent_len)
            outputs.append((arc_pred[1:sent_len], rel_pred[1:sent_len]))

        if arc_targets is not None:
            return arc_accuracy * 100., rel_accuracy * 100., overall_accuracy * 100., outputs
        return outputs

    def save(self, save_path):
        self._pc.save(save_path)

    def load(self, load_path):
        self._pc.populate(load_path)
