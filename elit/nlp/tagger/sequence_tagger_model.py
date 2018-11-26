# ========================================================================
# Copyright 2018 ELIT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========================================================================
# -*- coding:utf-8 -*-
# Author：ported from PyTorch implementation of flair: https://github.com/zalandoresearch/flair to MXNet
# Date: 2018-09-21 21:10
import os
import pickle
from typing import List, Tuple, Union

import mxnet as mx
import mxnet.ndarray as nd
from mxnet import autograd
from mxnet.gluon import nn, rnn

from elit.nlp.tagger.corpus import Dictionary, Sentence, Token
from elit.nlp.tagger.embeddings import TokenEmbeddings, StackedEmbeddings, CharLMEmbeddings, WordEmbeddings
from elit.nlp.tagger.mxnet_util import mxnet_prefer_gpu

START_TAG = '<START>'
STOP_TAG = '<STOP>'


def to_scalar(var: nd.NDArray):
    # return var.view(-1).data.tolist()[0]
    return var[0]


def argmax(vec: nd.NDArray):
    # _, idx = torch.max(vec, 1)
    # return to_scalar(idx)
    return vec.argmax(1)


def log_sum_exp(vec: nd.NDArray):
    max_score = vec[0, argmax(vec)]
    # max_score_broadcast = max_score.reshape(1, -1).expand(1, vec.size()[1])
    return max_score + nd.log(nd.sum(nd.exp(vec - max_score)))


def argmax_batch(vecs: nd.NDArray):
    # _, idx = torch.max(vecs, 1)
    # return idx
    return vecs.argmax(axis=1)


def log_sum_exp_batch(vecs):
    maxi = nd.max(vecs, 1)
    maxi_bc = maxi.expand_dims(1).tile((1, vecs.shape[1]))
    recti_ = nd.log(nd.sum(nd.exp(vecs - maxi_bc), 1))
    return maxi + recti_


def pad_tensors(tensor_list, type_=nd.NDArray):
    ml = max([x.shape[0] for x in tensor_list])
    shape = [len(tensor_list), ml] + list(tensor_list[0].shape[1:])
    template = type_(*shape)
    template.fill_(0)
    lens_ = [x.shape[0] for x in tensor_list]
    for i, tensor in enumerate(tensor_list):
        template[i, :lens_[i]] = tensor

    return template, lens_


class SequenceTagger(nn.Block):

    def __init__(self,
                 hidden_size: int,
                 embeddings: TokenEmbeddings,
                 tag_dictionary: Dictionary,
                 tag_type: str,
                 use_crf: bool = True,
                 use_rnn: bool = True,
                 rnn_layers: int = 1):

        super(SequenceTagger, self).__init__()

        self.use_rnn = use_rnn
        self.hidden_size = hidden_size
        self.use_crf = use_crf
        self.rnn_layers = rnn_layers

        self.trained_epochs = 0

        self.embeddings = embeddings

        # set the dictionaries
        self.tag_dictionary = tag_dictionary
        self.tag_type = tag_type
        self.tagset_size = len(tag_dictionary)

        # initialize the network architecture
        self.nlayers = rnn_layers
        self.hidden_word = None

        self.dropout = nn.Dropout(0.5, axes=[0])

        # self.dropout: nn.Block = LockedDropout(0.5)

        rnn_input_dim = self.embeddings.embedding_length

        self.relearn_embeddings = True

        if self.relearn_embeddings:
            self.embedding2nn = nn.Dense(in_units=rnn_input_dim, units=rnn_input_dim, flatten=False)

        # bidirectional LSTM on top of embedding layer
        self.rnn_type = 'LSTM'
        # if self.rnn_type in ['LSTM', 'GRU']:
        #
        #     if self.nlayers == 1:
        #         self.rnn = getattr(rnn, self.rnn_type)(rnn_input_dim, hidden_size,
        #                                               num_layers=self.nlayers,
        #                                               bidirectional=True)
        #     else:
        #         self.rnn = getattr(rnn, self.rnn_type)(rnn_input_dim, hidden_size,
        #                                               num_layers=self.nlayers,
        #                                               dropout=0.5,
        #                                               bidirectional=True)
        self.rnn = rnn.LSTM(input_size=rnn_input_dim, hidden_size=hidden_size, num_layers=self.nlayers,
                            bidirectional=True)

        # self.nonlinearity = nn.Tanh()

        # final linear map to tag space
        if self.use_rnn:
            self.linear = nn.Dense(in_units=hidden_size * 2, units=len(tag_dictionary), flatten=False)
        else:
            self.linear = nn.Dense(in_units=self.embeddings.embedding_length, units=len(tag_dictionary), flatten=False)

        if self.use_crf:
            transitions = nd.random.normal(0, 1, (self.tagset_size, self.tagset_size))
            transitions[self.tag_dictionary.get_idx_for_item(START_TAG), :] = -10000
            transitions[:, self.tag_dictionary.get_idx_for_item(STOP_TAG)] = -10000
            self.transitions = self.params.get('transitions', shape=(self.tagset_size, self.tagset_size),
                                               init=mx.init.Constant(transitions))

    def save(self, model_folder: str):
        os.makedirs(model_folder, exist_ok=True)
        config = {
            # 'embeddings': self.embeddings,
            'hidden_size': self.hidden_size,
            'tag_dictionary': self.tag_dictionary,
            'tag_type': self.tag_type,
            'use_crf': self.use_crf,
            'use_rnn': self.use_rnn,
            'rnn_layers': self.rnn_layers,
        }
        config_path = os.path.join(model_folder, 'config.pkl')
        with open(config_path, 'wb') as f:
            pickle.dump(config, f)
            model_path = os.path.join(model_folder, 'model.bin')
            self.save_parameters(model_path)

    @classmethod
    def load_from_file(cls, model_folder, context: mx.Context = None, **kwargs):
        if context is None:
            context = mxnet_prefer_gpu()
        config_path = os.path.join(model_folder, 'config.pkl')
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
            with context:
                embedding_types = [

                    WordEmbeddings(
                        '{}data/embedding/fasttext100.vec.txt'.format(kwargs.get('word_embedding_path', ''))),

                    # comment in this line to use character embeddings
                    # CharacterEmbeddings(),

                    # comment in these lines to use contextual string embeddings
                    CharLMEmbeddings('{}data/model/lm-news-forward'.format(kwargs.get('word_embedding_path', '')),
                                     context=context),
                    CharLMEmbeddings('{}data/model/lm-news-backward'.format(kwargs.get('word_embedding_path', '')),
                                     context=context),
                ]

                embeddings = StackedEmbeddings(embeddings=embedding_types)
                model = SequenceTagger(
                    hidden_size=config['hidden_size'],
                    embeddings=embeddings,
                    tag_dictionary=config['tag_dictionary'],
                    tag_type=config['tag_type'],
                    use_crf=config['use_crf'],
                    use_rnn=config['use_rnn'],
                    rnn_layers=config['rnn_layers'])
                model.load_parameters(os.path.join(model_folder, 'model.bin'), ctx=context)
            return model

    def forward(self, sentences: List[Sentence]) -> Tuple[nd.NDArray, nd.NDArray, List]:
        """

        :param sentences:
        :return: features, tags, lengths
        """
        # first, sort sentences by number of tokens
        sentences.sort(key=lambda x: len(x), reverse=True)
        longest_token_sequence_in_batch = len(sentences[0])

        self.embeddings.embed(sentences)

        all_sentence_tensors = []
        lengths = []
        tag_list = []

        padding = nd.zeros((1, self.embeddings.embedding_length), dtype='float32')

        for sentence in sentences:

            # get the tags in this sentence
            tag_idx = []

            lengths.append(len(sentence.tokens))

            word_embeddings = []

            for token in sentence:
                # get the tag
                tag_idx.append(self.tag_dictionary.get_idx_for_item(token.get_tag(self.tag_type)))
                # get the word embeddings
                word_embeddings.append(token.get_embedding().reshape((1, -1)))

            # pad shorter sentences out
            for add in range(longest_token_sequence_in_batch - len(sentence.tokens)):
                word_embeddings.append(padding)

            word_embeddings_tensor = nd.concat(*word_embeddings, dim=0)

            # if torch.cuda.is_available():
            #     tag_list.append(torch.cuda.LongTensor(tag_idx))
            # else:
            tag_list.append(nd.array(tag_idx))

            all_sentence_tensors.append(word_embeddings_tensor.expand_dims(1))

        # padded tensor for entire batch
        sentence_tensor = nd.concat(*all_sentence_tensors, dim=1)  # (IN, NN, C)
        # if torch.cuda.is_available():
        #     sentence_tensor = sentence_tensor.cuda()

        # --------------------------------------------------------------------
        # FF PART
        # --------------------------------------------------------------------
        sentence_tensor = self.dropout(sentence_tensor)

        if self.relearn_embeddings:
            sentence_tensor = self.embedding2nn(sentence_tensor)

        if self.use_rnn:
            # packed = torch.nn.utils.rnn.pack_padded_sequence(sentence_tensor, lengths)

            sentence_tensor = self.rnn(sentence_tensor)

            # sentence_tensor, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(rnn_output)

            sentence_tensor = self.dropout(sentence_tensor)

        features = self.linear(sentence_tensor)
        tags = nd.zeros((len(tag_list), lengths[0]), dtype='int32')
        for i, (t, l) in enumerate(zip(tag_list, lengths)):
            tags[i, :l] = t
        return features.transpose([1, 0, 2]), tags, lengths

        # predictions_list = []
        # for sentence_no, length in enumerate(lengths):
        #     sentence_predictions = []
        #     for token_no in range(length):
        #         sentence_predictions.append(
        #             features[token_no, sentence_no, :].expand_dims(0)
        #         )
        #     predictions_list.append(nd.concat(*sentence_predictions, dim=0))
        #
        # return predictions_list, tag_list

    def _score_sentence(self, feats, tags, lens_):

        start = nd.array([
            self.tag_dictionary.get_idx_for_item(START_TAG)
        ], dtype='int32')
        start = start.expand_dims(0).tile((tags.shape[0], 1))

        stop = nd.array([
            self.tag_dictionary.get_idx_for_item(STOP_TAG)
        ], dtype='int32')

        stop = stop.expand_dims(0).tile((tags.shape[0], 1))

        pad_start_tags = nd.concat(*[start, tags], dim=1)
        pad_stop_tags = nd.concat(*[tags, stop], dim=1)

        for i in range(len(lens_)):
            pad_stop_tags[i, lens_[i]:] = \
                self.tag_dictionary.get_idx_for_item(STOP_TAG)

        score = []

        for i in range(feats.shape[0]):
            r = nd.array(list(range(lens_[i])), dtype='int32')

            score.append(nd.sum(
                self.transitions.data()[pad_stop_tags[i, :lens_[i] + 1], pad_start_tags[i, :lens_[i] + 1]]
            ) + nd.sum(feats[i, r, tags[i, :lens_[i]]]))

        return nd.stack(*score).squeeze()

    def viterbi_decode(self, feats):
        backpointers = []
        init_vvars = nd.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_dictionary.get_idx_for_item(START_TAG)] = 0
        forward_var = init_vvars

        for feat in feats:
            next_tag_var = forward_var.reshape((1, -1)).tile((self.tagset_size, 1)) + self.transitions.data()
            bptrs_t = nd.argmax(next_tag_var, axis=1)
            viterbivars_t = next_tag_var[list(range(len(bptrs_t))), bptrs_t]
            forward_var = viterbivars_t + feat
            backpointers.append(bptrs_t)

        terminal_var = forward_var + self.transitions.data()[self.tag_dictionary.get_idx_for_item(STOP_TAG)]
        terminal_var[self.tag_dictionary.get_idx_for_item(STOP_TAG)] = -10000.
        terminal_var[self.tag_dictionary.get_idx_for_item(START_TAG)] = -10000.
        best_tag_id = int(terminal_var.argmax(axis=0).asscalar())
        path_score = terminal_var[best_tag_id]
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(int(best_tag_id.asscalar()))
        start = best_path.pop()
        assert start == self.tag_dictionary.get_idx_for_item(START_TAG)
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentences: List[Sentence], tag_type: str):
        feats, tags, lens_ = self.forward(sentences)

        # feats, lens_ = pad_tensors(feats)
        # tags, _ = pad_tensors(tags, torch.LongTensor)

        if self.use_crf:

            forward_score = self._forward_alg(feats, lens_)
            gold_score = self._score_sentence(feats, tags, lens_)

            score = forward_score - gold_score

            return score.sum()

        else:

            score = 0
            for i in range(len(feats)):
                sentence_feats = feats[i]
                sentence_tags = tags[i]

                tag_tensor = nd.array(sentence_tags)
                score += nn.functional.cross_entropy(sentence_feats, tag_tensor)

            return score

    def _forward_alg(self, feats, lens_):

        batch_size = feats.shape[0]
        tagset_size = feats.shape[2]
        length = feats.shape[1]

        init_alphas = nd.full((self.tagset_size,), -10000.)
        init_alphas[self.tag_dictionary.get_idx_for_item(START_TAG)] = 0.

        forward_var_list = [init_alphas.tile((feats.shape[0], 1))]
        transitions = self.transitions.data().expand_dims(0).tile((feats.shape[0], 1, 1))

        for i in range(feats.shape[1]):
            emit_score = feats[:, i, :]

            tag_var = \
                emit_score.expand_dims(2).tile((1, 1, transitions.shape[2])) + \
                transitions + \
                forward_var_list[i].expand_dims(2).tile((1, 1, transitions.shape[2])).transpose([0, 2, 1])

            max_tag_var = nd.max(tag_var, axis=2)

            new_tag_var = tag_var - max_tag_var.expand_dims(2).tile((1, 1, transitions.shape[2]))

            agg_ = nd.log(nd.sum(nd.exp(new_tag_var), axis=2))

            forward_var_list.append(nd.full((feats.shape[0], feats.shape[2]), val=max_tag_var + agg_))

            # cloned = forward_var.clone()
            # forward_var[:, i + 1, :] = max_tag_var + agg_

            # forward_var = cloned

        forward_var = nd.stack(*forward_var_list)[lens_, nd.array(list(range(feats.shape[0])), dtype='int32'), :]

        terminal_var = forward_var + \
                       self.transitions.data()[self.tag_dictionary.get_idx_for_item(STOP_TAG)].expand_dims(0).tile((
                           forward_var.shape[0], 1))

        alpha = log_sum_exp_batch(terminal_var)

        return alpha

    def predict_scores(self, sentence: Sentence):
        feats, tags, lengths = self.forward([sentence])
        feats = feats[0]
        tags = tags[0]
        if self.use_crf:
            score, tag_seq = self.viterbi_decode(feats)
        else:
            score, tag_seq = nd.max(feats, 1)
            tag_seq = list(tag_seq.cpu().data)

        return score, tag_seq

    def predict(self, sentences: Union[List[Sentence], Sentence], mini_batch_size=32) -> List[Sentence]:

        if type(sentences) is Sentence:
            sentences = [sentences]

        # remove previous embeddings
        for sentence in sentences:
            sentence.clear_embeddings(also_clear_word_embeddings=True)

        # make mini-batches
        batches = [sentences[x:x + mini_batch_size] for x in range(0, len(sentences), mini_batch_size)]

        for batch in batches:
            score, tag_seq = self._predict_scores_batch(batch)
            predicted_id = tag_seq
            all_tokens = []
            for sentence in batch:
                all_tokens.extend(sentence.tokens)

            for (token, pred_id) in zip(all_tokens, predicted_id):
                token = token
                # get the predicted tag
                predicted_tag = self.tag_dictionary.get_item_for_index(pred_id)
                token.add_tag(self.tag_type, predicted_tag)

        return sentences

    def _predict_scores_batch(self, sentences: List[Sentence]):
        all_feats, tags, lengths = self.forward(sentences)

        overall_score = 0
        all_tags_seqs = []

        for feats in all_feats:
            # viterbi to get tag_seq
            if self.use_crf:
                score, tag_seq = self.viterbi_decode(feats)
            else:
                score, tag_seq = nd.max(feats, 1)
                tag_seq = list(tag_seq.data())

            # overall_score += score
            all_tags_seqs.extend(tag_seq)

        return overall_score, all_tags_seqs

    @staticmethod
    def load(model: str):
        tagger = SequenceTagger.load_from_file(model)
        return tagger


class LockedDropout(nn.Block):
    def __init__(self, dropout_rate=0.5):
        super(LockedDropout, self).__init__()
        self.dropout_rate = dropout_rate

    def forward(self, x: nd.NDArray):
        if not autograd.is_training() or not self.dropout_rate:
            return x

        keep_rate = 1. - self.dropout_rate
        m = nd.random.negative_binomial(1, keep_rate, (1, x.shape[1], x.shape[2]))
        mask = m / keep_rate
        return nd.broadcast_mul(x, mask)


if __name__ == '__main__':
    tagger = SequenceTagger.load_from_file('data/model/ner/eng')
    sent = Sentence()
    sent.add_token(Token('the', pos='DT'))
    sent.add_token(Token('European', pos='NNP'))
    sent.add_token(Token('Union', pos='NNP'))
    result = tagger.predict(sent)[0]
    print([t.text + '/' + t.tags['ner'] for t in result])
