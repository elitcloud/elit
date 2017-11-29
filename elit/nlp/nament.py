# ========================================================================
# Copyright 2017 Emory University
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

import numpy as np
from mxnet import gluon, nd

from elit.util.component import CONTEXT_WINDOWS, BILOU, WORD_VSM, LABEL_MAP, NGRAM_CONV, DROPOUT
from elit.nlp.util import argparse_train
from elit.nlp.model import NLPEval
from elit.nlp.state import NLPState, ForwardState

__author__ = 'Jinho D. Choi'


NER_GOLD = 'ner_gold'
NAMENT_GAZ = 'nament_gaz'


class NERState(ForwardState):
    def __init__(self, document, params):
        """
        NERState inherits the one-pass, left-to-right tagging algorithm.
        :param document: the input document
        :type document: list of elit.util.structure.Sentence
        :param params: a dictionary containing [label_map, word_vsm, nament_gaz, zero_output]
        :type params: dict
        """
        super().__init__(document, params, NER_GOLD)

        # parameters
        self.context_windows = params[CONTEXT_WINDOWS]  # list or tuple or range of int
        self.nament_gaz = params[NAMENT_GAZ]

    @property
    def x(self):
        word_emb = self.word_emb[self.sen_id]
        scores = self.output[self.sen_id]

        p = [self.x_position(i) for i in self.context_windows]
        w = [self.x_extract(i, word_emb, self.word_vsm.zero) for i in self.context_windows]
        s = [self.x_extract(i, scores, self.zero_output) for i in self.context_windows]

        return np.column_stack((p, w, s))


class NERModel(gluon.Block):
    def __init__(self, params):
        super(NERModel, self).__init__()
        word_dim = params[WORD_VSM].dim
        num_class = len(params[LABEL_MAP])
        ngram_conv = params[NGRAM_CONV]
        dropout = params[DROPOUT]

        with self.name_scope():
            k = len(NLPState.X_LST) + word_dim + num_class
            self.ngram_conv = []
            for i, f in enumerate(ngram_conv):
                name = 'ngram_conv_' + str(i)
                setattr(self, name,
                        gluon.nn.Conv2D(channels=f, kernel_size=(i + 1, k), strides=(1, k), activation='relu'))
                self.ngram_conv.append(getattr(self, name))

            self.dropout = gluon.nn.Dropout(dropout)
            self.out = gluon.nn.Dense(num_class)

    def forward(self, x):
        # ngram convolutions
        t = [conv(x).reshape((0, -1)) for conv in self.ngram_conv]
        x = nd.concat(*t, dim=1)
        x = self.dropout(x)

        # output layer
        x = self.out(x)
        return x


class NEREval(NLPEval):
    def __init__(self):
        """
        NEREval measures the F1 output of named entity recognition.
        """
        self.correct = 0
        self.p_total = 0
        self.r_total = 0

    def update(self, state):
        get = state.label_map.score

        for i, sentence in enumerate(state.document):
            gold = [get(s) for s in sentence[NER_GOLD]]
            auto = [get(s) for s in np.argmax(state.scores[i], axis=1)]
            # BILOU.quick_fix(auto)

            gold = BILOU.collect(gold)
            auto = BILOU.collect(auto)

            self.p_total += len(gold)
            self.r_total += len(auto)

            for k, g in gold:
                if g == auto.score():
                    self.correct += 1

    def score(self):
        p = 100.0 * self.correct / self.p_total
        r = 100.0 * self.correct / self.r_total
        return 2 * p * r / (p + r)

    def reset(self):
        self.correct = self.p_total = self.r_total = 0


def args_train():
    parser = argparse_train('Train: named entity recognition')

    # lexicon
    parser.add_argument('-wv', '--word_vsm', type=str, metavar='filepath', help='vector space model for word embeddings')
    parser.add_argument('-cw', '--context_windows', type=lambda x: tuple(map(int, x.split(','))), metavar='int[,int]*', default=(-2, -1, 0, 1, 2), help='context window for feature extraction')

    # train
    parser.add_argument('-lr', '--learning_rate', type=float, metavar='float', default=0.01, help='learning rate')
    parser.add_argument('-nc', '--ngram_conv', type=lambda x: tuple(map(int, x.split(','))), metavar='int[,int]*', default=(128, 128, 128, 128, 128), help='list of filter numbers for n-gram convolutions')
    parser.add_argument('-do', '--dropout', type=float, metavar='float', default=0.2, help='dropout')

    return parser.parse_args()










# class ConvBlock(gluon.Block):
#     def __init__(self):
#         super(ConvBlock, self).__init__()
#         with self.name_scope():
#             self.fc0 = gluon.nn.Dense(5)
#             self.fc1 = gluon.nn.Dense(5)
#             self.fc2 = gluon.nn.Dense(5)
#             self.fcn = [self.fc0, self.fc1, self.fc2]
#
#     def forward(self, x):
#         t = [fc(x) for fc in self.fcn]
#         x = nd.concat(*t, dim=0)
#         return x
#
# X = nd.array([
#     [[1,0,0,0,0], [2,0,0,0,0], [3,0,0,0,0], [4,0,0,0,0]],
#     [[0,1,0,0,0], [0,2,0,0,0], [0,3,0,0,0], [0,4,0,0,0]],
#     [[0,0,1,0,0], [0,0,2,0,0], [0,0,3,0,0], [0,0,4,0,0]],
#     [[0,0,0,1,0], [0,0,0,2,0], [0,0,0,3,0], [0,0,0,4,0]],
#     [[0,0,0,0,1], [0,0,0,0,2], [0,0,0,0,3], [0,0,0,0,4]]
# ])
#
# Y = nd.array([[0,1,2],[0,1,2],[0,1,2],[0,-1,-1],[0,1,-1]])
#
# ctx = mx.cpu()
# net = ConvBlock()
# net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
# trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
#
# batch_size = 2
# loss_func = MultiLabelSoftmaxCrossEntropyLoss()
# data = gluon.data.DataLoader(gluon.data.ArrayDataset(X, Y), batch_size=batch_size)
#
# for x, y in data:
#     x = x.as_in_context(ctx)
#     y = y.as_in_context(ctx)
#     y = nd.transpose(y).reshape((-1,1))
#     with autograd.record():
#         output = net(x)
#         print(output)
#         print(y)
#         loss = loss_func(output, y)
#         loss.backward()
#
#
#     trainer.step(x.shape[0])
#
#
# @property
#     def x(self):
#         """
#         If the total number of words in the document is greater than max_len, the rest gets discarded.
#         :param document: a sentence or a list of sentences, where each sentence is represented by a dictionary.
#         :type document: Union[list of dict, dict]
#         :return: a list of feature vectors for the corresponding words across sentences.
#         :rtype: list of numpy.array -> max_len * (word_vsm.dim + ambi_vsm.dim + num_class)
#         """
#         def position(i, size):
#             return self.fst_word if i == 0 else self.lst_word if i+1 == size else self.mid_word
#
#         def aux(sentence):
#             word_emb = sentence.setdefault(WORD_EMB, self.word_vsm.get_list(sentence[TOKEN]))
#             ambi_emb = sentence.setdefault(AMBI_EMB, self.ambi_vsm.get_list(sentence[TOKEN]))
#             pos_scores = sentence.get(POS_OUT, None)
#
#             return [np.concatenate((
#                 position(i, len(sentence)),
#                 word_emb[i],
#                 ambi_emb[i],
#                 self.zero_scores if pos_scores is None else pos_scores[i]))
#                 for i in range(len(sentence))]
#
#         matrix = [v for s in document for v in aux(s)]
#
#         # zero padding
#         if len(matrix) < self.max_len:
#             matrix.extend([np.zeros(len(matrix[0]))] * (self.max_len - len(matrix)))
#         elif len(matrix) > self.max_len:
#             matrix = matrix[:self.max_len]
#
#         return np.array(matrix)
#
#     def gold_labels(self, document):
#         labels = np.concatenate([d[POS_GOLD] for d in document])
#
#         # zero padding
#         if len(labels) < self.max_len:
#             return np.append(labels, np.full(self.max_len - len(labels), -1))
#         elif len(labels) > self.max_len:
#             return labels[:self.max_len]
#         else:
#             return labels
#
#     def set_scores(self, document, output):
#         """
#         :param document: a sentence or a list of sentences, where each sentence is represented by a dictionary.
#         :type document: Union[list of dict, dict]
#         :param output: the output of the part-of-speech tag predictions.
#         :param output: numpy.array -> max_len * num_class
#         """
#         def index(i):
#             return (begin + i) * self.num_class
#
#         def get(sentence):
#             return [output[index(i):index(i+1)] for i in range(0, len(sentence))]
#
#         begin = 0
#
#         if isinstance(document, dict):
#             document[POS_OUT] = get(document)
#         else:
#             for d in document:
#                 sc = get(d)
#                 d[POS_OUT] = sc
#                 begin += sc
#
# def trunc_pads(labels, output):
#     idx = next((i for i, label in enumerate(labels) if label.asscalar() == -1), None)
#     return (labels[:idx], output[:idx]) if idx else (labels, output)
