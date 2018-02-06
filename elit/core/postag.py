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
import argparse
import glob
import logging
import random

import mxnet as mx
import numpy as np
from mxnet import gluon, nd

from elit.util.component import NLPState, epoch
from elit.util.lexicon import LabelMap, FastText, Word2Vec
from elit.util.structure import Sentence, group_sentences, POS_OUT
from elit.util.structure import TOKENS

__author__ = 'Jinho D. Choi'

# development only
POS_GOLD = 'pos_gold'


class POSState(NLPState):
    def __init__(self, document, params):
        """
        POSState implements the one-pass, left-to-right tagging algorithm.
        :param document: the input document
        :type document: list of elit.util.structure.Sentence
        :param params: parameters used for feature extraction, etc.
        :type params: dict
        """
        super().__init__(document)

        # parameters
        self.label_map = params['label_map']                # elit.util.lexicon.LabelMap
        self.word_vsm = params['word_vsm']                  # elit.util.lexicon.VectorSpaceModel
        self.ambi_vsm = params['ambi_vsm']                  # elit.util.lexicon.VectorSpaceModel
        self.context_windows = params['context_windows']    # list or tuple or range of int
        self.zero_output = params['zero_output']            # numpy.array

        # embeddings
        self.word_emb = [self.word_vsm.get_list(s[TOKENS]) for s in document]
        self.ambi_emb = [self.ambi_vsm.get_list(s[TOKENS]) for s in document] if self.ambi_vsm else None

        # state trackers
        self.sen_id = 0
        self.tok_id = 0
        self.reset()

    def reset(self):
        for s in self.document: s[POS_OUT] = [self.zero_output] * len(s)
        self.sen_id = 0
        self.tok_id = 0

    def process(self, output):
        # apply the output to the current state
        pos_output = self.document[self.sen_id][POS_OUT]
        pos_output[self.tok_id] = output

        # move onto the next state
        self.tok_id += 1
        if self.tok_id == len(pos_output):
            self.tok_id = 0
            self.sen_id += 1

    @property
    def has_next(self):
        return self.sen_id < len(self.document)

    @property
    def x(self):
        def position(window):
            i = self.tok_id + window
            return NLPState.x_fst if i == 0 else NLPState.x_lst if i+1 == size else NLPState.x_any

        def extract(window, emb, zero):
            i = self.tok_id + window
            return emb[i] if 0 <= i < size else zero

        sentence = self.document[self.sen_id]
        word_emb = self.word_emb[self.sen_id]
        ambi_emb = self.ambi_emb[self.sen_id] if self.ambi_vsm else None
        pos_out = sentence[POS_OUT]
        size = len(sentence)

        t = [position(i) for i in self.context_windows]
        w = [extract(i, word_emb, self.word_vsm.zero) for i in self.context_windows]
        a = [extract(i, ambi_emb, self.ambi_vsm.zero) for i in self.context_windows] if self.ambi_vsm else None
        p = [extract(i, pos_out, self.zero_output) for i in self.context_windows]

        return np.column_stack((t, w, a, p)) if a else np.column_stack((t, w, p))

    @property
    def y(self):
        return self.document[self.sen_id][POS_GOLD][self.tok_id]


class POSModel(gluon.Block):
    def __init__(self, params):
        super(POSModel, self).__init__()
        word_dim = params['word_vsm'].dim
        ambi_dim = params['ambi_vsm'].dim
        num_class = len(params['label_map'])
        base_conv = params['base_conv']
        ngram_conv = params['ngram_conv']
        fully_connected = params['fully_connected']

        with self.name_scope():
            # base convolution
            f0 = base_conv
            k0 = len(NLPState.x_lst) + word_dim + ambi_dim + num_class
            self.base_conv = gluon.nn.Conv2D(channels=f0, kernel_size=(1, k0), strides=(1, k0), activation='relu')
            # self.fc = gluon.nn.Dense(fully_connected)

            # n-gram convolutions
            self.ngram_conv = []
            for i, f in enumerate(ngram_conv):
                name = 'ngram_conv_' + str(i)
                setattr(self, name, gluon.nn.Conv2D(channels=f, kernel_size=(i+1, f0), strides=(1, f0), activation='relu'))
                self.ngram_conv.append(getattr(self, name))

            # output layer
            self.out = gluon.nn.Dense(num_class)

    def forward(self, x):
        # base convolution
        x = self.base_conv(x)
        # x = self.fc(x)

        # ngram convolutions
        # size = x.shape[1] * x.shape[2] * x.shape[3]
        x = nd.transpose(x, (0, 3, 2, 1))
        t = [conv(x).reshape((0, -1)) for conv in self.ngram_conv]
        x = nd.concat(*t, dim=1)

        # output layer
        x = self.out(x)
        return x


class POSEval:
    def __init__(self):
        """
        POSEval measures the accuracy of part-of-speech tagging.
        """
        self.correct = 0
        self.total = 0

    def update(self, state):
        for sentence in state.document:
            gold = sentence[POS_GOLD]
            pred = np.argmax(sentence[POS_OUT], axis=1)
            self.correct += np.sum(gold == pred)
            self.total += len(gold)

    def get(self):
        return 100.0 * self.correct / self.total

    def reset(self):
        self.correct = self.total = 0


def train_args():
    def windows(s):
        return tuple(map(int, s.split(',')))

    parser = argparse.ArgumentParser('Train: part-of-speech tagging')

    # data
    args = parser.add_argument_group('Data')
    args.add_argument('-t', '--trn_path', type=str, metavar='filepath', help='path to the training data')
    args.add_argument('-d', '--dev_path', type=str, metavar='filepath', help='path to the development data')

    # lexicon
    args = parser.add_argument_group('Lexicon')
    args.add_argument('-wv', '--word_vsm', type=str, metavar='filepath', help='vector space model for word embeddings')
    args.add_argument('-av', '--ambi_vsm', type=str, metavar='filepath', default=None, help='vector space model for ambiguity classes')
    args.add_argument('-cw', '--context_windows', type=windows, metavar='int[,int]*', default=(-2, -1, 0, 1, 2), help='context window for feature extraction')

    # train
    args = parser.add_argument_group('Train')
    args.add_argument('-ep', '--epoch', type=int, metavar='int', default=100, help='number of epochs')
    args.add_argument('-tb', '--trn_batch', type=int, metavar='int', default=32, help='batch size for training')
    args.add_argument('-db', '--dev_batch', type=int, metavar='int', default=512, help='batch size for evaluation')
    args.add_argument('-gid', '--gpu_id', type=int, metavar='int', default=0, help='ID of the GPU to be used')
    args.add_argument('-lr', '--learning_rate', type=float, metavar='float', default=0.01, help='learning rate')
    args.add_argument('-wd', '--weight_decay', type=float, metavar='float', default=0.0, help='decaying rate')
    args.add_argument('-mt', '--momentum', type=float, metavar='float', default=0.0, help='momentum')

    # model
    args = parser.add_argument_group('Model')
    args.add_argument('-bc', '--base_conv', type=int, metavar='int', default=64, help='number of filters for the base convolution')
    args.add_argument('-fc', '--fully_connected', type=int, metavar='int', default=240, help='number of units in the fully connected layer')
    args.add_argument('-nc', '--ngram_conv', type=windows, metavar='int[,int]*', default=(16,16,16,16,16), help='list of filter numbers for n-gram convolutions')

    return parser.parse_args()


def read_tsv(filepath, label_map, word_col=0, pos_col=1):
    """
    Read data from files in the TSV format, specified by the filepath.
    :param filepath: the path to a file (train.tsv) or multiple files (data_path/*.tsv).
    :type filepath: str
    :param label_map: the map containing class labels and their unique IDs.
    :type label_map: elit.util.lexicon.LabelMap
    :param word_col: the column index of words in the TSV file.
    :type word_col: int
    :param pos_col: the column index of postags in the TSV file.
    :type pos_col: int
    :return: a list of documents, where each document is a list of sentences.
    :rtype: list of (list of elit.util.structure.Sentence)
    """
    def aux(filename):
        wc = 0
        fin = open(filename)
        word_list = []
        gold_list = []

        for line in fin:
            l = line.split()
            if l:
                word_list.append(l[word_col])
                gold_list.append(label_map.add(l[pos_col]))
            elif word_list:
                s = Sentence({TOKENS: word_list, POS_GOLD: np.array(gold_list)})
                wc += len(s)
                sentences.append(s)
                word_list, gold_list = [], []
        return wc

    sentences = []
    word_count = 0
    for file in glob.glob(filepath): word_count += aux(file)
    documents = group_sentences(sentences)

    logging.info('Read: %s (sc = %d, wc = %d, grp = %d)' % (filepath, len(sentences), word_count, len(documents)))
    return documents


def get_params(label_map, args):
    params = {}

    params['label_map'] = label_map
    params['context_windows'] = args.context_windows
    params['base_conv'] = args.base_conv
    params['ngram_conv'] = args.ngram_conv
    params['fully_connected'] = args.fully_connected
    params['zero_output'] = np.zeros(len(label_map)).astype('float32')

    # do not save vector space models
    params['word_vsm'] = FastText(args.word_vsm)
    params['ambi_vsm'] = Word2Vec(args.ambi_vsm) if args.ambi_vsm else None

    return params


def reshape_x(x):
    return x.reshape((0, 1, x.shape[1], x.shape[2]))


def train():
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    args = train_args()

    # random seed for reproducibility
    random.seed(9)
    mx.random.seed(9)

    # read data
    label_map = LabelMap()
    trn_states = read_tsv(args.trn_path, label_map)
    dev_states = read_tsv(args.dev_path, label_map)

    # parameter configuration
    params = get_params(label_map, args)

    # create tagger
    model = POSModel(params)
    trn_states = [POSState(d, params) for d in trn_states]
    dev_states = [POSState(d, params) for d in dev_states]

    # init mxnet
    ctx = mx.gpu(args.gpu_id)
    ini = mx.init.Xavier(magnitude=2.24, rnd_type='gaussian')
    model.collect_params().initialize(ini, ctx=ctx)
    loss_func = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(model.collect_params(), 'adagrad', {'learning_rate': args.learning_rate})  # , 'momentum': args.momentum, 'wd': args.weight_decay})

    # train
    acc = POSEval()
    best_e, best_acc = -1, -1
    for e in range(args.epoch):
        trn_time = epoch(model, trn_states, args.trn_batch, ctx, reshape_x, trainer, loss_func)
        dev_time = epoch(model, dev_states, args.dev_batch, ctx, reshape_x, metric=acc)
        if best_acc < acc.get(): best_e, best_acc = e, acc.get()
        logging.info('%4d: trn_time: %3d, dev_time: %d, acc: %5.2f (%d), best-acc: %5.2f @%4d' %
                     (e, trn_time, dev_time, acc.get(), acc.total, best_acc, best_e))
        acc.reset()


if __name__ == '__main__':
    train()





















# def transpose_filters(x):
#     return nd.stack(*[nd.transpose(a) for a in x])
#


# def merge_filters(x):
#     y = [nd.transpose(a).asnumpy() for a in x]
#     x = x.reshape((-1, 1, x.shape[2], x.shape[1]))
#     for i in range(len(x)): x[i] = y[i]
#     return x
#







# class ConvBlock(gluon.Block):
#     def __init__(self):
#         super(ConvBlock, self).__init__()
#         with self.name_scope():
#             f0 = 3
#             k0 = 5
#             fn = 4
#             self.conv0 = gluon.nn.Conv2D(channels=f0, kernel_size=(1, k0), strides=(1, -1), activation='relu')
#
#             for i in range(1,3):
#                 setattr(self, 'conv'+str(i), gluon.nn.Conv2D(channels=fn, kernel_size=(i, f0), strides=(1, -1), activation='relu'))
#
#             self.convn = [getattr(self, 'conv'+str(i)) for i in range(1,3)]
#             self.fc0 = gluon.nn.Dense(5)
#             self.fc1 = gluon.nn.Dense(5)
#             self.fc2 = gluon.nn.Dense(5)
#             self.fcn = [self.fc0, self.fc1, self.fc2]
#
#     def forward(self, x):
#         x = transpose_filters(self.conv0(x))
#
#         t = [transpose_filters(conv(x)).reshape((0, -1)) for conv in self.convn]
#         x = nd.concat(*t, dim=1)
#
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
# # loss_func = gluon.loss.SoftmaxCrossEntropyLoss()
# loss_func = SoftmaxCrossEntropyLossD()
# data = gluon.data.DataLoader(gluon.data.ArrayDataset(X, Y), batch_size=batch_size)
#
# for x, y in data:
#     x = x.as_in_context(ctx)
#     x = x.reshape((0, 1, x.shape[1], x.shape[2]))
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






# class POSTagger(metaclass=abc.ABCMeta):
#     def __init__(self, word_vsm, ambi_vsm, label_map, max_len):
#         """
#         :param word_vsm: the vector space model for word embeddings (e.g., Word2Vec, FastText).
#         :type word_vsm: elit.util.lexicon.VectorSpaceModel
#         :param ambi_vsm: the vector space model for ambiguity classes.
#         :type ambi_vsm: elit.util.lexicon.VectorSpaceModel
#         :param num_class: the total number of part-of-speech tags.
#         :type num_class: int
#         :param max_len: the maximum number of words in each document.
#         :type max_len: int
#         """
#         self.word_vsm = word_vsm
#         self.ambi_vsm = ambi_vsm
#         self.label_map = label_map
#         self.max_len = max_len
#         self.num_class = len(label_map)
#
#         self.zero_scores = np.zeros(self.num_class).astype('float32')
#         self.fst_word = np.array([1,0])
#         self.mid_word = np.array([0,0])
#         self.lst_word = np.array([0,1])
#
#     def input_matrix(self, document):
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
#             word_emb = sentence.setdefault(WORD_EMB, self.word_vsm.get_list(sentence[TOKENS]))
#             ambi_emb = sentence.setdefault(AMBI_EMB, self.ambi_vsm.get_list(sentence[TOKENS]))
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
#     def set_scores(self, document, scores):
#         """
#         :param document: a sentence or a list of sentences, where each sentence is represented by a dictionary.
#         :type document: Union[list of dict, dict]
#         :param scores: the scores of the part-of-speech tag predictions.
#         :param scores: numpy.array -> max_len * num_class
#         """
#         def index(i):
#             return (begin + i) * self.num_class
#
#         def get(sentence):
#             return [scores[index(i):index(i+1)] for i in range(0, len(sentence))]
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

# def trunc_pads(labels, output):
#     idx = next((i for i, label in enumerate(labels) if label.asscalar() == -1), None)
#     return (labels[:idx], output[:idx]) if idx else (labels, output)
