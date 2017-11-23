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

from elit.util.component import NLPState, process_online, ForwardState, CONTEXT_WINDOWS, WORD_VSM, LABEL_MAP, \
    ZERO_OUTPUT, \
    NLPEval, NGRAM_CONV, DROPOUT
from elit.util.lexicon import LabelMap, FastText, Word2Vec
from elit.util.math import transition_probs, viterbi
from elit.util.structure import Sentence, group_sentences
from elit.util.structure import TOKENS

__author__ = 'Jinho D. Choi'


POS_GOLD = 'pos_gold'
AMBI_VSM = 'ambi_vsm'


class POSState(ForwardState):
    def __init__(self, document, params):
        """
        POSState inherits the one-pass, left-to-right tagging algorithm.
        :param document: the input document
        :type document: list of elit.util.structure.Sentence
        :param params: a dictionary containing [label_map, word_vsm, ambi_vsm, zero_output]
        :type params: dict
        """
        super().__init__(document, params, POS_GOLD)

        # parameters
        self.context_windows = params[CONTEXT_WINDOWS]    # list or tuple or range of int
        self.ambi_vsm = params[AMBI_VSM]                  # elit.util.lexicon.VectorSpaceModel

        # embeddings
        self.ambi_emb = [self.ambi_vsm.get_list(s[TOKENS]) for s in document] if self.ambi_vsm else None

    @property
    def x(self):
        word_emb = self.word_emb[self.sen_id]
        ambi_emb = self.ambi_emb[self.sen_id] if self.ambi_vsm else None
        scores = self.scores[self.sen_id]

        p = [self._x_position(i) for i in self.context_windows]
        w = [self._extract_x(i, word_emb, self.word_vsm.zero) for i in self.context_windows]
        a = [self._extract_x(i, ambi_emb, self.ambi_vsm.zero) for i in self.context_windows] if self.ambi_vsm else None
        s = [self._extract_x(i, scores, self.zero_output) for i in self.context_windows]

        return np.column_stack((p, w, a, s)) if a else np.column_stack((p, w, s))


class POSModel(gluon.Block):
    def __init__(self, params):
        super(POSModel, self).__init__()
        word_dim = params[WORD_VSM].dim
        ambi_vsm = params.get(AMBI_VSM, None)
        ambi_dim = ambi_vsm.dim if ambi_vsm else 0
        num_class = len(params[LABEL_MAP])
        ngram_conv = params[NGRAM_CONV]
        dropout = params[DROPOUT]

        with self.name_scope():
            k = len(NLPState.x_lst) + word_dim + ambi_dim + num_class
            self.ngram_conv = []
            for i, f in enumerate(ngram_conv):
                name = 'ngram_conv_' + str(i)
                setattr(self, name, gluon.nn.Conv2D(channels=f, kernel_size=(i+1, k), strides=(1, k), activation='relu'))
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


class POSEval(NLPEval):
    def __init__(self):
        """
        POSEval measures the accuracy of part-of-speech tagging.
        """
        self.correct = 0
        self.total = 0

    def update(self, state):
        for i, sentence in enumerate(state.document):
            gold = sentence[POS_GOLD]
            pred = np.argmax(state.scores[i], axis=1)
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
    args.add_argument('-gid', '--gpu_id', type=int, metavar='int', default=0, help='ID of the GPU to be used')
    args.add_argument('-ep', '--epoch', type=int, metavar='int', default=50, help='number of epochs')
    args.add_argument('-tb', '--trn_batch', type=int, metavar='int', default=32, help='batch size for training')
    args.add_argument('-db', '--dev_batch', type=int, metavar='int', default=512, help='batch size for evaluation')
    args.add_argument('-lr', '--learning_rate', type=float, metavar='float', default=0.01, help='learning rate')
    args.add_argument('-nc', '--ngram_conv', type=windows, metavar='int[,int]*', default=(128,128,128,128,128), help='list of filter numbers for n-gram convolutions')
    args.add_argument('-do', '--dropout', type=float, metavar='float', default=0.2, help='dropout')

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

    params[LABEL_MAP] = label_map
    params[ZERO_OUTPUT] = np.zeros(len(label_map)).astype('float32')
    params[CONTEXT_WINDOWS] = args.context_windows
    params[NGRAM_CONV] = args.ngram_conv
    params[DROPOUT] = args.dropout

    # do not save vector space models
    params[WORD_VSM] = FastText(args.word_vsm)
    params[AMBI_VSM] = Word2Vec(args.ambi_vsm) if args.ambi_vsm else None

    s = ['Configuration',
         '- train batch: %d' % args.trn_batch,
         '- develop batch: %d' % args.dev_batch,
         '- dropout ratio: %f' % params[DROPOUT],
         '- learning reate: %f' % args.learning_rate,
         '- context window: %s' % str(params[CONTEXT_WINDOWS]),
         '- convolution filters: %s' % str(params[NGRAM_CONV])]

    return params, '\n'.join(s)


def reshape_x(x):
    return x.reshape((0, 1, x.shape[1], x.shape[2]))


def train():
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    args = train_args()
    random.seed(11)
    mx.random.seed(11)

    # read data
    label_map = LabelMap()
    trn_states = read_tsv(args.trn_path, label_map)
    dev_states = read_tsv(args.dev_path, label_map)

    # parameter configuration
    params, s = get_params(label_map, args)
    logging.info(s)

    # create tagger
    model = POSModel(params)
    trn_states = [POSState(d, params) for d in trn_states]
    dev_states = [POSState(d, params) for d in dev_states]

    # init mxnet
    ctx = mx.gpu(args.gpu_id)
    ini = mx.init.Xavier(magnitude=2.24, rnd_type='gaussian')
    model.collect_params().initialize(ini, ctx=ctx)
    loss_func = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(model.collect_params(), 'adagrad', {'learning_rate': args.learning_rate})

    # train
    trn_acc = POSEval()
    dev_acc = POSEval()
    best_e, best_acc = -1, -1

    for e in range(args.epoch):
        trn_acc.reset()
        dev_acc.reset()

        trn_time = process_online(model, trn_states, args.trn_batch, ctx, trainer, loss_func, trn_acc, reshape_x)
        dev_time = process_online(model, dev_states, args.dev_batch, ctx, metric=dev_acc, reshape_x=reshape_x)

        if best_acc < dev_acc.get(): best_e, best_acc = e, dev_acc.get()
        logging.info('%4d: trn-time: %d, dev-time: %d, trn-acc: %5.2f (%d), dev-acc: %5.2f (%d), best-acc: %5.2f @%4d' %
                     (e, trn_time, dev_time, trn_acc.get(), trn_acc.total, dev_acc.get(), dev_acc.total, best_acc, best_e))


if __name__ == '__main__':
    train()
