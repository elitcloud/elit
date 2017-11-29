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
import logging
import random
from argparse import Namespace

import mxnet as mx
import numpy as np
from mxnet import gluon, nd

from elit.nlp.model import LabelMap, NLPModel, x_position, x_extract
from elit.nlp.state import ForwardState
from elit.nlp.util import read_tsv, process_online, reshape_conv2d
from elit.util.lexicon import FastText, Word2Vec
from elit.util.structure import TOKEN, POS

__author__ = 'Jinho D. Choi'


class POSState(ForwardState):
    def __init__(self, document, zero_output, word_vsm, ambi_vsm=None):
        """
        POSState inherits the one-pass, left-to-right decoding strategy.
        :param document: the input document.
        :type document: list of elit.util.structure.Sentence
        :param zero_output: a vector of size `num_class` where all values are 0; used to zero pad label embeddings.
        :type zero_output: numpy.array
        :param word_vsm: the vector space model for word embeddings.
        :type word_vsm: elit.util.lexicon.VectorSpaceModel
        :param ambi_vsm: the vector space model for ambiguity classes.
        :type ambi_vsm: elit.util.lexicon.VectorSpaceModel
        """
        super().__init__(document, zero_output)
        self.word_emb = [word_vsm.get_list(s[TOKEN]) for s in document]
        self.ambi_emb = [ambi_vsm.get_list(s[TOKEN]) for s in document] if ambi_vsm else None


class POSModel(NLPModel):
    def __init__(self, params, **kwargs):
        """
        POSModel defines the statistical model for part-of-speech tagging.
        :param params: a namespace containing parameters to build this model.
        :type params: argparse.Namespace
        :param kwargs: parameters for gluon.Block
        :type kwargs: dict
        """
        super(POSModel, self).__init__(params, **kwargs)

        # parameters
        self.word_vsm = params.word_vsm
        self.ambi_vsm = params.ambi_vsm
        self.zero_output = params.zero_output
        self.context_windows = params.context_windows

        # network
        word_dim = self.word_vsm.dim
        ambi_dim = self.ambi_vsm.dim if self.ambi_vsm else 0
        num_class = len(self.label_map)
        k = len(X_ANY) + word_dim + ambi_dim + num_class
        self.ngram_conv = []

        with self.name_scope():
            for i, f in enumerate(params.ngram_filters):
                conv = gluon.nn.Conv2D(channels=f, kernel_size=(i+1, k), strides=(1, k), activation='relu')
                name = 'ngram_conv_' + str(i)
                self.ngram_conv.append(conv)
                setattr(self, name, conv)

            self.dropout = gluon.nn.Dropout(params.dropout)
            self.out = gluon.nn.Dense(num_class)

    def forward(self, x):
        # n-gram convolutions
        t = [conv(x).reshape((0, -1)) for conv in self.ngram_conv]
        x = nd.concat(*t, dim=1)
        x = self.dropout(x)

        # output layer
        x = self.out(x)
        return x

    def create_state(self, document):
        return POSState(document, self.zero_output, self.word_vsm, self.ambi_vsm)

    def x(self, state):
        word_emb = state.word_emb[state.sen_id]
        ambi_emb = state.ambi_emb[state.sen_id] if self.ambi_vsm else None
        output = state.output[state.sen_id]
        size = len(word_emb)
        tid = state.tok_id

        p = [x_position(tid, w, size) for w in self.context_windows]
        w = [x_extract(tid, w, size, word_emb, self.word_vsm.zero) for w in self.context_windows]
        a = [x_extract(tid, w, size, ambi_emb, self.ambi_vsm.zero) for w in self.context_windows] if self.ambi_vsm else None
        o = [x_extract(tid, w, size, output, self.zero_output) for w in self.context_windows]

        return np.column_stack((p, w, a, o)) if a else np.column_stack((p, w, o))

    def y(self, state):
        label = state.document[state.sen_id][POS][state.tok_id]
        return self.label_map.add(label)

    def eval(self, state, counts):
        self.trim_output(state)

        for i, sentence in enumerate(state.document):
            gold = np.array([self.label_map.index(t) for t in sentence[POS]])
            pred = np.argmax(state.output[i], axis=1)
            counts.correct += np.sum(gold == pred)
            counts.total += len(gold)

    def set_labels(self, state):
        self.trim_output(state)

        for i, output in enumerate(state.output):
            sentence = state.document[i]
            sentence[POS] = [self.label_map.get(np.argmax(o)) for o in output]


def args_train():
    def int_tuple(s):
        return tuple(map(int, s.split(',')))

    parser = argparse.ArgumentParser('Train: part-of-speech tagging')

    # data
    parser.add_argument('-t', '--trn_path', type=str, metavar='filepath', help='path to the training data')
    parser.add_argument('-d', '--dev_path', type=str, metavar='filepath', help='path to the development data')

    # lexicon
    parser.add_argument('-wv', '--word_vsm', type=str, metavar='filepath', help='vector space model for word embeddings')
    parser.add_argument('-av', '--ambi_vsm', type=str, metavar='filepath', default=None, help='vector space model for ambiguity classes')
    parser.add_argument('-cw', '--context_windows', type=int_tuple, metavar='int[,int]*', default=(-2, -1, 0, 1, 2), help='context windows for feature extraction')

    # train
    parser.add_argument('-gid', '--gpu_id', type=int, metavar='int', default=0, help='ID of the GPU to be used')
    parser.add_argument('-ep', '--epoch', type=int, metavar='int', default=50, help='number of epochs')
    parser.add_argument('-tb', '--trn_batch', type=int, metavar='int', default=32, help='batch size for training')
    parser.add_argument('-db', '--dev_batch', type=int, metavar='int', default=512, help='batch size for evaluation')
    parser.add_argument('-lr', '--learning_rate', type=float, metavar='float', default=0.01, help='learning rate')
    parser.add_argument('-nf', '--ngram_filters', type=int_tuple, metavar='int[,int]*', default=(128,128,128,128,128), help='number of filters for n-gram convolutions')
    parser.add_argument('-do', '--dropout', type=float, metavar='float', default=0.2, help='dropout')
    parser.add_argument('-nc', '--num_class', type=int, metavar='int', default=45, help='number of classes')

    args = parser.parse_args()

    log = ['Configuration',
           '- train batch: %d' % args.trn_batch,
           '- develop batch: %d' % args.dev_batch,
           '- dropout ratio: %f' % args.dropout,
           '- learning rate: %f' % args.learning_rate,
           '- context window: %s' % str(args.context_windows),
           '- n-gram filters: %s' % str(args.ngram_filters)]

    return args, log


def train():
    def eval(counts):
        return 100.0 * counts.correct / counts.total

    logging.basicConfig(format='%(message)s', level=logging.INFO)
    mx.random.seed(11)
    random.seed(11)

    # parameters
    args, log = args_train()
    params = Namespace(
        label_map=LabelMap(),
        word_vsm=FastText(args.word_vsm),
        ambi_vsm=Word2Vec(args.ambi_vsm) if args.ambi_vsm else None,
        zero_output=np.zeros(args.num_class).astype('float32'),
        context_windows=args.context_windows,
        ngram_filters=args.ngram_filters)

    logging.info(log)

    # model
    model = POSModel(params)

    # states
    cols = {TOKEN: 0, POS: 1}
    trn_states = read_tsv(args.trn_path, model.create_state, cols)
    dev_states = read_tsv(args.dev_path, model.create_state, cols)

    # network
    ctx = mx.gpu(args.gpu_id)
    ini = mx.init.Xavier(magnitude=2.24, rnd_type='gaussian')
    model.collect_params().initialize(ini, ctx=params.ctx)
    loss_func = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(model.collect_params(), 'adagrad', {'learning_rate': args.learning_rate})

    # train
    best_e, best_acc = -1, -1

    for e in range(args.epoch):
        trn_eval = Namespace(correct=0, total=0)
        dev_eval = Namespace(correct=0, total=0)

        trn_time = process_online(model, trn_states, args.trn_batch, ctx, trainer, loss_func, trn_eval, reshape_conv2d)
        dev_time = process_online(model, dev_states, args.dev_batch, ctx, eval_counts=dev_eval, reshape_x=reshape_conv2d)

        trn_acc = eval(trn_eval)
        dev_acc = eval(dev_eval)
        if best_acc < dev_acc: best_e, best_acc = e, dev_acc

        logging.info('%4d: trn-time: %d, dev-time: %d, trn-acc: %5.2f (%d), dev-acc: %5.2f (%d), best-acc: %5.2f @%4d' %
                     (e, trn_time, dev_time, trn_acc, trn_eval.total, dev_acc, dev_eval.total, best_acc, best_e))


if __name__ == '__main__':
    train()
