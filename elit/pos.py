# ========================================================================
# Copyright 2018 Emory University
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
import re
import sys
from types import SimpleNamespace

import mxnet as mx

from elit.model import conv2d_args, hidden_args, loss_args, context_args
from elit.structure import TOK
from elit.token_tagger import TokenSequenceTagger, TokenBatchTagger
from elit.utils.file import tsv_reader, json_reader
from elit.vsm import FastText

__author__ = "Gary Lai"


class PosTagger(object):

    def __init__(self):
        pass


# ======================================== Command-Line ========================================
def reader_args(s):
    """
    :param s: (tsv|json)(;\\d:\\d)*
    :return: reader, SimpleNamespace(tok, pos)
    """
    r = s.split(';')
    if r[0] == 'tsv':
        t = r[1].split(':')
        return tsv_reader, SimpleNamespace(tok=int(t[0]), tag=int(t[1]))
    else:
        return json_reader, None


def train_args():
    def int_tuple(s):
        """
        :param s: \\d(,\\d)*
        :return: tuple of int
        """
        return tuple(map(int, s.split(',')))

    parser = argparse.ArgumentParser('Train: part-of-speech tagging')

    # data
    parser.add_argument('-t', '--trn_path', type=str, metavar='filepath',
                        help='path to the training data (input)')
    parser.add_argument('-d', '--dev_path', type=str, metavar='filepath',
                        help='path to the development data (input)')
    parser.add_argument('-m', '--model_path', type=str, metavar='filepath', default=None,
                        help='path to the model data (output)')
    parser.add_argument('-r', '--reader', type=reader_args, metavar='(tsv|json)(;\\d:\\d)*',
                        default=(tsv_reader, SimpleNamespace(tok=0, tag=1)),
                        help='reader configuration')

    # generic
    parser.add_argument('-key', '--key', type=str, metavar='str',
                        help='path to the model data (output)')
    parser.add_argument('-seq', '--sequence', type=bool, metavar='boolean', default=False,
                        help='if set, use sequence mode')
    parser.add_argument('-chu', '--chunking', type=bool, metavar='boolean', default=False,
                        help='if set, generate chunks')

    # lexicon
    parser.add_argument('-wv', '--word_vsm', type=str, metavar='filepath',
                        help='vector space model for word embeddings')

    # configuration
    parser.add_argument('-nc', '--num_class', type=int, metavar='int',
                        help='number of classes (part-of-speech tags)')
    parser.add_argument('-fw', '--feature_windows', type=int_tuple, metavar='int[,int]*',
                        default=tuple(range(-3, 4)),
                        help='contextual windows for feature extraction')
    parser.add_argument('-ir', '--input_dropout', type=float, metavar='float', default=0.0,
                        help='dropout rate applied to the input layer')
    parser.add_argument('-cc', '--conv2d_config', type=conv2d_args,
                        metavar='(ngram:filters:activation:pool:dropout)(;#1)*',
                        default=tuple(SimpleNamespace(ngram=i, filters=128, activation='relu', pool=None, dropout=0.2) for i in range(1, 6)),
                        help='configuration for the convolution layer')
    parser.add_argument('-hc', '--hidden_config', type=hidden_args, metavar='(dim:activation:dropout)(;#1)*', default=None,
                        help='configuration for the hidden layer')

    # training
    parser.add_argument('-cx', '--ctx', type=str, metavar='[cg]\\d', default=None,
                        help='device context')
    parser.add_argument('-ep', '--epoch', type=int, metavar='int', default=50,
                        help='number of epochs')
    parser.add_argument('-tb', '--trn_batch', type=int, metavar='int', default=64,
                        help='batch size for training')
    parser.add_argument('-db', '--dev_batch', type=int, metavar='int', default=4096,
                        help='batch size for evaluation')
    parser.add_argument('-lo', '--loss', type=loss_args, metavar='str', default=None,
                        help='loss function')
    parser.add_argument('-op', '--optimizer', type=str, metavar='str', default='adagrad',
                        help='optimizer algorithm')
    parser.add_argument('-lr', '--learning_rate', type=float, metavar='float', default=0.01,
                        help='learning rate')
    parser.add_argument('-wd', '--weight_decay', type=float, metavar='float', default=0.0,
                        help='weight decay')

    args = parser.parse_args()
    return args


def evaluate_args():
    parser = argparse.ArgumentParser('Train: part-of-speech tagging')

    # data
    parser.add_argument('-d', '--dev_path', type=str, metavar='filepath',
                        help='path to the evaluation data (input)')
    parser.add_argument('-m', '--model_path', type=str, metavar='filepath', default=None,
                        help='path to the model data (output)')
    parser.add_argument('-r', '--reader', type=reader_args, metavar='(tsv|json)(;\\d:\\d)*',
                        default=(tsv_reader, SimpleNamespace(tok=0, tag=1)),
                        help='reader configuration')

    parser.add_argument('-seq', '--sequence', type=bool, metavar='boolean', default=False,
                        help='if set, use sequence mode')

    # lexicon
    parser.add_argument('-wv', '--word_vsm', type=str, metavar='filepath',
                        help='vector space model for word embeddings')

    # evaluation
    parser.add_argument('-cx', '--ctx', type=str, metavar='[cg]\\d', default=None,
                        help='device context')
    parser.add_argument('-db', '--dev_batch', type=int, metavar='int', default=4096,
                        help='batch size for evaluation')

    args = parser.parse_args()
    return args


def train():
    # cml arguments
    args = train_args()
    if args.ctx is None: args.ctx = mx.cpu()
    if args.loss is None: args.loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()
    initializer = mx.init.Xavier(magnitude=2.24, rnd_type='gaussian')

    # vector space models
    vsm_list = tuple((FastText(args.word_vsm), TOK) for _ in range(1))

    # component
    comp = TokenSequenceTagger(args.ctx, vsm_list) if args.sequence else TokenBatchTagger(args.ctx, vsm_list)
    comp.init(args.key, args.chunking, args.num_class, args.feature_windows, args.input_dropout, args.conv2d_config, args.hidden_config, initializer)

    # data
    reader, rargs = args.reader
    trn_docs = reader(args.trn_path, args.key, rargs)
    dev_docs = reader(args.dev_path, args.key, rargs)

    # train
    comp.train(trn_docs, dev_docs, args.model_path, args.trn_batch, args.dev_batch, args.epoch, args.loss, args.optimizer, args.learning_rate, args.weight_decay)
    print('# of classes: %d' % len(comp.label_map))


def evaluate():
    # cml arguments
    args = train_args()
    if args.ctx is None: args.ctx = mx.cpu()

    # vector space models
    vsm_list = tuple((FastText(args.word_vsm), TOK) for _ in range(1))

    # component
    comp = TokenSequenceTagger(args.ctx, vsm_list) if args.sequence else TokenBatchTagger(args.ctx, vsm_list)
    comp.load(args.model_path)

    # data
    reader, rargs = args.reader
    dev_docs = reader(args.dev_path, comp.key, rargs)

    # decode
    states = comp.create_states(dev_docs)
    e = comp._evaluate(states, args.dev_batch)
    print(str(e))


if __name__ == '__main__':
    train()
    evaluate()


