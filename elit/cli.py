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
from types import SimpleNamespace

import mxnet as mx

__author__ = 'Jinho D. Choi'


def arg_conv(s):
    """
    :param s: (ngram:filters:activation:dropout)(;ngram:channels:activation:dropout)*
    """
    def create(config):
        c = config.split(':')
        return SimpleNamespace(ngram=int(c[0]), filters=int(c[1]), activation=c[2], dropout=float(c[3]))

    return [create(config) for config in s.split(';')]


def train_args():



    def int_tuple(s):
        return tuple(map(int, s.split(',')))



    parser = argparse.ArgumentParser('Train: part-of-speech tagging')

    # data
    parser.add_argument('-t', '--trn_path', type=str, metavar='filepath',
                        help='path to the training data (input)')
    parser.add_argument('-d', '--dev_path', type=str, metavar='filepath',
                        help='path to the development data (input)')
    parser.add_argument('-m', '--mod_path', type=str, metavar='filepath', default=None,
                        help='path to the model data (output)')
    parser.add_argument('-l', '--log_path', type=str, metavar='filepath', default='logs',
                        help='path to the log data (output)')

    parser.add_argument('-vt', '--tsv_tok', type=int, metavar='int', default=0,
                        help='the column index of tokens in TSV')
    parser.add_argument('-vp', '--tsv_pos', type=int, metavar='int', default=1,
                        help='the column index of pos-tags in TSV')

    # lexicon
    parser.add_argument('-wv', '--word_vsm', type=str, metavar='filepath',
                        help='vector space model for word embeddings')
    parser.add_argument('-av', '--ambi_vsm', type=str, metavar='filepath', default=None,
                        help='vector space model for ambiguity classes')

    # configuration

    parser.add_argument('-cw', '--windows', type=int_tuple, metavar='int[,int]*',
                        default=(-3, -2, -1, 0, 1, 2, 3),
                        help='contextual windows for feature extraction')
    parser.add_argument('-nf', '--ngram_filters', type=int_tuple, metavar='int[,int]*',
                        default=(128, 128, 128, 128, 128),
                        help='number of filters for n-gram conv2d')
    parser.add_argument('-do', '--dropout', type=float, metavar='float', default=0.2,
                        help='dropout')

    parser.add_argument('-cx', '--ctx', type=context, metavar='[cg]\d', default=0,
                        help='device context')
    parser.add_argument('-ep', '--epoch', type=int, metavar='int', default=50,
                        help='number of epochs')
    parser.add_argument('-tb', '--trn_batch', type=int, metavar='int', default=64,
                        help='batch size for training')
    parser.add_argument('-db', '--dev_batch', type=int, metavar='int', default=1024,
                        help='batch size for evaluation')
    parser.add_argument('-lr', '--learning_rate', type=float, metavar='float', default=0.01,
                        help='learning rate')

    parser.add_argument('-ido', '--input_dropout', type=float, metavar='float', default=0,
                        help='dropout rate applied to the input layer')
    parser.add_argument('-nc', '--num_class', type=int, metavar='int', default=50,
                        help='number of classes (part-of-speech tags)')


    args = parser.parse_args()

    return args
