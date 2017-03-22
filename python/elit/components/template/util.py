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
import os
from typing import List, Tuple, Callable

import mxnet as mx

from elit.reader import TSVReader
from elit.structure import NLPGraph

__author__ = 'Jinho D. Choi'


# ============================== Reader ==============================

def read_graphs(reader: TSVReader, filename: str) -> List[NLPGraph]:
    logging.info('Reading: '+os.path.basename(filename))
    reader.open(filename)
    graphs = reader.next_all
    reader.close()
    logging.info('%s graphs' % len(graphs))
    return graphs


# ============================== Arguments ==============================

def argparse_data(parser: argparse.ArgumentParser, tsv: Tuple[Callable[[Tuple[int]], TSVReader], str]=None):
    """
    :param parser: the parent parser.
    :param tsv: (indices -> TSVReader, comment).
    """
    data = parser.add_argument_group('Data')

    data.add_argument('--trn_data', type=str, metavar='path', help='path to the training file')
    data.add_argument('--dev_data', type=str, metavar='path', help='path to the development file')

    if tsv:
        def reader(s: str):
            t = tuple(map(int, s.split(',')))
            return tsv[0](t)

        data.add_argument('--tsv', type=reader, metavar='int(,int)*', help=tsv[1])

    return data


def argparse_model(parser: argparse.ArgumentParser):
    def context(s: str):
        m = mx.cpu if s[0] == 'c' else mx.gpu
        s = s[1:]

        if ',' in s:
            r = tuple(map(int, s.split(',')))
        elif '-' in s:
            t = tuple(map(int, s.split('-')))
            r = range(t[0], t[1]+1)
        else:
            r = [int(s)]

        return m(r[0]) if len(r) == 1 else [m(i) for i in r]

    model = parser.add_argument_group('Model')

    model.add_argument('--batch_size', type=int, metavar='int', default=128,
                       help='size of the mini batch')
    model.add_argument('--threads', type=int, metavar='int', default=4,
                       help='number of threads used for feature extraction')
    model.add_argument('--context', type=context, metavar='g|c:int(,int)*|int-int', default=mx.cpu(),
                       help='context used for fitting the model')

    return model


def argparse_ffnn(parser: argparse.ArgumentParser):
    def hidden(s: str):
        size, activation, dropout = s.split(',')
        return int(size), activation, float(dropout)

    ffnn = parser.add_argument_group('Model', 'Feed forward neural network')

    ffnn.add_argument('--hidden', nargs='*', type=hidden, metavar='int,str,float', default=[],
                      help='(size, act type, dropout rate) for hidden layers')
    ffnn.add_argument('--input_dropout', type=float, metavar='float', default=0,
                      help='dropout rate for input layer')
    ffnn.add_argument('--output_size', type=int, metavar='int', default=2,
                      help='size of the output layer')


# def add_arguments_fit(parser: argparse.ArgumentParser):
#     def context(s: str):
#         try:
#
#         except:
#             raise argparse.ArgumentTypeError('Invalid format: '+s)
#
#
#     parser.add_argument('--context', nargs='+' type=str,
#                        help='cpu|gpu ')
#     train.add_argument('--gpus', type=str,
#                        help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu')
#     train.add_argument('--kv-store', type=str, default='device',
#                        help='key-value store type')
#     train.add_argument('--num-epochs', type=int, default=100,
#                        help='max num of epochs')
#     train.add_argument('--lr', type=float, default=0.1,
#                        help='initial learning rate')
#     train.add_argument('--lr-factor', type=float, default=0.1,
#                        help='the ratio to reduce lr on each step')
#     train.add_argument('--lr-step-epochs', type=str,
#                        help='the epochs to reduce the lr, e.g. 30,60')
#     train.add_argument('--optimizer', type=str, default='sgd',
#                        help='the optimizer type')
#     train.add_argument('--mom', type=float, default=0.9,
#                        help='momentum for sgd')
#     train.add_argument('--wd', type=float, default=0.0001,
#                        help='weight decay for sgd')
#     train.add_argument('--batch-size', type=int, default=128,
#                        help='the batch size')
#     train.add_argument('--disp-batches', type=int, default=20,
#                        help='show progress for every n batches')
#     train.add_argument('--model-prefix', type=str,
#                        help='model prefix')
#     parser.add_argument('--monitor', dest='monitor', type=int, default=0,
#                         help='log network parameters every N iters if larger than 0')
#     train.add_argument('--load-epoch', type=int,
#                        help='load the model on an epoch using the model-load-prefix')
#     train.add_argument('--top-k', type=int, default=0,
#                        help='report the top-k accuracy. 0 means no report.')
#     train.add_argument('--test-io', type=int, default=0,
#                        help='1 means test reading speed without training')
#     return train


# def create_argparser(description: str) -> argparse.ArgumentParser:
#
#     parser = argparse.ArgumentParser(description=description)
#
#     # neural networks
#     add_arguments_ffnn(parser)
#
#
#     # data
#     parser.add_argument('--train', nargs=1, type=str, help='path to the training data')
#     parser.add_argument('--develop', nargs=1, type=str, help='path to the development data')
#
#     # mxnet
#     parser.add_argument('--context', nargs=2, type)
#
#     return parser
