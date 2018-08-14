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
import sys

import mxnet as mx

from elit.token_tagger import TokenSequenceTagger, TokenBatchTagger
from elit.util.cli import args_context, args_loss, feature_windows_args, args_reader, conv2d_args, \
    args_hidden
from elit.util.structure import TOK
from elit.util.vsm import FastText

__author__ = "Gary Lai"


class PosTagger(object):

    def __init__(self):
        pass


# ======================================== Command-Line ========================================
class PosCli(object):

    def __init__(self):
        self.key = 'pos'
        parser = argparse.ArgumentParser(
            usage='''
    elit pos <command> [<args>]

commands:
    train   part-of-speech tagger
    eval    named entity recognition
''',
            description='Part-of-speech tagging')
        parser.add_argument('command', help='command to run')
        args = parser.parse_args(sys.argv[2:3])
        if not hasattr(self, args.command):
            logging.info('Unrecognized command')
            parser.print_help()
            exit(1)
        getattr(self, args.command)()

    def train(self):

        # positional arguments:
        # These are required arguments. Positional arguments don't use hyphen or dash prefix.
        parser = argparse.ArgumentParser(description='Train: Part-of-speech tagging',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('trn_path', type=str, metavar='TRN_PATH',
                            help='path to the training data (input)')
        parser.add_argument('dev_path', type=str, metavar='DEV_PATH',
                            help='path to the development data (input)')
        #   Lexicon
        parser.add_argument('word_vsm', type=str, metavar='WORD_VSM_PATH',
                            help='path to the vector space model for word embeddings')

        # Optional arguments:
        # These are optional arguments.
        #
        #   Generic
        parser.add_argument('-m', '--model_path', type=str, metavar='MODEL_PATH', default=None,
                            help='path to the model data (output)')
        parser.add_argument('-r', '--reader', type=args_reader, metavar='READER',
                            default="tsv:0,1",
                            help='reader configuration. format: (tsv|json)(:\\d,\\d)*')
        parser.add_argument('-seq', '--sequence', type=bool, metavar='BOOLEAN', default=False,
                            help='if set, use sequence mode')
        parser.add_argument('-chu', '--chunking', type=bool, metavar='BOOLEAN', default=False,
                            help='if set, generate chunks')

        #   Configuration
        parser.add_argument('-nc', '--num_class', type=int, metavar='INT',
                            help='number of classes (part-of-speech tags)')
        parser.add_argument('-fw', '--feature_windows', type=feature_windows_args,
                            metavar='FEATURE_WINDOWS', default=tuple(range(-3, 4)),
                            help='contextual windows for feature extraction')
        parser.add_argument('-ir', '--input_dropout', type=float, metavar='float', default=0.0,
                            help='dropout rate applied to the input layer')
        # parser.add_argument('-cc', '--conv2d_config', type=conv2d_args,
        #                     metavar='(ngram:filters:activation:pool:dropout)(;#1)*',
        #                     default=tuple(
        #                         SimpleNamespace(ngram=i, filters=128, activation='relu', pool=None,
        #                                         dropout=0.2) for i in range(1, 6)),
        #                     help='configuration for the convolution layer')
        parser.add_argument('-cc', '--conv2d_config', type=str,
                            metavar='CONV2D_CONFIG', nargs='*',
                            default=["{},128,relu,none,0.2".format(i) for i in range(1, 6)],
                            help='configuration for the convolution layer')
        # parser.add_argument('-hc', '--hidden_config', type=hidden_args,
        #                     metavar='(dim:activation:dropout)(;#1)*', default=None,
        #                     help='configuration for the hidden layer')
        parser.add_argument('-hc', '--hidden_config', type=str,
                            metavar='HIDDEN_CONFIG', nargs='*', default=None,
                            help='configuration for the hidden layer')
        training = parser.add_argument_group("configuration for the training")
        training.add_argument('-cx', '--ctx', type=args_context, metavar='CTX', default="c0",
                              help='device context: ([cg])(\d*) ex: c0')
        training.add_argument('-ep', '--epoch', type=int, metavar='EPOCH', default=50,
                            help='number of epochs')
        training.add_argument('-tb', '--trn_batch', type=int, metavar='TRN_BATCH', default=64,
                            help='batch size for training')
        training.add_argument('-db', '--dev_batch', type=int, metavar='DEV_BATCH', default=4096,
                            help='batch size for evaluation')
        training.add_argument('-lo', '--loss', type=args_loss, metavar='LOSS',
                              default='softmaxcrossentropyloss', help='loss function')
        training.add_argument('-op', '--optimizer', type=str, metavar='OPTIMIZER', default='adagrad',
                            help='optimizer algorithm')
        training.add_argument('-lr', '--learning_rate', type=float, metavar='LEARNING_RATE', default=0.01,
                            help='learning rate')
        training.add_argument('-wd', '--weight_decay', type=float, metavar='WEIGHT_DECAY', default=0.0,
                            help='weight decay')

        args = parser.parse_args(sys.argv[3:])

        conv2d_config = conv2d_args(args.conv2d_config)
        hidden_config = args_hidden(args.hidden_config)

        # cml arguments
        initializer = mx.init.Xavier(magnitude=2.24, rnd_type='gaussian')
        #
        # vector space models
        vsm_list = tuple((FastText(args.word_vsm), TOK) for _ in range(1))
        #
        # component
        comp = TokenSequenceTagger(args.ctx, vsm_list) if args.sequence else TokenBatchTagger(
            args.ctx, vsm_list)
        comp.init(self.key, args.chunking, args.num_class, args.feature_windows, args.input_dropout,
                  conv2d_config, hidden_config, initializer)

        # data
        reader, rargs = args.reader
        trn_docs = reader(args.trn_path, self.key, rargs)
        dev_docs = reader(args.dev_path, self.key, rargs)

        # train
        comp.train(trn_docs, dev_docs, args.model_path, args.trn_batch, args.dev_batch,
                   args.epoch, args.loss, args.optimizer, args.learning_rate, args.weight_decay)
        logging.info('# of classes: %d' % len(comp.label_map))

    def eval(self):
        # positional arguments:
        # These are required arguments. Positional arguments don't use hyphen or dash prefix.
        parser = argparse.ArgumentParser(description='Evaluation: Part-of-speech tagging',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parser.add_argument('dev_path', type=str, metavar='DEV_PATH',
                            help='path to the development data (input)')
        #   Lexicon
        parser.add_argument('word_vsm', type=str, metavar='WORD_VSM_PATH',
                            help='path to the vector space model for word embeddings')

        # Optional arguments:
        # These are optional arguments.
        #
        #   Generic
        parser.add_argument('-m', '--model_path', type=str, metavar='MODEL_PATH', default=None,
                            help='path to the model data (output)')
        parser.add_argument('-r', '--reader', type=args_reader, metavar='READER',
                            default="tsv:0,1",
                            help='reader configuration. format: (tsv|json)(:\\d,\\d)*')
        parser.add_argument('-seq', '--sequence', type=bool, metavar='BOOLEAN', default=False,
                            help='if set, use sequence mode')
        parser.add_argument('-db', '--dev_batch', type=int, metavar='DEV_BATCH', default=4096,
                              help='batch size for evaluation')

        args = parser.parse_args(sys.argv[3:])

        # vector space models
        vsm_list = tuple((FastText(args.word_vsm), TOK) for _ in range(1))

        # component
        comp = TokenSequenceTagger(args.ctx, vsm_list) if args.sequence else TokenBatchTagger(
            args.ctx, vsm_list)
        comp.load(args.model_path)

        # data
        reader, rargs = args.reader
        dev_docs = reader(args.dev_path, self.key, rargs)

        # decode
        states = comp.create_states(dev_docs)
        e = comp._evaluate(states, args.dev_batch)
        print(str(e))
