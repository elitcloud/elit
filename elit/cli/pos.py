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
import sys

__author__ = "Gary Lai"


class PosCli(object):

    def __init__(self):
        parser = argparse.ArgumentParser(
            usage='''
    elit pos <command> [<args>]

commands:
    train      train part-of-speech tagger
        '''
        )
        parser.add_argument('command', help='command to run')
        args = parser.parse_args(sys.argv[2:])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            parser.print_help()
            exit(1)
        getattr(self, args.command)()

    def pos_args(self):
        from types import SimpleNamespace
        from ..util import tsv_reader, json_reader
        from ..model import conv2d_args
        from ..model import hidden_args

        def reader(s):
            """
            :param s: (tsv|json)(;\d:\d)*
            :return: reader, SimpleNamespace(tok, pos)
            """
            r = s.split(';')
            if r[0] == 'tsv':
                t = r[1].split(':')
                return tsv_reader, SimpleNamespace(tok=int(t[0]), pos=int(t[1]))
            else:
                return json_reader, None

        def int_tuple(s):
            """
            :param s: \d(,\d)*
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
        parser.add_argument('-r', '--reader', type=reader, metavar='(tsv|json)(;\d:\d)*',
                            default=(tsv_reader, SimpleNamespace(tok=0, pos=1)),
                            help='reader configuration')

        # lexicon
        parser.add_argument('-wv', '--word_vsm', type=str, metavar='filepath',
                            help='vector space model for word embeddings')
        parser.add_argument('-av', '--ambi_vsm', type=str, metavar='filepath', default=None,
                            help='vector space model for ambiguity classes')

        # configuration
        parser.add_argument('-le', '--label_embedding', type=bool, metavar='boolean', default=False,
                            help='if set, use label embeddings as features')
        parser.add_argument('-fw', '--feature_windows', type=int_tuple, metavar='int[,int]*',
                            default=tuple(range(-3, 4)),
                            help='contextual windows for feature extraction')
        parser.add_argument('-nc', '--num_class', type=int, metavar='int', default=50,
                            help='number of classes (part-of-speech tags)')
        parser.add_argument('-ir', '--input_dropout', type=float, metavar='float', default=0.0,
                            help='dropout rate applied to the input layer')
        parser.add_argument('-cc', '--conv2d_config', type=conv2d_args,
                            metavar='(ngram:filters:activation:pool:dropout)(;#1)*',
                            default=tuple(SimpleNamespace(ngram=i, filters=128, activation='relu',
                                                          pool='avg', dropout=0.2) for i in
                                          range(1, 5)),
                            help='configuration for the convolution layer')
        parser.add_argument('-hc', '--hidden_config', type=hidden_args,
                            metavar='(dim:activation:dropout)(;#1)*', default=None,
                            help='configuration for the hidden layer')

        # training
        parser.add_argument('-cx', '--ctx', type=str, metavar='[cg]\d', default='c0',
                            help='device context')
        parser.add_argument('-ep', '--epoch', type=int, metavar='int', default=50,
                            help='number of epochs')
        parser.add_argument('-tb', '--trn_batch', type=int, metavar='int', default=64,
                            help='batch size for training')
        parser.add_argument('-db', '--dev_batch', type=int, metavar='int', default=4096,
                            help='batch size for evaluation')
        parser.add_argument('-op', '--optimizer', type=str, metavar='str', default='adagrad',
                            help='optimizer algorithm')
        parser.add_argument('-lr', '--learning_rate', type=float, metavar='float', default=0.01,
                            help='learning rate')
        parser.add_argument('-wd', '--weight_decay', type=float, metavar='float', default=0.0,
                            help='weight decay')

        args = parser.parse_args(sys.argv[3:])
        return args

    def train(self):
        args = self.pos_args()

        # vector space models
        from ..vsm import FastText
        from ..pos import POSTagger
        vsm_list = [FastText(args.word_vsm)]
        if args.ambi_vsm:
            from ..vsm import Word2Vec
            vsm_list.append(Word2Vec(args.ambi_vsm))

        # component
        comp = POSTagger(args.ctx, vsm_list)
        comp.init(args.label_embedding, args.feature_windows, args.num_class, args.input_dropout,
                  args.conv2d_config, args.hidden_config)

        # data
        reader, reader_args = args.reader
        trn_data = reader(args.trn_path, reader_args)
        dev_data = reader(args.dev_path, reader_args)

        # train
        comp.train(trn_data, dev_data, args.model_path, args.trn_batch, args.dev_batch, args.epoch,
                   args.optimizer, args.learning_rate, args.weight_decay)

    def evaluate(self):
        args = self.pos_args()

        # vector space models
        from ..vsm import FastText
        from ..pos import POSTagger
        from ..util import group_states
        vsm_list = [FastText(args.word_vsm)]
        if args.ambi_vsm:
            from ..vsm import Word2Vec
            vsm_list.append(Word2Vec(args.ambi_vsm))

        # component
        comp = POSTagger(args.ctx, vsm_list)
        comp.load(args.model_path)

        # data
        reader, reader_args = args.reader
        dev_data = reader(args.dev_path, reader_args)

        # decode
        states = group_states(dev_data, comp.create_state)
        e = comp._evaluate(states, reset=True)
        print('DEV: %5.2f (%d/%d)' % (e.get(), e.correct, e.total))


if __name__ == '__main__':
    PosCli()
