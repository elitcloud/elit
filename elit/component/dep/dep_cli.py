# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-05-31 11:02
import argparse
import sys

from elit.cli import ComponentCLI
from elit.component.dep.common.utils import _load_conll
from elit.component.dep.dependency_parser import DEPBiaffineParser
from elit.component.tagger.corpus import conll_to_documents
from elit.component.tagger.pos_tagger import POSFlairTagger
from elit.component.tokenizer import EnglishTokenizer
from elit.resources.pre_trained_models import ELIT_POS_FLAIR_EN_MIXED, ELIT_DEP_BIAFFINE_EN_MIXED
from elit.structure import Document, Sentence
from elit.util.io import eprint, merge_args_with_config


class DependencyParserCLI(ComponentCLI):
    def __init__(self):
        super().__init__(name='dep_parser', description='Dependency Parser')

    @classmethod
    def train(cls):
        parser = argparse.ArgumentParser(description='Train a dependency parser')
        parser.add_argument('--config_path', type=str, help='config file in json format')
        parser.add_argument('--train_path', type=str, help='file path to the training set')
        parser.add_argument('--dev_path', type=str, help='file path to the development set')
        parser.add_argument('--model_path', type=str, help='file path where the model will be saved')
        parser.add_argument('--word_embeddings', type=str, default=':'.join(('fasttext', 'crawl-300d-2M-subword')),
                            help='word embeddings to use')
        parser.add_argument('--min_occur_count', type=int, default=2,
                            help='threshold of rare words, which will be replaced with UNKs')
        parser.add_argument('--lstm_layers', type=int, default=3, help='layers of lstm')
        parser.add_argument('--word_dims', type=int, default=100, help='dimension of word embedding')
        parser.add_argument('--tag_dims', type=int, default=100, help='dimension of tag embedding')
        parser.add_argument('--dropout_emb', type=float, default=0.33, help='dimension of tag embedding')
        parser.add_argument('--lstm_hiddens', type=int, default=400, help='size of lstm hidden states')
        parser.add_argument('--dropout_lstm_input', type=float, default=0.33, help='dropout on x in variational RNN')
        parser.add_argument('--dropout_lstm_hidden', type=float, default=0.33, help='dropout on h in variational RNN')
        parser.add_argument('--mlp_arc_size', type=int, default=500,
                            help='output size of MLP for arc feature extraction')
        parser.add_argument('--mlp_rel_size', type=int, default=500,
                            help='output size of MLP for rel feature extraction')
        parser.add_argument('--dropout_mlp', type=float, default=0.33, help='dropout on the output of LSTM')
        parser.add_argument('--learning_rate', type=float, default=2e-3, help='learning rate')
        parser.add_argument('--decay', type=float, default=.75, help='see ExponentialScheduler')
        parser.add_argument('--decay_steps', type=int, default=5000, help='see ExponentialScheduler')
        parser.add_argument('--beta_1', type=float, default=.9, help='see ExponentialScheduler')
        parser.add_argument('--beta_2', type=float, default=.9, help='see ExponentialScheduler')
        parser.add_argument('--epsilon', type=float, default=1e-12, help='see ExponentialScheduler')
        parser.add_argument('--num_buckets_train', type=int, default=40, help='number of buckets for training data set')
        parser.add_argument('--num_buckets_valid', type=int, default=10, help='number of buckets for dev data set')
        parser.add_argument('--num_buckets_test', type=int, default=10, help='number of buckets for test data set')
        parser.add_argument('--train_iters', type=int, default=50000, help='training iterations')
        parser.add_argument('--train_batch_size', type=int, default=5000, help='training batch size')
        parser.add_argument('--validate_every', type=int, default=100,
                            help='validate on dev set every such number of batches')
        parser.add_argument('--save_after', type=int, default=5000, help='skip saving model in early epochs')

        args = None
        try:
            args = parser.parse_args(sys.argv[3:])
            args = merge_args_with_config(args)
            args['pretrained_embeddings'] = args['word_embeddings'].split(':')
            for k in ['train_path', 'dev_path', 'model_path']:
                if not args[k]:
                    eprint('--{} is required'.format(k))
                    exit(1)
            args['trn_docs'] = [_load_conll(args['train_path'])]
            args['dev_docs'] = [_load_conll(args['dev_path'])]
            args['save_dir'] = args['model_path']
        except SystemExit:
            parser.print_help()
            exit(1)
        dep_parser = DEPBiaffineParser()
        dep_parser.train(**args)

    @classmethod
    def decode(cls):
        parser = argparse.ArgumentParser(description='Use a dependency parser to decode raw text')
        parser.add_argument('--model_path', type=str, default=ELIT_DEP_BIAFFINE_EN_MIXED,
                            help='file path to the saved model')
        args = None
        try:
            args = parser.parse_args(sys.argv[3:])
        except SystemExit:
            parser.print_help()
            exit(1)

        this_module = DEPBiaffineParser()
        this_module.load(args.model_path)
        pos_tagger = POSFlairTagger()
        pos_tagger.load(ELIT_POS_FLAIR_EN_MIXED)
        components = [EnglishTokenizer(), pos_tagger, this_module]
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            docs = line
            for c in components:
                docs = c.decode(docs)
            for d in docs:  # type: Document
                for sent in d.to_conll():
                    print(sent)

    @classmethod
    def evaluate(cls):
        parser = argparse.ArgumentParser(description='Evaluate a pos tagger')
        parser.add_argument('--model_path', type=str, default=ELIT_DEP_BIAFFINE_EN_MIXED,
                            help='file path to the saved model')
        parser.add_argument('--test_path', type=str, required=True, help='gold file in conll format')
        args = None
        try:
            args = parser.parse_args(sys.argv[3:])
        except SystemExit:
            parser.print_help()
            exit(1)
        this_module = DEPBiaffineParser()
        this_module.load(args.model_path)
        this_module.evaluate(test_file=args.test_path)
