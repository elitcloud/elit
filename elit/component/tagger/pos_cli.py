# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-05-31 11:02
import argparse
import json
import sys

from elit.cli import ComponentCLI
from elit.component.tagger.corpus import conll_to_documents
from elit.component.tagger.pos_tagger import POSTagger
from elit.component.tokenizer import Tokenizer, EnglishTokenizer
from elit.resources.pre_trained_models import POS_JUMBO
from elit.structure import Document, Sentence
from elit.util.io import eprint, merge_args_with_config


class POSTaggerCLI(ComponentCLI):
    def __init__(self):
        super().__init__(name='pos_tagger', description='POS Tagger')

    @classmethod
    def train(cls):
        parser = argparse.ArgumentParser(description='Train a pos tagger')
        parser.add_argument('--config_path', type=str, help='config file in json format')
        parser.add_argument('--train_path', type=str, help='file path to the training set')
        parser.add_argument('--dev_path', type=str, help='file path to the development set')
        parser.add_argument('--model_path', type=str, help='file path where the model will be saved')
        parser.add_argument('--word_embeddings', type=str, default=':'.join(('fasttext', 'crawl-300d-2M-subword')),
                            help='word embeddings to use')
        parser.add_argument('--flair_embeddings', dest='flair', action='store_false', help='use Flair embeddings')
        parser.add_argument('--learning_rate', type=float, default=0.1)
        parser.add_argument('--mini_batch_size', type=int, default=32)
        parser.add_argument('--max_epochs', type=int, default=100)
        parser.add_argument('--anneal_factor', type=float, default=0.5)
        parser.add_argument('--embeddings_in_memory', action='store_true', help='store embeddings in GPU memory')

        args = None
        try:
            args = parser.parse_args(sys.argv[3:])
            args = merge_args_with_config(args)
            args['pretrained_embeddings'] = args['word_embeddings'].split(':')
            for k in ['train_path', 'dev_path', 'model_path']:
                if not args[k]:
                    eprint('--{} is required'.format(k))
                    exit(1)
            args['trn_docs'] = conll_to_documents(args['train_path'], headers={0: 'text', 1: 'pos'})
            args['dev_docs'] = conll_to_documents(args['dev_path'], headers={0: 'text', 1: 'pos'})
        except SystemExit:
            parser.print_help()
            exit(1)
        tagger = POSTagger()
        tagger.train(**args)

    @classmethod
    def decode(cls):
        parser = argparse.ArgumentParser(description='Use a pos tagger to decode raw text')
        parser.add_argument('--model_path', type=str, default=POS_JUMBO,
                            help='file path to the saved model')
        args = None
        try:
            args = parser.parse_args(sys.argv[3:])
        except SystemExit:
            parser.print_help()
            exit(1)
        tagger = POSTagger()
        tagger.load(args.model_path)
        components = [EnglishTokenizer(), tagger]
        for line in sys.stdin:
            line = line.strip()
            docs = line
            for c in components:
                docs = c.decode(docs)
            for d in docs:  # type: Document
                for sent in d:  # type: Sentence
                    print(' '.join(
                        '{}/{}'.format(word, pos) for word, pos in zip(sent.tokens, sent.part_of_speech_tags)))

    @classmethod
    def evaluate(cls):
        parser = argparse.ArgumentParser(description='Evaluate a pos tagger')
        parser.add_argument('--model_path', type=str, default=POS_JUMBO,
                            help='file path to the saved model')
        parser.add_argument('--test_path', type=str, required=True, help='gold file in tsv format')
        args = None
        try:
            args = parser.parse_args(sys.argv[3:])
        except SystemExit:
            parser.print_help()
            exit(1)
        tagger = POSTagger()
        tagger.load(args.model_path)
        tagger.evaluate(conll_to_documents(args.test_path, headers={0: 'text', 1: 'pos'}))
