# ========================================================================
# Copyright 2018 ELIT
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
import json
import sys

from elit.cli import ComponentCLI
from elit.nlp.embedding import init_emb
from elit.nlp.token_tagger import TokenTaggerConfig
from elit.util.logger import set_logger

__author__ = "Gary Lai"


class TokenTaggerCLI(ComponentCLI):
    def __init__(self):
        super().__init__('token_tagger', 'Token Tagger')

    # override
    @classmethod
    def train(cls):
        # create a arg-parser
        parser = argparse.ArgumentParser(description='Train a token tagger',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # data
        data_group = parser.add_argument_group("data arguments")

        data_group.add_argument('trn_path', type=str, help='filepath to the training data (input)')
        data_group.add_argument('dev_path', type=str, help='filepath to the development data (input)')
        data_group.add_argument('model_path', type=str, default=None, help='filepath to the model data (output); if not set, the model is not saved')
        data_group.add_argument('--embs_config', action='append', nargs='+', required=True, help='list of word embeddings models and file')

        # tagger
        tagger_group = parser.add_argument_group("tagger arguments")
        tagger_group.add_argument('mode', type=str, help='mode: cnn or rnn')
        tagger_group.add_argument('key', type=str, help='key to the document dictionary where the predicted tags are to be stored')

        # network
        network_group = parser.add_argument_group("network arguments")
        network_group.add_argument('config', type=str, help="path to config file")

        # arguments
        args = parser.parse_args(sys.argv[3:])

        with open(args.config, 'r') as d:
            config = TokenTaggerConfig(json.load(d))

        set_logger(config.log_path)

        trn_docs, label_map = config.reader(args.trn_path, config.tsv_heads, args.key)
        dev_docs, _ = config.reader(args.dev_path, config.tsv_heads, args.key)

        embs = [init_emb(config) for config in args.embs_config]
        if args.mode == 'rnn':
            from elit.nlp.token_tagger import RNNTokenTagger
            comp = RNNTokenTagger(ctx=config.ctx, key=args.key, label_map=label_map, embs=embs, chunking=config.chunking,
                                  rnn_config=config.rnn_config, output_config=config.output_config)
        elif args.mode == 'cnn':
            from elit.nlp.token_tagger import CNNTokenTagger
            comp = CNNTokenTagger(ctx=config.ctx, key=args.key, label_map=label_map, embs=embs, chunking=config.chunking,
                                  input_config=config.input_config,
                                  output_config=config.output_config,
                                  fuse_conv_config=config.fuse_conv_config,
                                  ngram_conv_config=config.ngram_conv_config,
                                  hidden_configs=config.hidden_configs)
        else:
            raise TypeError('mode {} is not supported.'.format(args.mode))

        comp.train(trn_docs=trn_docs, dev_docs=dev_docs, model_path=args.model_path, epoch=config.epoch,
                   trn_batch=config.trn_batch, dev_batch=config.dev_batch,
                   loss=config.loss, optimizer=config.optimizer, optimizer_params=config.optimizer_params)

    @classmethod
    def decode(cls):
        # create a arg-parser
        parser = argparse.ArgumentParser(
            description='Decode with the token tagger',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        # data
        group = parser.add_argument_group("data arguments")

        group.add_argument('input_path', type=str, metavar='INPUT_PATH', help='filepath to the input data')
        group.add_argument('output_path', type=str, metavar='OUTPUT_PATH', help='filepath to the output data')
        group.add_argument('model_path', type=str, metavar='MODEL_PATH', help='filepath to the model data')
        group.add_argument('--embs_config', action='append', nargs='+', required=True, help='list of word embeddings models and file')

        # tagger
        tagger_group = parser.add_argument_group("tagger arguments")
        tagger_group.add_argument('mode', type=str, help='mode: cnn or rnn')
        tagger_group.add_argument('key', type=str, help='key to the document dictionary where the predicted tags are to be stored')

        # network
        network_group = parser.add_argument_group("network arguments")
        network_group.add_argument('config', type=str, help="path to config file")

        args = parser.parse_args(sys.argv[3:])

        with open(args.config, 'r') as d:
            config = TokenTaggerConfig(json.load(d))

        set_logger(config.log_path)

        embs = [init_emb(config) for config in args.embs_config]

        comp = None
        if args.mode == 'rnn':
            from elit.nlp.token_tagger import RNNTokenTagger
            comp = RNNTokenTagger(ctx=config.ctx, key=args.key, embs=embs)
        elif args.mode == 'cnn':
            from elit.nlp.token_tagger import CNNTokenTagger
            comp = CNNTokenTagger(ctx=config.ctx, key=args.key, embs=embs)

        comp.load(args.model_path)

        docs, _ = config.reader(args.eval_path, config.tsv_heads, args.key)

        result = comp.decode(docs=docs, batch_size=config.batch_size)

        with open(args.output_path, 'w') as fout:
            json.dump(result, fout)

    @classmethod
    def evaluate(cls):
        # create a arg-parser
        parser = argparse.ArgumentParser(description='Evaluate the token tagger', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        # data
        group = parser.add_argument_group("data arguments")
        group.add_argument('eval_path', type=str, metavar='EVAL_PATH', help='filepath to the evaluation data')
        group.add_argument('model_path', type=str, metavar='MODEL_PATH', help='filepath to the model data')
        group.add_argument('--embs_config', action='append', nargs='+', required=True, help='list of word embeddings models and file')

        # tagger
        tagger_group = parser.add_argument_group("tagger arguments")
        tagger_group.add_argument('mode', type=str, help='mode: cnn or rnn')
        tagger_group.add_argument('key', type=str, help='key to the document dictionary where the predicted tags are to be stored')

        # network
        network_group = parser.add_argument_group("network arguments")
        network_group.add_argument('config', type=str, help="path to config file")

        args = parser.parse_args(sys.argv[3:])

        with open(args.config, 'r') as d:
            config = TokenTaggerConfig(json.load(d))

        set_logger(config.log_path)

        embs = [init_emb(config) for config in args.embs_config]

        comp = None
        if args.mode == 'rnn':
            from elit.nlp.token_tagger import RNNTokenTagger
            comp = RNNTokenTagger(ctx=config.ctx, key=args.key, embs=embs)
        elif args.mode == 'cnn':
            from elit.nlp.token_tagger import CNNTokenTagger
            comp = CNNTokenTagger(ctx=config.ctx, key=args.key, embs=embs)

        comp.load(args.model_path)

        eval_docs, _ = config.reader(args.eval_path, config.tsv_heads, args.key)

        acc, eva_time = comp.evaluate(docs=eval_docs, batch_size=config.batch_size)
        print(acc, eva_time)
