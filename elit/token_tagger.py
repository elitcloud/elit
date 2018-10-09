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
import json
import logging
import pickle
import sys
from types import SimpleNamespace
from typing import Tuple, Optional, List

import mxnet as mx
import numpy as np
from mxnet import nd
from mxnet.gluon.data import Dataset, DataLoader
from mxnet.metric import Accuracy
from tqdm import tqdm

from elit.cli import ComponentCLI, set_logger
from elit.component import MXNetComponent
from elit.eval import MxF1
from elit.model import FFNNModel
from elit.util.io import pkl, gln, json_reader, tsv_reader, params
from elit.util.mx import mxloss
from elit.util.structure import to_gold, BILOU, DOC_ID
from elit.util.vsm import LabelMap, init_vsm

__author__ = 'Jinho D. Choi, Gary Lai'


# ======================================== Dataset ========================================
class TokenTaggerDataset(Dataset):

    def __init__(self, vsms: List[SimpleNamespace],
                 key,
                 docs,
                 feature_windows,
                 label_map: LabelMap,
                 label: bool,
                 ctx=None,
                 transform=None):
        """

        :param vsms:
        :param key:
        :param docs:
        :param feature_windows:
        :param label_map:
        :param label:
        :param ctx:
        :param transform:
        """
        self.data = []
        self.vsms = vsms
        self.key = key
        self.feature_windows = feature_windows
        self.label_map = label_map
        self.label = label
        self.ctx = ctx
        self.transform = transform
        self.pad = nd.zeros(sum([vsm.model.dim for vsm in self.vsms]), ctx=self.ctx)
        self.init_data(docs)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        if self.transform is not None:
            return self.transform(x, y)
        else:
            return x, y

    def __len__(self):
        return len(self.data)

    def extract_x(self, idx, w):
        return nd.stack(*[w[idx + win] if 0 <= (idx + win) < len(w) else self.pad for win in self.feature_windows])

    def extract_y(self, label):
        if label is False:
            return -1
        return self.label_map.cid(label)

    def extract_sen(self, sen):
        return nd.array([np.concatenate(i) for i in zip(*[vsm.model.embedding_list(sen.tokens) for vsm in self.vsms])],
                        ctx=self.ctx).reshape(
            0, -1)

    def init_data(self, docs):
        for doc in tqdm(docs):
            for sen in tqdm(doc, desc="loading doc: {}".format(doc[DOC_ID]), leave=False):
                w = self.extract_sen(sen)
                if self.label:
                    for idx, label in enumerate(sen[to_gold(self.key)]):
                        x = self.extract_x(idx, w)
                        y = self.extract_y(label)
                        self.data.append((x, y))
                else:
                    for idx, _ in enumerate(sen):
                        x = self.extract_x(idx, w)
                        y = self.extract_y(self.label)
                        self.data.append((x, y))


# ======================================== Component ========================================

class TokenTagger(MXNetComponent):
    """
    :class:`TokenTagger` provides an abstract class to implement a tagger that predicts a tag for every token.
    """

    def __init__(self, ctx: mx.Context, vsm_path: list):
        """
        :param ctx: the (list of) device context(s) for :class:`mxnet.gluon.Block`.
        :param vsms: the sequence of namespace(model, key),
                         where the key indicates the key of the values to retrieve embeddings for (e.g., tok).
        """
        super().__init__(ctx)
        self.vsms = [init_vsm(n) for n in vsm_path]
        self.pad = nd.zeros(sum([vsm.model.dim for vsm in self.vsms]), ctx=self.ctx)

        # to be loaded/saved
        self.key = None
        self.model = None
        self.label_map = LabelMap()
        self.chunking = None
        self.feature_windows = None
        self.input_config = None
        self.output_config = None
        self.fuse_conv_config = None
        self.ngram_conv_config = None
        self.hidden_configs = None
        self.initializer = None

    def __str__(self):
        s = ('Token Tagger',
             '- key: {}'.format(self.key),
             '- label_map: {}'.format(self.label_map),
             '- chunking: {}'.format(self.chunking),
             '- feature windows: {}'.format(self.feature_windows),
             '- %s' % str(self.model))
        return '\n'.join(s)

    def init(self,
             key: str,
             feature_windows: Tuple[int, ...],
             label_map: LabelMap,
             chunking: bool,
             input_config: Optional[SimpleNamespace] = SimpleNamespace(dropout=0.0),
             output_config: Optional[SimpleNamespace] = None,
             fuse_conv_config: Optional[SimpleNamespace] = None,
             ngram_conv_config: Optional[SimpleNamespace] = None,
             hidden_configs: Optional[Tuple[SimpleNamespace]] = None,
             initializer: mx.init.Initializer = mx.init.Xavier(magnitude=2.24, rnd_type='gaussian'),
             **kwargs):
        """
        :param key: the key to each sentence where the tags are to be saved.
        :param feature_windows: the content windows for feature extraction.
        :param label_map:
        :param input_config: the dropout rate to be applied to the input layer.
        :param output_config:
        :param fuse_conv_config: the configuration for the fuse convolution layer.
        :param ngram_conv_config: the configuration for the n-gram convolution layer.
        :param hidden_configs: the configurations for the hidden layers.
        :param initializer: the weight initializer for :class:`mxnet.gluon.Block`.
        :param kwargs: extra parameters to initialize :class:`mxnet.gluon.Block`.
        """
        # configuration
        self.key = key
        self.label_map = label_map
        self.chunking = chunking
        self.feature_windows = feature_windows
        self.initializer = initializer

        input_config.col = sum([vsm.model.dim for vsm in self.vsms])
        input_config.row = len(feature_windows)
        output_config.num_class = len(self.label_map)

        # initialization
        self.input_config = input_config
        self.output_config = output_config
        self.fuse_conv_config = fuse_conv_config
        self.ngram_conv_config = ngram_conv_config
        self.hidden_configs = hidden_configs

        self.model = FFNNModel(
            input_config=self.input_config,
            output_config=self.output_config,
            fuse_conv_config=self.fuse_conv_config,
            ngram_conv_config=self.ngram_conv_config,
            hidden_configs=self.hidden_configs,
            **kwargs)
        self.model.collect_params().initialize(self.initializer, ctx=self.ctx)
        logging.info(self.__str__())

    # override
    def load(self, model_path: str, **kwargs):
        """
        :param model_path: the path to a pre-trained model to be loaded.
        :type model_path: str
        """
        with open(pkl(model_path), 'rb') as fin:
            self.key = pickle.load(fin)
            self.label_map = pickle.load(fin)
            self.chunking = pickle.load(fin)
            self.feature_windows = pickle.load(fin)
            self.input_config = pickle.load(fin)
            self.output_config = pickle.load(fin)
            self.fuse_conv_config = pickle.load(fin)
            self.ngram_conv_config = pickle.load(fin)
            self.hidden_configs = pickle.load(fin)
        logging.info('{} is loaded'.format(pkl(model_path)))
        self.model = FFNNModel(
            input_config=self.input_config,
            output_config=self.output_config,
            fuse_conv_config=self.fuse_conv_config,
            ngram_conv_config=self.ngram_conv_config,
            hidden_configs=self.hidden_configs,
            **kwargs)
        self.model.load_params(params(model_path), self.ctx)
        logging.info('{} is loaded'.format(params(model_path)))
        logging.info(self.__str__())

    # override
    def save(self, model_path, **kwargs):
        """
        :param model_path: the filepath where the model is to be saved.
        :type model_path: str
        """
        with open(pkl(model_path), 'wb') as fout:
            pickle.dump(self.key, fout)
            pickle.dump(self.label_map, fout)
            pickle.dump(self.chunking, fout)
            pickle.dump(self.feature_windows, fout)
            pickle.dump(self.input_config, fout)
            pickle.dump(self.output_config, fout)
            pickle.dump(self.fuse_conv_config, fout)
            pickle.dump(self.ngram_conv_config, fout)
            pickle.dump(self.hidden_configs, fout)
        logging.info('{} is saved'.format(pkl(model_path)))
        self.model.save_params(params(model_path))
        logging.info('{} is saved'.format(params(model_path)))

    def accuracy(self, data_iterator, docs=None):
        return self.chunk_accuracy(data_iterator, docs) if self.chunking else self.token_accuracy(data_iterator)

    def token_accuracy(self, data_iterator):
        acc = Accuracy()
        for data, label in data_iterator:
            data = data.as_in_context(self.ctx)
            label = label.as_in_context(self.ctx)
            output = self.model(data)
            preds = nd.argmax(output, axis=1)
            acc.update(preds=preds, labels=label)
        return acc.get()[1]

    def chunk_accuracy(self, data_iterator, docs):
        acc = ChunkF1()
        preds = []
        idx = 0
        for data, label in data_iterator:
            data = data.as_in_context(self.ctx)
            output = self.model(data)
            pred = nd.argmax(output, axis=1)
            [preds.append(self.label_map.get(int(p.asscalar()))) for p in pred]

        for doc in docs:
            for sen in doc:
                labels = sen[to_gold(self.key)]
                acc.update(labels=labels, preds=preds[idx:idx + len(labels)])
                idx += len(labels)
        return acc.get()[1]

    def data_loader(self, docs, batch_size, shuffle=False, label=True):
        return DataLoader(TokenTaggerDataset(vsms=self.vsms,
                                             key=self.key,
                                             docs=docs,
                                             feature_windows=self.feature_windows,
                                             label_map=self.label_map,
                                             label=label),
                          batch_size=batch_size,
                          shuffle=shuffle)


# ======================================== EvalMetric ========================================

class ChunkF1(MxF1):

    def update(self, labels, preds):
        gold = BILOU.to_chunks(labels)
        pred = BILOU.to_chunks(preds)
        self.correct += len(set.intersection(set(gold), set(pred)))
        self.p_total += len(pred)
        self.r_total += len(gold)


class TokenTaggerConfig(object):

    def __init__(self, source: dict):
        self.source = source

    def __str__(self):
        return str(self.source)

    @property
    def reader(self):
        return tsv_reader if self.source.get('reader', 'tsv').lower() == 'tsv' else json_reader

    @property
    def log_path(self):
        return self.source.get('log_path', None)

    @property
    def tsv_heads(self):
        return self.source['tsv_heads'] if self.source.get('reader', 'tsv').lower() == 'tsv' else None

    @property
    def chunking(self):
        return self.source.get('chunking', False)

    @property
    def feature_windows(self):
        return self.source.get('feature_windows', [3, 2, 1, 0, -1, -2, -3])

    @property
    def position_embedding(self):
        return self.source.get('position_embedding', False)

    @property
    def label_embedding(self):
        return self.source.get('label_embedding', False)

    @property
    def input_config(self):
        assert self.source.get('input_config') is not None

        return SimpleNamespace(
            dropout=self.source['input_config'].get('dropout', 0.0)
        )

    @property
    def output_config(self):
        assert self.source.get('output_config') is not None

        return SimpleNamespace(
            dropout=self.source['output_config'].get('dropout', 0.0)
        )

    @property
    def fuse_conv_config(self):
        if self.source.get('fuse_conv_config') is None:
            return None
        return SimpleNamespace(
            filters=self.source['fuse_conv_config'].get('filters', 128),
            activation=self.source['fuse_conv_config'].get('activation', 'relu'),
            pool=self.source['fuse_conv_config'].get('pool', None),
            dropout=self.source['fuse_conv_config'].get('dropout', 0.2)
        )

    @property
    def ngram_conv_cofig(self):
        if self.source.get('ngram_conv_config') is None:
            return None
        return SimpleNamespace(
            ngrams=self.source['ngram_conv_config'].get('ngrams', [1, 2, 3, 4, 5]),
            filters=self.source['ngram_conv_config'].get('filters', 128),
            activation=self.source['ngram_conv_config'].get('activation', 'relu'),
            pool=self.source['ngram_conv_config'].get('pool', None),
            dropout=self.source['ngram_conv_config'].get('dropout', 0.2)
        )

    @property
    def hidden_configs(self):
        if self.source.get('hidden_configs') is None:
            return None

        def hidden_layer(config):
            return SimpleNamespace(**config)

        return [hidden_layer(config) for config in self.source.get('hidden_configs')]

    @property
    def ctx(self):
        device = self.source.get('device', 'cpu')
        core = self.source.get('device', 0)
        if device.lower() == 'cpu':
            return mx.cpu()
        else:
            return mx.gpu()

    @property
    def epoch(self):
        return self.source.get('epoch', 100)

    @property
    def batch_size(self):
        return self.source.get('batch_size', 2048)

    @property
    def trn_batch(self):
        return self.source.get('trn_batch', 64)

    @property
    def dev_batch(self):
        return self.source.get('dev_batch', 2048)

    @property
    def loss(self):
        return mxloss(self.source.get('loss', 'softmaxcrossentropyloss'))

    @property
    def optimizer(self):
        return self.source.get('optimizer', 'adagrad')

    @property
    def optimizer_params(self):
        return self.source.get('optimizer_params', {})


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

        data_group.add_argument('trn_path', type=str, metavar='TRN_PATH', help='filepath to the training data (input)')
        data_group.add_argument('dev_path', type=str, metavar='DEV_PATH',
                                help='filepath to the development data (input)')
        data_group.add_argument('model_path', type=str, metavar='MODEL_PATH', default=None,
                                help='filepath to the model data (output); if not set, the model is not saved')
        data_group.add_argument('--vsm_path', action='append', nargs='+', metavar='VSM_PATH', required=True,
                                help='vsm path')

        # tagger
        tagger_group = parser.add_argument_group("tagger arguments")

        tagger_group.add_argument('key', type=str, metavar='KEY',
                                  help='key to the document dictionary where the predicted tags are to be stored')

        # network
        network_group = parser.add_argument_group("network arguments")
        network_group.add_argument('config', type=str, metavar='CONFIG', help="config file")

        # arguments
        args = parser.parse_args(sys.argv[3:])
        # print(args)
        with open(args.config, 'r') as d:
            config = TokenTaggerConfig(json.load(d))

        set_logger(config.log_path)

        trn_docs, label_map = config.reader(args.trn_path, config.tsv_heads, args.key)
        dev_docs, _ = config.reader(args.dev_path, config.tsv_heads, args.key)

        # component
        comp = TokenTagger(config.ctx, args.vsm_path)

        comp.init(
            key=args.key,
            feature_windows=config.feature_windows,
            label_map=label_map,
            chunking=config.chunking,
            input_config=config.input_config,
            output_config=config.output_config,
            fuse_conv_config=config.fuse_conv_config,
            ngram_conv_config=config.ngram_conv_cofig,
            hidden_configs=config.hidden_configs)
        # train
        comp.train(
            trn_docs=trn_docs,
            dev_docs=dev_docs,
            model_path=args.model_path,
            label_map=label_map)
        logging.info('# of classes: %d' % len(comp.label_map))

    # override
    @classmethod
    def decode(cls):
        # create a arg-parser
        parser = argparse.ArgumentParser(
            description='Decode with the token tagger',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        # data
        group = parser.add_argument_group("data arguments")

        group.add_argument('input_path', type=str, metavar='INPUT_PATH',
                           help='filepath to the input data')
        group.add_argument('output_path', type=str, metavar='OUTPUT_PATH',
                           help='filepath to the output data')
        group.add_argument('model_path', type=str, metavar='MODEL_PATH',
                           help='filepath to the model data')
        group.add_argument('--vsm_path', action='append', nargs='+', metavar='VSM_PATH', required=True,
                           help='vsm path')

        # network
        network_group = parser.add_argument_group("network arguments")
        network_group.add_argument('config', type=str, metavar='CONFIG', help="config file")

        args = parser.parse_args(sys.argv[3:])

        with open(args.config, 'r') as d:
            config = TokenTaggerConfig(json.load(d))

        set_logger(config.log_path)

        # component
        comp = TokenTagger(config.ctx, args.vsm_path)
        comp.load(args.model_path)
        docs, _ = config.reader(args.input_path, config.tsv_heads, comp.key)
        result = comp.decode(docs=docs, batch_size=config.batch_size)

        with open(args.output_path, 'w') as fout:
            json.dump(result, fout)

    # override
    @classmethod
    def evaluate(cls):
        # create a arg-parser
        parser = argparse.ArgumentParser(
            description='Evaluate the token tagger',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        # data
        group = parser.add_argument_group("data arguments")

        group.add_argument('eval_path', type=str, metavar='EVAL_PATH',
                           help='filepath to the evaluation data')
        group.add_argument('model_path', type=str, metavar='MODEL_PATH',
                           help='filepath to the model data')
        group.add_argument('--vsm_path', action='append', nargs='+', metavar='VSM_PATH', required=True,
                           help='vsm path')

        # network
        network_group = parser.add_argument_group("network arguments")
        network_group.add_argument('config', type=str, metavar='CONFIG', help="config file")

        args = parser.parse_args(sys.argv[3:])

        with open(args.config, 'r') as d:
            config = TokenTaggerConfig(json.load(d))

        set_logger(config.log_path)

        # component
        comp = TokenTagger(config.ctx, args.vsm_path)
        comp.load(args.model_path)
        docs, _ = config.reader(args.eval_path, config.tsv_heads, comp.key)
        comp.evaluate(docs=docs, batch_size=config.batch_size)
