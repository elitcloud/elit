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
import datetime
import json
import logging
import pickle
import sys
import time
from types import SimpleNamespace
from typing import Tuple, Optional, Sequence, List

import mxnet as mx
from mxnet import nd, gluon, autograd
from mxnet.gluon import Trainer
from mxnet.gluon.data import Dataset, DataLoader
from tqdm import tqdm, trange

from elit.cli import ComponentCLI, set_logger
from elit.component import MXNetComponent
from elit.eval import F1, MxF1
from elit.model import FFNNModel
from elit.util.io import pkl, gln, json_reader, tsv_reader
from elit.util.mx import mxloss
from elit.util.structure import Document, to_gold, BILOU, TOK, DOC_ID
from elit.util.vsm import LabelMap, init_vsm

__author__ = 'Jinho D. Choi'


# ======================================== Dataset ========================================
class TokenTaggerDataset(Dataset):

    def __init__(self, vsms: List[SimpleNamespace],
                 key,
                 docs,
                 feature_windows,
                 label_map: LabelMap,
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
        self.pad = nd.zeros(sum([vsm.model.dim for vsm in self.vsms]))
        self.key = key
        self.doc = docs
        self.feature_windows = feature_windows
        self.label_map = label_map
        self.transform = transform
        self.init_data(docs)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        if self.transform is not None:
            return self.transform(x, y)
        else:
            return x, y

    def __len__(self):
        return len(self.data)

    def init_data(self, docs):
        for doc in tqdm(docs):
            for sen in tqdm(doc, desc="loading doc: {}".format(doc[DOC_ID]), leave=False):
                w = nd.array([i for i in zip(*[vsm.model.embedding_list(sen) for vsm in self.vsms])]).reshape(0, -1)
                for idx, (tok, label) in enumerate(zip(sen[TOK], sen[to_gold(self.key)])):
                    x = nd.stack(
                        *[w[idx + window] if 0 <= (idx + window) < len(w) else self.pad for window in self.feature_windows])
                    y = self.label_map.cid(label)
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
        self.pad = nd.zeros(sum([vsm.model.dim for vsm in self.vsms]))

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
             '- feature windows: %s' % str(self.feature_windows),
             '- %s' % str(self.model))
        return '\n'.join(s)

    # override
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
            self.feature_windows = pickle.load(fin)
            self.chunking = pickle.load(fin)
            self.input_config = pickle.load(fin)
            self.output_config = pickle.load(fin)
            self.fuse_conv_config = pickle.load(fin)
            self.ngram_conv_config = pickle.load(fin)
            self.hidden_configs = pickle.load(fin)

        self.model = FFNNModel(
            input_config=self.input_config,
            output_config=self.output_config,
            fuse_conv_config=self.fuse_conv_config,
            ngram_conv_config=self.ngram_conv_config,
            hidden_configs=self.hidden_configs,
            **kwargs)
        self.model.collect_params().load(gln(model_path), self.ctx)
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

        self.model.collect_params().save(gln(model_path))

    def train(self, trn_docs: Sequence[Document], dev_docs: Sequence[Document], model_path: str,
              label_map: LabelMap = None,
              epoch=100,
              trn_batch=64,
              dev_batch=2048,
              loss_fn=gluon.loss.SoftmaxCrossEntropyLoss(),
              optimizer='adagrad',
              optimizer_params=None,
              **kwargs):
        if optimizer_params is None:
            optimizer_params = {'learning_rate': 0.01}

        log = ('Configuration',
               '- context(s): {}'.format(self.ctx),
               '- trn_batch size: {}'.format(trn_batch),
               '- dev_batch size: {}'.format(dev_batch),
               '- max epoch : {}'.format(epoch),
               '- loss func : {}'.format(loss_fn),
               '- optimizer : {} <- {}'.format(optimizer, optimizer_params))
        logging.info('\n'.join(log))
        logging.info("Load trn data")
        trn_data = DataLoader(TokenTaggerDataset(self.vsms, self.key, trn_docs, self.feature_windows, self.label_map),
                              batch_size=trn_batch,
                              shuffle=True)
        logging.info("Load dev data")
        dev_data = DataLoader(TokenTaggerDataset(self.vsms, self.key, dev_docs, self.feature_windows, self.label_map),
                              batch_size=dev_batch,
                              shuffle=False)
        trainer = Trainer(self.model.collect_params(),
                          optimizer=optimizer,
                          optimizer_params=optimizer_params)
        smoothing_constant = .01
        moving_loss = 0

        logging.info('Training')
        best_e, best_eval = -1, -1
        self.model.hybridize()
        epochs = trange(1, epoch + 1)
        for e in epochs:
            st = time.time()
            for i, (data, label) in enumerate(tqdm(trn_data, leave=False)):
                data = data.as_in_context(self.ctx)
                label = label.as_in_context(self.ctx)
                with autograd.record():
                    output = self.model(data)
                    loss = loss_fn(output, label)
                loss.backward()
                trainer.step(data.shape[0])
                ##########################
                #  Keep a moving average of the losses
                ##########################
                curr_loss = nd.mean(loss).asscalar()
                moving_loss = (curr_loss if ((i == 0) and (e == 0))
                               else (1 - smoothing_constant) * moving_loss + smoothing_constant * curr_loss)
            et = time.time()
            if self.chunking:
                trn_acc = self.chunk_accuracy(trn_docs)
                dev_acc = self.chunk_accuracy(dev_docs)
            else:
                trn_acc = self.token_accuracy(trn_data)
                dev_acc = self.token_accuracy(dev_data)
            if best_eval < dev_acc:
                best_e, best_eval = e, dev_acc
                self.save(model_path=model_path)

            desc = ("epoch: {}".format(e),
                    "time: {}".format(datetime.timedelta(seconds=(et - st))),
                    "loss: {}".format(moving_loss),
                    "train_acc: {}".format(trn_acc),
                    "dev_acc: {}".format(dev_acc),
                    "best epoch: {}".format(best_e),
                    "best eval: {}".format(best_eval))
            epochs.set_description(desc=' '.join(desc))
        return best_eval

    def token_accuracy(self, data_iterator):
        acc = mx.metric.Accuracy()
        for data, label in data_iterator:
            data = data.as_in_context(self.ctx)
            label = label.as_in_context(self.ctx)
            output = self.model(data)
            preds = nd.argmax(output, axis=1)
            acc.update(preds=preds, labels=label)
        return acc.get()[1]

    def chunk_accuracy(self, docs):
        acc = ChunkF1()
        for doc in docs:
            for sen in doc:
                w = nd.array([i for i in zip(*[vsm.model.embedding_list(sen) for vsm in self.vsms])]).reshape(0, -1)
                x_batch = []
                labels = []
                for idx, (tok, label) in enumerate(zip(sen[TOK], sen[to_gold(self.key)])):
                    x = nd.stack(*[w[idx + window] if 0 <= (idx + window) < len(w) else self.pad for window in self.feature_windows])
                    x_batch.append(x)
                    labels.append(label)
                output = self.model(nd.stack(*x_batch))
                preds = nd.argmax(output, axis=1)
                preds = [self.label_map.get(int(pred.asscalar())) for pred in preds]
                acc.update(preds=preds, labels=labels)
        return acc.get()[1]


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
    def sqeuence(self):
        return self.source.get('sequence', False)

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

        ctx = config.ctx

        # component
        comp = TokenTagger(ctx, args.vsm_path)

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
        pass

    #     # create a arg-parser
    #     parser = argparse.ArgumentParser(
    #         description='Decode with the token tagger',
    #         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #     args = parser.parse_args(sys.argv[3:])
    #
    #     # data
    #     group = parser.add_argument_group("data arguments")
    #
    #     group.add_argument('-i', '--input_path', type=str, metavar='filepath',
    #                        help='filepath to the input data')
    #     group.add_argument(
    #         '-o',
    #         '--output_path',
    #         type=str,
    #         metavar='filepath',
    #         default=None,
    #         help='filepath to the output data (default: input_path.json)')
    #     group.add_argument('-m', '--model_path', type=str, metavar='filepath',
    #                        default=None,
    #                        help='filepath to the model data')
    #     group.add_argument(
    #         '-r',
    #         '--reader',
    #         type=args_reader,
    #         metavar='json or tsv=(str:int)(,#1)*',
    #         default=args_reader('json'),
    #         help='type of the reader and its configuration to match the data format (default: json)')
    #     group.add_argument(
    #         '-v',
    #         '--vsms',
    #         type=args_vsm,
    #         metavar='(fasttext|word2vec:key:filepath)( #1)*',
    #         nargs='+',
    #         help='list of (type of vector space model, key, filepath)')
    #     group.add_argument(
    #         '-l',
    #         '--log',
    #         type=str,
    #         metavar='filepath',
    #         default=None,
    #         help='filepath to the logging file; if not set, use stdout')
    #
    #     # evaluation
    #     group = parser.add_argument_group("arguments for decoding")
    #
    #     group.add_argument(
    #         '-cx',
    #         '--context',
    #         type=args_context,
    #         metavar='[cg]int',
    #         nargs='+',
    #         default=mx.cpu(),
    #         help='device context(s)')
    #     group.add_argument(
    #         '-ib',
    #         '--input_batch',
    #         type=int,
    #         metavar='int',
    #         default=2048,
    #         help='batch size for the input data')
    #
    #     # arguments
    #     args = parser.parse_args(args)
    #     set_logger(args.log)
    #
    #     args.vsms = [
    #         SimpleNamespace(
    #             model=n.type(
    #                 n.filepath),
    #             key=n.key) for n in args.vsms]
    #     if isinstance(args.context, list) and len(args.context) == 1:
    #         args.context = args.context[0]
    #     if args.output_path is None:
    #         args.output_path = args.input_path + '.json'
    #
    #     # component
    #     comp = TokenTagger(args.context, args.vsms)
    #     comp.load(args.model_path)
    #
    #     # data
    #     docs = args.reader.type(args.input_path, args.reader.params)
    #
    #     # evaluate
    #     comp.decode(docs, args.input_batch)
    #
    #     with open(args.output_path, 'w') as fout:
    #         json.dump(docs, fout)

    # override
    @classmethod
    def evaluate(cls):
        pass
    #     # create a arg-parser
    #     parser = argparse.ArgumentParser(
    #         description='Evaluate the token tagger',
    #         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #     args = parser.parse_args(sys.argv[3:])
    #
    #     # data
    #     group = parser.add_argument_group("data arguments")
    #
    #     group.add_argument('-d', '--dev_path', type=str, metavar='filepath',
    #                        help='filepath to the development data (input)')
    #     group.add_argument('-m', '--model_path', type=str, metavar='filepath',
    #                        default=None,
    #                        help='filepath to the model data')
    #     group.add_argument(
    #         '-r',
    #         '--reader',
    #         type=args_reader,
    #         metavar='json or tsv=(str:int)(,#1)*',
    #         default=args_reader('json'),
    #         help='type of the reader and its configuration to match the data format (default: json)')
    #     group.add_argument(
    #         '-v',
    #         '--vsms',
    #         type=args_vsm,
    #         metavar='(fasttext|word2vec:key:filepath)( #1)*',
    #         nargs='+',
    #         help='list of (type of vector space model, key, filepath)')
    #     group.add_argument(
    #         '-l',
    #         '--log',
    #         type=str,
    #         metavar='filepath',
    #         default=None,
    #         help='filepath to the logging file; if not set, use stdout')
    #
    #     # evaluation
    #     group = parser.add_argument_group("arguments for evaluation")
    #
    #     group.add_argument(
    #         '-cx',
    #         '--context',
    #         type=args_context,
    #         metavar='[cg]int',
    #         nargs='+',
    #         default=mx.cpu(),
    #         help='device context(s)')
    #     group.add_argument(
    #         '-db',
    #         '--dev_batch',
    #         type=int,
    #         metavar='int',
    #         default=2048,
    #         help='batch size for the development data')
    #
    #     # arguments
    #     args = parser.parse_args(args)
    #     set_logger(args.log)
    #
    #     args.vsms = [
    #         SimpleNamespace(
    #             model=n.type(
    #                 n.filepath),
    #             key=n.key) for n in args.vsms]
    #     if isinstance(args.context, list) and len(args.context) == 1:
    #         args.context = args.context[0]
    #
    #     # component
    #     comp = TokenTagger(args.context, args.vsms)
    #     comp.load(args.model_path)
    #
    #     # data
    #     dev_docs = args.reader.type(
    #         args.dev_path, args.reader.params, comp.key)
    #
    #     # evaluate
    #     comp.evaluate(dev_docs, args.dev_batch)
