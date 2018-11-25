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
import logging
import pickle
from types import SimpleNamespace
from typing import Sequence, Optional, Tuple, List

import mxnet as mx
from mxnet import nd, autograd
from mxnet.gluon.data import DataLoader
from mxnet.metric import Accuracy
from tqdm import tqdm

from elit.component import CNNComponent
from elit.dataset import LabelMap, TokensDataset
from elit.nlp.embedding import Embedding
from elit.eval import ChunkF1
from elit.model import CNNModel
from elit.structure import Document, to_gold
from elit.util.io import pkl, params

__author__ = 'Jinho D. Choi, Gary Lai'


class CNNTokenTagger(CNNComponent):

    def __init__(self, ctx: mx.Context, key: str, embs: List[Embedding],
                 feature_windows=(3, 2, 1, 0, -1, -2, -3),
                 input_config: Optional[SimpleNamespace] = None,
                 output_config: Optional[SimpleNamespace] = None,
                 fuse_conv_config: Optional[SimpleNamespace] = None,
                 ngram_conv_config: Optional[SimpleNamespace] = None,
                 hidden_configs: Optional[Tuple[SimpleNamespace]] = None,
                 initializer: mx.init.Initializer = mx.init.Xavier(magnitude=2.24, rnd_type='gaussian'),
                 label_map: LabelMap = None, chunking: bool = False,
                 **kwargs):
        """

        :param ctx:
        :param key:
        :param embs_config:
        :param label_map:
        :param chunking:
        :param feature_windows:
        :param input_config:
        :param output_config:
        :param fuse_conv_config:
        :param ngram_conv_config:
        :param hidden_configs:
        :param initializer:
        :param kwargs:
        """
        self.chunking = chunking
        self.feature_windows = feature_windows
        self.label_map = label_map
        if input_config is not None:
            input_config.col = sum([emb.dim for emb in embs])
            input_config.row = len(self.feature_windows)
        if output_config is not None:
            output_config.num_class = self.label_map.num_class() if label_map else 1

        super().__init__(ctx, key, embs, input_config, output_config,
                         fuse_conv_config, ngram_conv_config, hidden_configs, initializer, **kwargs)

    def __str__(self):
        s = ('CNNTokenTagger',
             '- context: {}'.format(self.ctx),
             '- key: {}'.format(self.key),
             '- embs: {}'.format(self.embs),
             '- label_map: {}'.format(self.label_map),
             '- chunking: {}'.format(self.chunking),
             '- feature windows: {}'.format(self.feature_windows),
             '- initializer: {}'.format(self.initializer),
             '- model: {}'.format(str(self.model)))
        return '\n'.join(s)

    def train_block(self, data_iter: DataLoader, docs: Sequence[Document]) -> float:
        """

        :param data_iter:
        :param sens:
        :return:
        """
        acc = Accuracy()
        for data, label in tqdm(data_iter, leave=False):
            data = data.as_in_context(self.ctx)
            label = label.as_in_context(self.ctx)
            with autograd.record():
                output = self.model(data)
                l = self.loss(output, label)
            l.backward()
            for preds, labels in zip(nd.argmax(output, axis=1), label):
                acc.update(labels=labels, preds=preds)
            self.trainer.step(data.shape[0])
        return float(acc.get()[1])

    def decode_block(self, data_iter: DataLoader, docs: Sequence[Document], **kwargs):
        """

        :param data_iter:
        :param docs:
        :param kwargs:
        """
        preds = []
        for data, label in data_iter:
            data = data.as_in_context(self.ctx)
            output = self.model(data)
            [preds.append(self.label_map.get(int(pred.asscalar()))) for pred in nd.argmax(output, axis=1)]

        idx = 0
        for doc in docs:
            for sen in doc.sentences:
                sen[self.key] = preds[idx:idx+len(sen)]
                idx += len(sen)

    def evaluate_block(self, data_iter: DataLoader, docs: Sequence[Document]) -> float:
        """

        :param data_iter:
        :param docs:
        :return:
        """
        self.decode_block(data_iter=data_iter, docs=docs)
        if self.chunking:
            acc = ChunkF1()
            for doc in docs:
                for sen in doc.sentences:
                    acc.update(labels=sen[to_gold(self.key)], preds=sen[self.key])
        else:
            acc = Accuracy()
            for doc in docs:
                for sen in doc.sentences:
                    labels = nd.array([self.label_map.cid(label) for label in sen[to_gold(self.key)]])
                    preds = nd.array([self.label_map.cid(pred) for pred in sen[self.key]])
                    acc.update(labels=labels, preds=preds)
        return acc.get()[1]

    def data_loader(self, docs: Sequence[Document], batch_size, shuffle=False, label=True, transform=None, **kwargs) -> DataLoader:
        """

        :param docs:
        :param batch_size:
        :param shuffle:
        :param label:
        :param transform:
        :param kwargs:
        :return:
        """
        if label is True and self.label_map is None:
            raise ValueError('Please specify label_map')
        return DataLoader(TokensDataset(docs=docs, embs=self.embs, key=self.key, label_map=self.label_map, feature_windows=self.feature_windows, label=label, transform=transform),
                          batch_size=batch_size,
                          shuffle=shuffle)

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
        self.model = CNNModel(
            input_config=self.input_config,
            output_config=self.output_config,
            fuse_conv_config=self.fuse_conv_config,
            ngram_conv_config=self.ngram_conv_config,
            hidden_configs=self.hidden_configs)
        # self.model.load_params(params(model_path), self.ctx)
        self.model.load_parameters(params(model_path), self.ctx)
        logging.info('{} is loaded'.format(params(model_path)))
        logging.info(self.__str__())
        return self

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
        self.model.save_parameters(params(model_path))
        logging.info('{} is saved'.format(params(model_path)))
