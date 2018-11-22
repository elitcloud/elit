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
import abc
import datetime
import logging
import time
from typing import Sequence, List

import mxnet as mx
from mxnet.gluon import Trainer
from mxnet.gluon.data import DataLoader
from tqdm import trange

from elit.component import NLPComponent
from elit.nlp.embedding import Embedding
from elit.structure import Document
from elit.util.mx import mx_loss

__author__ = "Gary Lai"


class MXComponent(NLPComponent):
    """
    :class:`TokenTagger` provides an abstract class to implement a tagger that predicts a tag for every token.
    """

    def __init__(self, ctx: mx.Context, key: str, embs: List[Embedding], **kwargs):
        """
        :param ctx: the (list of) device context(s) for :class:`mxnet.gluon.Block`.
        :param vsms: the sequence of namespace(model, key),
                         where the key indicates the key of the values to retrieve embeddings for (e.g., tok).
        """
        self.ctx = ctx
        self.key = key
        self.embs = embs
        self.model = None
        self.loss = None
        self.trainer = None

    def train(self, trn_docs: Sequence[Document], dev_docs: Sequence[Document], model_path: str,
              epoch: int = 100,
              trn_batch: int = 64,
              dev_batch: int = 2048,
              loss=mx_loss('softmaxcrossentropyloss'),
              optimizer='adagrad',
              optimizer_params=None,
              **kwargs) -> float:
        log = ('Training configuration',
               '- max epoch: {}'.format(epoch),
               '- trn batch: {}'.format(trn_batch),
               '- dev batch: {}'.format(dev_batch),
               '- loss func: {}'.format(loss),
               '- optimizer: {} <- {}'.format(optimizer, optimizer_params))
        logging.info('\n'.join(log))

        self.model.initialize(ctx=self.ctx, force_reinit=True)
        self.loss = loss
        self.trainer = Trainer(self.model.collect_params(), optimizer=optimizer, optimizer_params=optimizer_params)

        trn_data = self.data_loader(docs=trn_docs, batch_size=trn_batch, shuffle=True, label=True, bucket=True)
        dev_data = self.data_loader(docs=dev_docs, batch_size=dev_batch, shuffle=False, label=True, bucket=True)

        logging.info('Training')
        best_e, best_eval = -1, -1
        epochs = trange(1, epoch + 1)

        for e in epochs:
            trn_st = time.time()
            trn_acc = 100 * self.train_block(data_iter=trn_data, docs=trn_docs)
            trn_et = time.time()

            dev_st = time.time()
            dev_acc = 100 * self.evaluate_block(data_iter=dev_data, docs=dev_docs)
            dev_et = time.time()

            if best_eval < dev_acc:
                best_e, best_eval = e, dev_acc
                self.save(model_path=model_path)

            desc = ("epoch: {}".format(e),
                    "trn time: {}".format(datetime.timedelta(seconds=(trn_et - trn_st))),
                    "trn acc: {:.2f}".format(trn_acc),
                    "dev time: {}".format(datetime.timedelta(seconds=(dev_et - dev_st))),
                    "dev acc: {:.2f}".format(dev_acc),
                    "best epoch: {}".format(best_e),
                    "best eval: {:.2f}".format(best_eval))
            epochs.set_description(desc=' '.join(desc))
        return best_eval

    @abc.abstractmethod
    def train_block(self, data_iter: DataLoader, docs: Sequence[Document]) -> float:
        raise NotImplementedError

    def decode(self, docs: Sequence[Document], batch_size: int = 2048, **kwargs) -> Sequence[Document]:
        data_iter = self.data_loader(docs=docs, batch_size=batch_size, shuffle=False, label=False, bucket=True, **kwargs)
        self.decode_block(data_iter=data_iter, docs=docs)
        return docs

    @abc.abstractmethod
    def decode_block(self, data_iter: DataLoader, docs: Sequence[Document], **kwargs):
        raise NotImplementedError

    def evaluate(self, docs: Sequence[Document], batch_size: int = 2048, **kwargs) -> tuple:
        st = time.time()
        data_iter = self.data_loader(docs=docs, batch_size=batch_size, shuffle=False, label=True, bucket=True, **kwargs)
        acc = self.evaluate_block(data_iter=data_iter, docs=docs)
        et = time.time()
        return acc, et - st

    @abc.abstractmethod
    def evaluate_block(self, data_iter: DataLoader, docs: Sequence[Document]):
        raise NotImplementedError

    @abc.abstractmethod
    def data_loader(self, docs: Sequence[Document], batch_size, shuffle, **kwargs):
        raise NotImplementedError
