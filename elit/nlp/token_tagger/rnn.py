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
from typing import Sequence, List

import mxnet as mx
from gluonnlp.data import FixedBucketSampler
from mxnet import nd, autograd
from mxnet.gluon.data import DataLoader
from mxnet.gluon.utils import clip_global_norm
from mxnet.metric import Accuracy
from tqdm import tqdm

from elit.component import RNNComponent
from elit.dataset import LabelMap, sequence_batchify_fn, SequencesDataset
from elit.nlp.embedding import Embedding
from elit.eval import ChunkF1
from elit.model import RNNModel
from elit.structure import Document, to_gold
from elit.util.io import pkl, params

__author__ = 'Gary Lai'


class RNNTokenTagger(RNNComponent):

    def __init__(self, ctx: mx.Context, key: str, embs: List[Embedding],
                 rnn_config: SimpleNamespace = None, output_config: SimpleNamespace = None,
                 label_map: LabelMap = None, chunking: bool = False,
                 **kwargs):
        """

        :param ctx: ctx is the Context
        :param key:
        :param embs_config:
        :param label_map:
        :param chunking:
        :param rnn_config:
        :param output_config:
        :param kwargs:
        """
        self.label_map = label_map
        self.chunking = chunking
        if output_config is not None:
            output_config.num_class = self.label_map.num_class() if label_map else 1
        super().__init__(ctx, key, embs, rnn_config, output_config, **kwargs)

    def __str__(self):
        s = ('RNNTokenTagger',
             '- context: {}'.format(self.ctx),
             '- key: {}'.format(self.key),
             '- embs: {}'.format(self.embs),
             '- label_map: {}'.format(self.label_map),
             '- chunking: {}'.format(self.chunking),
             '- model: {}'.format(self.model))
        return '\n'.join(s)

    def train_block(self, data_iter: DataLoader, docs: Sequence[Document]) -> float:
        acc = Accuracy()
        for dids, sids, data, label in tqdm(data_iter, leave=False):
            # batch_size, sequence_length, input_size -> sequence_length, batch_size, input_size
            X = nd.transpose(data, axes=(1, 0, 2)).as_in_context(self.ctx)
            # batch_size, sequence_length -> sequence_length, batch_size
            Y = label.T.as_in_context(self.ctx)
            state = self.model.begin_state(batch_size=X.shape[1], ctx=self.ctx)
            for s in state:
                s.detach()
            with autograd.record():
                output, state = self.model(X, state)
                l = self.loss(output, Y)
            l.backward()
            grads = [param.grad(self.ctx) for param in self.model.collect_params().values()]
            clip_global_norm(grads, self.model.rnn_layer.clip * X.shape[0] * X.shape[1])

            # sequence_length, batch_size -> batch_size, sequence_length
            for batch, (preds, labels) in enumerate(zip(nd.argmax(output, axis=2).T, label)):
                sen = docs[dids[batch].asscalar()].sentences[sids[batch].asscalar()]
                sequence_length = len(sen)
                preds = preds[:sequence_length]
                labels = labels[:sequence_length]
                acc.update(labels=labels, preds=preds)
            self.trainer.step(data.shape[0])
        return float(acc.get()[1])

    def decode_block(self, data_iter: DataLoader, docs: Sequence[Document], **kwargs):
        for dids, sids, data, label in tqdm(data_iter, leave=False):
            X = nd.transpose(data, axes=(1, 0, 2)).as_in_context(self.ctx)
            state = self.model.begin_state(batch_size=X.shape[1], ctx=self.ctx)
            output, state = self.model(X, state)
            for batch, preds in enumerate(nd.argmax(output, axis=2).T):
                sen = docs[dids[batch].asscalar()].sentences[sids[batch].asscalar()]
                sequence_length = len(sen)
                preds = [self.label_map.get(int(pred.asscalar())) for pred in preds[:sequence_length]]
                sen[self.key] = preds

    def evaluate_block(self, data_iter: DataLoader, docs: Sequence[Document]) -> float:
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

    def data_loader(self, docs: Sequence[Document], batch_size, shuffle=False, label=True, **kwargs) -> DataLoader:
        if label is True and self.label_map is None:
            raise ValueError('Please specify label_map')
        batchify_fn = kwargs.get('batchify_fn', sequence_batchify_fn)
        bucket = kwargs.get('bucket', False)
        num_buckets = kwargs.get('num_buckets', 10)
        ratio = kwargs.get('ratio', 0)
        dataset = SequencesDataset(docs=docs, embs=self.embs, key=self.key, label_map=self.label_map, label=label)
        if bucket is True:
            dataset_lengths = list(map(lambda x: float(len(x[3])), dataset))
            batch_sampler = FixedBucketSampler(dataset_lengths, batch_size=batch_size, num_buckets=num_buckets, ratio=ratio, shuffle=shuffle)
            return DataLoader(dataset=dataset, batch_sampler=batch_sampler, batchify_fn=batchify_fn)
        else:
            return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, batchify_fn=batchify_fn)

    def load(self, model_path: str, **kwargs):
        """
        :param model_path: the path to a pre-trained model to be loaded.
        :type model_path: str
        """
        with open(pkl(model_path), 'rb') as fin:
            self.key = pickle.load(fin)
            self.label_map = pickle.load(fin)
            self.chunking = pickle.load(fin)
            self.rnn_config = pickle.load(fin)
            self.output_config = pickle.load(fin)
        logging.info('{} is loaded'.format(pkl(model_path)))
        self.model = RNNModel(rnn_config=self.rnn_config, output_config=self.output_config)
        self.model.load_parameters(params(model_path), self.ctx)
        # self.model.load_params(params(model_path), self.ctx)
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
            pickle.dump(self.rnn_config, fout)
            pickle.dump(self.output_config, fout)
        logging.info('{} is saved'.format(pkl(model_path)))
        self.model.save_parameters(params(model_path))
        logging.info('{} is saved'.format(params(model_path)))
