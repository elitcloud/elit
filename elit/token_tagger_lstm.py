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
import datetime
import logging
import time
from types import SimpleNamespace
from typing import Sequence

import mxnet as mx
from gluonnlp.data import FixedBucketSampler
from mxnet import gluon, autograd, nd
from mxnet.gluon import Trainer, Block
from mxnet.gluon.data import DataLoader
from mxnet.gluon.utils import clip_global_norm
from mxnet.metric import Accuracy
from tqdm import trange, tqdm

from elit.cli import set_logger
from elit.component import MXNetComponent
from elit.data import batchify_fn, SentDataset
from elit.util.io import tsv_reader
from elit.util.structure import Document, TOK, POS
from elit.util.vsm import LabelMap, init_vsm

__author__ = "Gary Lai"


class RNNModel(Block):

    def __init__(self, rnn_config: SimpleNamespace, output_config: SimpleNamespace):
        super().__init__()
        self.rnn_layer = self._init_rnn_layer(rnn_config)
        self.output_layer = self._init_output_layer(output_config)

    def _init_rnn_layer(self, config: SimpleNamespace):

        if config.mode == 'lstm':
            mode = gluon.rnn.LSTM
        elif config.mode == 'gru':
            mode = gluon.rnn.GRU
        else:
            mode = gluon.rnn.RNN

        layer = SimpleNamespace(
            rnn=mode(
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                bidirectional=config.bidirectional,
                dropout=config.dropout,
                input_size=config.input_size,
            ),
            clip=0.2
        )

        with self.name_scope():
            self.__setattr__(layer.rnn.name, layer.rnn)

        return layer

    def _init_output_layer(self, config: SimpleNamespace) -> SimpleNamespace:
        layer = SimpleNamespace(
            dense=gluon.nn.Dense(
                units=config.num_class,
                flatten=config.flatten
            )
        )

        with self.name_scope():
            self.__setattr__(layer.dense.name, layer.dense)

        return layer

    def forward(self, X, state, *args):
        Y, state = self.rnn_layer.rnn(X, state)
        output = self.output_layer.dense(Y)
        return output, state

    def begin_state(self, batch_size, ctx, *args, **kwargs):
        # 1 * batch_size * hidden_size
        return self.rnn_layer.rnn.begin_state(batch_size=batch_size, ctx=ctx, *args, **kwargs)


class TokenTaggerLSTM(MXNetComponent):

    def __init__(self,
                 ctx: mx.Context,
                 vsm_path: list,
                 key: str,
                 label_map: LabelMap,
                 rnn_config: SimpleNamespace,
                 output_config: SimpleNamespace,
                 **kwargs):
        super().__init__(ctx, **kwargs)
        self.ctx = ctx
        self.key = key
        self.label_map = label_map
        self.vsms = [init_vsm(v) for v in vsm_path]
        rnn_config.input_size = sum([vsm.model.dim for vsm in self.vsms])
        self.model = RNNModel(rnn_config=rnn_config, output_config=output_config)

    def train(self, trn_docs: Sequence[Document], dev_docs: Sequence[Document], model_path: str,
              label_map: LabelMap = None,
              epoch=100,
              trn_batch=64,
              dev_batch=64,
              loss_fn=gluon.loss.SoftmaxCrossEntropyLoss(),
              optimizer='adagrad',
              optimizer_params=None,
              **kwargs):

        log = ('Configuration',
               '- context(s): {}'.format(self.ctx),
               '- trn_batch size: {}'.format(trn_batch),
               '- dev_batch size: {}'.format(dev_batch),
               '- max epoch : {}'.format(epoch),
               '- loss func : {}'.format(loss_fn),
               '- optimizer : {} <- {}'.format(optimizer, optimizer_params))

        logging.info('\n'.join(log))
        logging.info("Load trn data")
        trn_data = self.data_loader(docs=trn_docs, batch_size=trn_batch, bucket=True, shuffle=True)
        # logging.info("Load dev data")
        dev_data = self.data_loader(docs=dev_docs, batch_size=dev_batch, shuffle=False)

        self.model.initialize(ctx=self.ctx, force_reinit=True)
        trainer = Trainer(self.model.collect_params(),
                          optimizer=optimizer,
                          optimizer_params=optimizer_params)

        logging.info('Training')
        best_e, best_eval = -1, -1

        epochs = trange(1, epoch + 1)
        for e in epochs:
            trn_st = time.time()
            correct, total, = 0, 0
            for i, (data, label) in enumerate(tqdm(trn_data, leave=False)):
                # batch_size, sequence_length, input_size -> sequence_length, batch_size, input_size
                X = nd.transpose(data, axes=(1, 0, 2)).as_in_context(self.ctx)
                Y = label.T.as_in_context(self.ctx)
                state = self.model.begin_state(batch_size=X.shape[1], ctx=ctx)
                for s in state:
                    s.detach()
                with autograd.record():
                    output, state = self.model(X, state)
                    loss = loss_fn(output, Y)
                loss.backward()
                grads = [i.grad(self.ctx) for i in self.model.collect_params().values()]
                clip_global_norm(grads, self.model.rnn_layer.clip * X.shape[0] * X.shape[1])
                correct += len([1 for o, y in zip(nd.argmax(output, axis=2).reshape(-1, ), Y.reshape(-1, )) if int(o.asscalar()) == int(y.asscalar())])
                total += len(Y.reshape(-1, ))
                trainer.step(data.shape[0])
            trn_acc = 100.0 * correct / total
            trn_et = time.time()

            dev_st = time.time()
            dev_acc = 100.0 * self.accuracy(data_iterator=dev_data, docs=dev_docs)
            dev_et = time.time()

            if best_eval < dev_acc:
                best_e, best_eval = e, dev_acc

            desc = ("epoch: {}".format(e),
                    "trn time: {}".format(datetime.timedelta(seconds=(trn_et - trn_st))),
                    "train_acc: {}".format(trn_acc),
                    "dev time: {}".format(datetime.timedelta(seconds=(dev_et - dev_st))),
                    "dev_acc: {}".format(dev_acc),
                    "best epoch: {}".format(best_e),
                    "best eval: {}".format(best_eval))
            epochs.set_description(desc=' '.join(desc))

    def accuracy(self, data_iterator, docs=None):
        return self.token_accuracy(data_iterator)

    def token_accuracy(self, data_iterator):
        acc = Accuracy()
        for data, label in data_iterator:
            X = nd.transpose(data, axes=(1, 0, 2)).as_in_context(self.ctx)
            Y = label.T.as_in_context(self.ctx)
            state = self.model.begin_state(batch_size=X.shape[1], ctx=ctx)
            output, state = self.model(X, state)
            preds = nd.argmax(output, axis=2)
            acc.update(preds=preds, labels=Y)
        return acc.get()[1]

    def data_loader(self, docs,
                    batch_size,
                    shuffle=False,
                    batchify_fn=batchify_fn,
                    label=True,
                    bucket=False,
                    num_buckets=10,
                    ratio=0,
                    **kwargs):
        dataset = SentDataset(docs=docs, label_map=self.label_map, vsms=self.vsms, key=POS, display=True)
        if bucket:
            dataset_lengths = list(map(lambda x: float(len(x[1])), dataset))
            batch_sampler = FixedBucketSampler(dataset_lengths,
                                               batch_size=batch_size,
                                               num_buckets=num_buckets,
                                               ratio=ratio,
                                               shuffle=shuffle)
            return DataLoader(dataset=dataset,
                              batch_sampler=batch_sampler,
                              batchify_fn=batchify_fn)
        else:
            return DataLoader(dataset=dataset,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              batchify_fn=batchify_fn)


if __name__ == '__main__':
    set_logger()
    vsm_path = [['fasttext', '/home/glai2/Documents//vsm/dim400.bin']]
    trn_docs, lm = tsv_reader(tsv_directory='/home/glai2/Documents/wsj-pos/trn', cols={TOK: 0, POS: 1}, key=POS)
    dev_docs, _ = tsv_reader(tsv_directory='/home/glai2/Documents/wsj-pos/dev', cols={TOK: 0, POS: 1}, key=POS)
    ctx = mx.gpu()
    rnn_config = SimpleNamespace(
        mode='lstm',
        hidden_size=256,
        bidirectional=True,
        num_layers=1,
        dropout=0,
        clip=0.2
    )
    output_config = SimpleNamespace(
        num_class=len(lm),
        flatten=False,
    )

    comp = TokenTaggerLSTM(
        ctx=ctx,
        vsm_path=vsm_path,
        key=POS,
        label_map=lm,
        rnn_config=rnn_config,
        output_config=output_config
    )
    comp.train(trn_docs=trn_docs, dev_docs=dev_docs, model_path='pos-lstm', label_map=lm)
