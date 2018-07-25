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
from types import SimpleNamespace

import numpy as np
import mxnet as mx
from mxnet import gluon, nd, autograd

__author__ = 'Jinho D. Choi'


class CNNModel(gluon.Block):
    def __init__(self, input_config, output_config, conv2d_config=None, hidden_config=None, **kwargs):
        super().__init__(**kwargs)

        def pool(c):
            if c.pool is None: return lambda x: x
            p = mx.gluon.nn.MaxPool2D if c.pool == 'max' else mx.gluon.nn.AvgPool2D
            n = input_config.row - c.ngram + 1
            return p(pool_size=(n, 1), strides=(n, 1))

        self.conv2d = [SimpleNamespace(
            conv=mx.gluon.nn.Conv2D(channels=c.filters, kernel_size=(c.ngram, input_config.col), strides=(1, input_config.col), activation=c.activation),
            dropout=mx.gluon.nn.Dropout(c.dropout),
            pool=pool(c)) for c in conv2d_config] if conv2d_config else None

        self.hidden = [SimpleNamespace(
            dense=mx.gluon.nn.Dense(units=h.dim, activation=h.activation),
            dropout=mx.gluon.nn.Dropout(h.dropout)) for h in hidden_config] if hidden_config else None

        with self.name_scope():
            self.input_dropout = mx.gluon.nn.Dropout(input_config.dropout)
            self.output = mx.gluon.nn.Dense(output_config.dim)

            if self.conv2d:
                for i, c in enumerate(self.conv2d, 1):
                    setattr(self, 'conv_'+str(i), c.conv)
                    setattr(self, 'conv_pool_'+str(i), c.pool)
                    setattr(self, 'conv_dropout_' + str(i), c.dropout)

            if self.hidden:
                for i, h in enumerate(self.hidden, 1):
                    setattr(self, 'hidden_' + str(i), h.dense)
                    setattr(self, 'hidden_dropout_' + str(i), h.dropout)

    def forward(self, x):
        # input layer
        x = self.input_dropout(x)

        # convolution layer
        if self.conv2d:
            # (batches, input.row, input.col) -> (batches, 1, input.row, input.col)
            x = x.reshape((0, 1, x.shape[1], x.shape[2]))

            # conv: [(batches, filters, maxlen - ngram + 1, 1) for ngram in ngrams]
            # pool: [(batches, filters, 1, 1) for ngram in ngrams]
            # reshape: [(batches, filters * x * y) for ngram in ngrams]
            t = [c.dropout(c.pool(c.conv(x))) for c in self.conv2d]
            x = nd.concat(*t, dim=1)

        if self.hidden:
            for h in self.hidden:
                x = h.dense(x)
                x = h.dropout(x)

        # output layer
        x = self.output(x)
        return x


def demo():
    ngrams = range(2, 5)
    train_size = 10
    ctx = mx.cpu(0)
    batches = 3

    input_config = SimpleNamespace(row=5, col=20, dropout=0.0)
    output_config = SimpleNamespace(dim=3)
    conv2d_config = [SimpleNamespace(ngram=i, filters=4, activation='relu', pool='max', dropout=0.0) for i in ngrams]
    hidden_config = [SimpleNamespace(dim=10, activation='relu', dropout=0.0)]

    xs = nd.array([np.random.rand(input_config.row, input_config.col) for i in range(train_size)])
    ys = nd.array(np.random.randint(output_config.dim, size=train_size))

    model = CNNModel(input_config, output_config, conv2d_config, hidden_config)
    model.collect_params().initialize(mx.init.Xavier(rnd_type='gaussian', magnitude=2.24), ctx=ctx)

    batches = gluon.data.DataLoader(gluon.data.ArrayDataset(xs, ys), batch_size=batches)
    trainer = gluon.Trainer(model.collect_params(), 'sgd', {'learning_rate': 0.01, 'wd': 0.0})
    loss_func = mx.gluon.loss.SoftmaxCrossEntropyLoss()

    for x, y in batches:
        x = x.as_in_context(ctx)
        y = y.as_in_context(ctx)

        with autograd.record():
            output = model(x)
            loss = loss_func(output, y)
            loss.backward()

        trainer.step(x.shape[0])


if __name__ == '__main__':
    demo()
