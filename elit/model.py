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

import mxnet as mx
from mxnet import gluon, nd

__author__ = 'Jinho D. Choi'


class FFNNModel(gluon.Block):
    def __init__(self, input_config, output_config, conv2d_config=None, hidden_config=None, **kwargs):
        """
        Feed-Forward Neural Network that includes n-gram convolution or/and hidden layers.
        :param input_config: (row, col, dropout); configuration for the input layer.
        :type input_config: SimpleNamespace(int, float)
        :param output_config: (dim); configuration for the output layer.
        :type output_config: SimpleNamespace(int)
        :param conv2d_config: (ngram, filters, activation, dropout); configuration for the 2D convolution layer.
        :type conv2d_config: list of SimpleNamespace(int, int, str, float)
        :param hidden_config: (dim, activation, dropout); configuration for the hidden layers.
        :type hidden_config: list of SimpleNamespace(int, str, float)
        :param kwargs: parameters for the initialization of mxnet.gluon.Block.
        :type kwargs: dict
        """
        super().__init__(**kwargs)

        def pool(c):
            if c.pool is None: return None
            p = mx.gluon.nn.MaxPool2D if c.pool == 'max' else mx.gluon.nn.AvgPool2D
            n = input_config.maxlen - c.ngram + 1
            return p(pool_size=(n, 1), strides=(n, 1))

        self.conv2d = [SimpleNamespace(
            conv=mx.gluon.nn.Conv2D(channels=c.filters, kernel_size=(c.ngram, input_config.dim), strides=(1, input_config.dim), activation=c.activation),
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
                    setattr(self, 'conv_dropout_' + str(i), c.dropout)
                    if c.pool: setattr(self, 'conv_pool_'+str(i), c.pool)

            if self.hidden:
                for i, h in enumerate(self.hidden, 1):
                    setattr(self, 'hidden_' + str(i), h.dense)
                    setattr(self, 'hidden_dropout_' + str(i), h.dropout)

    def forward(self, x):
        def conv(c):
            return c.dropout(c.pool(c.conv(x))) if c.pool else c.dropout(c.conv(x).reshape((0, -1)))

        # input layer
        x = self.input_dropout(x)

        # convolution layer
        if self.conv2d:
            # (batches, input.row, input.col) -> (batches, 1, input.row, input.col)
            x = x.reshape((0, 1, x.shape[1], x.shape[2]))

            # conv: [(batches, filters, maxlen - ngram + 1, 1) for ngram in ngrams]
            # pool: [(batches, filters, 1, 1) for ngram in ngrams]
            # reshape: [(batches, filters * x * y) for ngram in ngrams]
            t = [conv(c) for c in self.conv2d]
            x = nd.concat(*t, dim=1)

        if self.hidden:
            for h in self.hidden:
                x = h.dense(x)
                x = h.dropout(x)

        # output layer
        x = self.output(x)
        return x


def input_namespace(dim, maxlen, dropout=0.0):
    return SimpleNamespace(dim=dim, maxlen=maxlen, dropout=dropout)


def output_namespace(dim):
    return SimpleNamespace(dim=dim)


def conv2d_namespace(ngram, filters, activation, pool=None, dropout=0.0):
    return SimpleNamespace(ngram=ngram, filters=filters, activation=activation, pool=pool, dropout=dropout)


def conv2d_args(s):
    """
    :param s: (ngram:filters:activation:pool:dropout)(;#1)*
    :return: list of conv2d_namespace
    """
    def create(config):
        c = config.split(':')
        pool = c[3] if c[3].lower() != 'none' else None
        return conv2d_namespace(ngram=int(c[0]), filters=int(c[1]), activation=c[2], pool=pool, dropout=float(c[4]))

    return tuple(create(config) for config in s.split(';')) if s.lower() != 'none' else None


def hidden_namespace(dim, activation, dropout):
    return SimpleNamespace(dim=dim, activation=activation, dropout=dropout)


def hidden_args(s):
    """
    :param s: (dim:activation:dropout)(;#1)*
    :return: list of SimpleNamespace
    """
    def create(config):
        c = config.split(':')
        return SimpleNamespace(dim=int(c[0]), activation=c[1], dropout=float(c[2]))

    return tuple(create(config) for config in s.split(';')) if s.lower() != 'none' else None
