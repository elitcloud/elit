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
from typing import Optional, Union, Sequence

import mxnet as mx
from mxnet import gluon, nd
from mxnet.ndarray import NDArray

__author__ = 'Jinho D. Choi'


class FFNNModel(gluon.Block):
    """
    :class:`FFNNModel` implements a Feed-Forward Neural Network (FFNN) consisting of
    an input layer, n-gram convolution layers (optional), hidden layers (optional), and an output layer.
    """

    def __init__(self,
                 input_config: SimpleNamespace,
                 output_config: SimpleNamespace,
                 conv2d_config: Optional[Sequence[SimpleNamespace]] = None,
                 hidden_config: Optional[Sequence[SimpleNamespace]] = None,
                 **kwargs):
        """
        :param input_config: configuration for the input layer, that is the return value of :meth:`namespace_input`;
                             {row: int, col: int, dropout: float}.
        :param output_config: configuration for the output layer, that is the return value of :meth:`namespace_output`;
                              {dim: int}.
        :param conv2d_config: configuration for the 2D convolution layer, that is the return value of :meth:`namespace_conv2d`;
                              {ngram: int, filters: int, activation: str, pool: str, dropout: float}.
        :param hidden_config: configuration for the hidden layers that is the return value of :meth:`namespace_hidden`;
                              {dim: int, activation: str, dropout: float}.
        """
        super().__init__(**kwargs)

        # initialize
        self.input = SimpleNamespace(dropout=mx.gluon.nn.Dropout(input_config.dropout))
        self.output = SimpleNamespace(dense=mx.gluon.nn.Dense(output_config.dim))

        self.conv2d = [SimpleNamespace(
            conv=mx.gluon.nn.Conv2D(channels=c.filters, kernel_size=(c.ngram, input_config.col), strides=(1, input_config.col), activation=c.activation),
            dropout=mx.gluon.nn.Dropout(c.dropout),
            pool=conv2d_pool(c.pool, input_config.row - c.ngram + 1)) for c in conv2d_config] if conv2d_config else None

        self.hidden = [SimpleNamespace(
            dense=mx.gluon.nn.Dense(units=h.dim, activation=h.activation),
            dropout=mx.gluon.nn.Dropout(h.dropout)) for h in hidden_config] if hidden_config else None

        # name scope
        with self.name_scope():
            setattr(self, 'input_dropout', self.input.dropout)

            if self.conv2d:
                for i, c in enumerate(self.conv2d, 1):
                    setattr(self, 'conv_' + str(i), c.conv)
                    setattr(self, 'conv_dropout_' + str(i), c.dropout)
                    if c.pool: setattr(self, 'conv_pool_' + str(i), c.pool)

            if self.hidden:
                for i, h in enumerate(self.hidden, 1):
                    setattr(self, 'hidden_' + str(i), h.dense)
                    setattr(self, 'hidden_dropout_' + str(i), h.dropout)

            setattr(self, 'output_0', self.output.dense)

    def forward(self, x: NDArray) -> NDArray:
        """
        :param x: the 3D input matrix whose dimensions represent (batch size, feature size, embedding size).
        :return: the output.
        """

        def conv(c: SimpleNamespace):
            return c.dropout(c.pool(c.conv(x))) if c.pool else c.dropout(c.conv(x).reshape((0, -1)))

        # input layer
        x = self.input.dropout(x)

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
        y = self.output.dense(x)
        return y


def namespace_input(col: int, row: int, dropout: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(col=col, row=row, dropout=dropout)


def namespace_output(dim: int) -> SimpleNamespace:
    return SimpleNamespace(dim=dim)


def namespace_conv2d(ngram: int, filters: int, activation: str, pool: str = None, dropout: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(ngram=ngram, filters=filters, activation=activation, pool=pool, dropout=dropout)


def namespace_hidden(dim: int, activation: str, dropout: float) -> SimpleNamespace:
    return SimpleNamespace(dim=dim, activation=activation, dropout=dropout)


def conv2d_pool(pool: str, n: int) -> Union[mx.gluon.nn.MaxPool2D, mx.gluon.nn.AvgPool2D, None]:
    if pool is None: return None
    p = mx.gluon.nn.MaxPool2D if pool == 'max' else mx.gluon.nn.AvgPool2D
    return p(pool_size=(n, 1), strides=(n, 1))
