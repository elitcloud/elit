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
from typing import Optional, Tuple, Union

import mxnet as mx
from mxnet import gluon, nd

__author__ = 'Jinho D. Choi'


class FFNNModel(gluon.Block):
    def __init__(self,
                 ctx: mx.Context,
                 initializer: mx.init.Initializer,
                 input_config: SimpleNamespace,
                 output_config: SimpleNamespace,
                 conv2d_config: Optional[Tuple[SimpleNamespace, ...]] = None,
                 hidden_config: Optional[Tuple[SimpleNamespace, ...]] = None,
                 **kwargs):
        """
        Feed-Forward Neural Network (FFNN) that includes n-gram convolutions or/and hidden layers.
        :param ctx: a device context.
        :param initializer: a weight initializer for the gluon block.
        :param input_config: configuration for the input layer -> elit.model.input_namespace();
                             {row: int, col: int, dropout: float}.
        :param output_config: configuration for the output layer -> elit.model.output_namespace();
                              {dim: int}.
        :param conv2d_config: configuration for the 2D convolution layer -> elit.model.conv2d_namespace();
                              {ngram: int, filters: int, activation: str, pool: str, dropout: float}.
        :param hidden_config: configuration for the hidden layers -> elit.model.hidden_namespace();
                              {dim: int, activation: str, dropout: float}.
        :param kwargs: extra parameters for the initialization of mxnet.gluon.Block.
        """
        super().__init__(**kwargs)
        self.ctx = ctx

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
            setattr(self, 'output_0', self.output.dense)

            if self.conv2d:
                for i, c in enumerate(self.conv2d, 1):
                    setattr(self, 'conv_' + str(i), c.conv)
                    setattr(self, 'conv_dropout_' + str(i), c.dropout)
                    if c.pool: setattr(self, 'conv_pool_' + str(i), c.pool)

            if self.hidden:
                for i, h in enumerate(self.hidden, 1):
                    setattr(self, 'hidden_' + str(i), h.dense)
                    setattr(self, 'hidden_dropout_' + str(i), h.dropout)

        self.collect_params().initialize(initializer, ctx=self.ctx)

    def forward(self, x):
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
        x = self.output.dense(x)
        return x


# ======================================== Configuration ========================================

def input_namespace(col: int, row: int, dropout: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(col=col, row=row, dropout=dropout)


def output_namespace(dim: int) -> SimpleNamespace:
    return SimpleNamespace(dim=dim)


def conv2d_namespace(ngram: int, filters: int, activation: str, pool: str = None, dropout: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(ngram=ngram, filters=filters, activation=activation, pool=pool, dropout=dropout)


def hidden_namespace(dim: int, activation: str, dropout: float) -> SimpleNamespace:
    return SimpleNamespace(dim=dim, activation=activation, dropout=dropout)


def conv2d_pool(pool: str, n: int) -> Union[mx.gluon.nn.MaxPool2D, mx.gluon.nn.AvgPool2D, None]:
    if pool is None: return None
    p = mx.gluon.nn.MaxPool2D if pool == 'max' else mx.gluon.nn.AvgPool2D
    return p(pool_size=(n, 1), strides=(n, 1))


# ======================================== Argument ========================================

def conv2d_args(s: str) -> Tuple[SimpleNamespace, ...]:
    """
    :param s: (ngram:filters:activation:pool:dropout)(;#1)*
    :return: a tuple of conf2d_namespace()
    """

    def create(config):
        c = config.split(':')
        pool = c[3] if c[3].lower() != 'none' else None
        return conv2d_namespace(ngram=int(c[0]), filters=int(c[1]), activation=c[2], pool=pool, dropout=float(c[4]))

    return tuple(create(config) for config in s.split(';')) if s.lower() != 'none' else None


def hidden_args(s: str) -> Tuple[SimpleNamespace, ...]:
    """
    :param s: (dim:activation:dropout)(;#1)*
    :return: a tuple of hidden_namespace()
    """

    def create(config):
        c = config.split(':')
        return SimpleNamespace(dim=int(c[0]), activation=c[1], dropout=float(c[2]))

    return tuple(create(config) for config in s.split(';')) if s.lower() != 'none' else None


def context_args(s: str) -> mx.Context:
    """
    :param s: [cg]\\d*
    :return: a device context
    """
    d = int(s[1:]) if len(s) > 1 else 0
    return mx.gpu(d) if s[0] == 'g' else mx.cpu(d)


def loss_args(s: str) -> mx.gluon.loss.Loss:
    s = s.lower()

    if s == 'softmaxcrossentropyloss':
        return mx.gluon.loss.SoftmaxCrossEntropyLoss()
    if s == 'sigmoidbinarycrossentropyloss':
        return mx.gluon.loss.SigmoidBinaryCrossEntropyLoss()
    if s == 'l2loss':
        return mx.gluon.loss.L2Loss()
    if s == 'l2loss':
        return mx.gluon.loss.L1Loss()
    if s == 'kldivloss':
        return mx.gluon.loss.KLDivLoss()
    if s == 'huberloss':
        return mx.gluon.loss.HuberLoss()
    if s == 'hingeloss':
        return mx.gluon.loss.HingeLoss()
    if s == 'squaredhingeloss':
        return mx.gluon.loss.SquaredHingeLoss()
    if s == 'logisticloss':
        return mx.gluon.loss.LogisticLoss()
    if s == 'tripletloss':
        return mx.gluon.loss.TripletLoss()
    if s == 'ctcloss':
        return mx.gluon.loss.CTCLoss()

    raise ValueError("Unsupported loss: " + s)
