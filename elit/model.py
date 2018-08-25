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
import abc
import inspect
from types import SimpleNamespace
from typing import Optional, Sequence, List

import mxnet as mx
from mxnet import gluon, nd
from mxnet.ndarray import NDArray

__author__ = 'Jinho D. Choi'


class NLPModel(gluon.Block):
    """
    :class:`NLPModel` is an abstract class providing helper methods to implement a neural network model.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abc.abstractmethod
    def forward(self, *args):
        raise NotImplementedError(
            '%s.%s()' %
            (self.__class__.__name__, inspect.stack()[0][3]))

    # ======================================== namespaces ====================

    @staticmethod
    def namespace_input_layer(
            row: int,
            col: int,
            dropout: float = 0.0) -> SimpleNamespace:
        """
        :param row: the row dimension of the input matrix.
        :param col: the column dimension of the input matrix.
        :param dropout: the dropout rate applied to the input matrix.
        :return: the namespace of (row, col, dropout) for the input layer.
        """
        return SimpleNamespace(col=col, row=row, dropout=dropout)

    @staticmethod
    def namespace_output_layer(dim: int) -> SimpleNamespace:
        """
        :param dim: the output dimension.
        :return: the namespace of (dim) for the output layer.
        """
        return SimpleNamespace(dim=dim)

    @staticmethod
    def namespace_fuse_conv_layer(
            filters: int,
            activation: str,
            dropout: float = 0.0) -> SimpleNamespace:
        """
        :param filters: the number of filters.
        :param activation: the activation function.
        :param dropout: the dropout applied to the output of this convolution.
        :return: the namespace of (filters, activation, dropout) for the fuse convolution layer.
        """
        return SimpleNamespace(
            filters=filters,
            activation=activation,
            dropout=dropout)

    @staticmethod
    def namespace_ngram_conv_layer(
            ngrams: Sequence[int],
            filters: int,
            activation: str,
            pool: str = None,
            dropout: float = 0.0) -> SimpleNamespace:
        """
        :param ngrams: the sequence of n-gram kernels applied to the convolutions.
        :param filters: the number of filters applied to each convolution.
        :param activation: the activation function applied to each convolution.
        :param dropout: the dropout rate applied to each convolution.
        :param pool: the pooling operation applied to each convolution (max|avg).
        :return: the namespace of (ngrams, filters, activation, pool, dropout) for the convolution layer.
        """
        return SimpleNamespace(
            ngrams=ngrams,
            filters=filters,
            activation=activation,
            pool=pool,
            dropout=dropout)

    @staticmethod
    def namespace_hidden_layer(
            dim: int,
            activation: str,
            dropout: float) -> SimpleNamespace:
        """
        :param dim: the dimension of the hidden layer.
        :param activation: the activation function applied to the hidden layer.
        :param dropout: the dropout rate applied to the output of the hidden layer.
        :return: the namespace of (dim, activation, dropout) for the hidden layer.
        """
        return SimpleNamespace(dim=dim, activation=activation, dropout=dropout)

    # ======================================== create_layers =================

    def init_input_layer(self, config: SimpleNamespace) -> SimpleNamespace:
        """
        :param config: the output of :meth:`FFNNModel.namespace_input_layer`.
        :return: the namespace of (dropout) for the input layer.
        """
        layer = SimpleNamespace(dropout=mx.gluon.nn.Dropout(config.dropout))

        with self.name_scope():
            self.__setattr__('input_dropout', layer.dropout)

        return layer

    def init_output_layer(self, config: SimpleNamespace) -> SimpleNamespace:
        """
        :param config: the output of :meth:`FFNNModel.namespace_output_layer`.
        :return: the namespace of (dense) for the output layer.
        """
        layer = SimpleNamespace(dense=mx.gluon.nn.Dense(config.dim))

        with self.name_scope():
            self.__setattr__('output', layer.dense)

        return layer

    def init_fuse_conv_layer(
            self,
            config: SimpleNamespace,
            input_col: int) -> SimpleNamespace:
        """
        :param config: the output of :meth:`FFNNModel.namespace_input_conv_layer`.
        :param input_col: the column dimension of the input matrix.
        :return: the namespace of (conv, dropout) for the fuse convolution layer.
        """
        layer = SimpleNamespace(
            conv=mx.gluon.nn.Conv2D(
                channels=config.filters, kernel_size=(
                    1, input_col), strides=(
                    1, input_col), activation=config.activation), dropout=mx.gluon.nn.Dropout(
                config.dropout))

        with self.name_scope():
            self.__setattr__('de_conv', layer.conv)
            self.__setattr__('de_dropout', layer.dropout)

        return layer

    def init_ngram_conv_layer(
            self,
            config: SimpleNamespace,
            input_row: int,
            input_col: int) -> List[SimpleNamespace]:
        """
        :param config: the output of :meth:`FFNNModel.namespace_ngram_conv_layer`.
        :param input_row: the row dimension of the input matrix.
        :param input_col: the column dimension of the input matrix.
        :return: the namespace of (conv, dropout, pool) for the n-gram convolution layer.
        """

        def pool(pool_type: str, n: int):
            if pool_type is None:
                return None
            p = mx.gluon.nn.MaxPool2D if pool_type == 'max' else mx.gluon.nn.AvgPool2D
            return p(pool_size=(n, 1), strides=(n, 1))

        def conv(n: int):
            layer = SimpleNamespace(
                conv=mx.gluon.nn.Conv2D(
                    channels=config.filters, kernel_size=(
                        n, input_col), strides=(
                        1, input_col), activation=config.activation), dropout=mx.gluon.nn.Dropout(
                    config.dropout), pool=pool(
                        config.pool, input_row - n + 1))

            with self.name_scope():
                self.__setattr__('ngram_conv_%d' % n, layer.conv)
                self.__setattr__('ngram_conv_%d_dropout' % n, layer.dropout)
                if layer.pool:
                    self.__setattr__('ngram_conv_%d_pool' % n, layer.pool)

            return layer

        return [conv(ngram) for ngram in config.ngrams]

    def init_hidden_layers(
            self,
            configs: Sequence[SimpleNamespace]) -> List[SimpleNamespace]:
        """
        :param configs: the sequence of outputs of :meth:`FFNNModel.namespace_hidden_layers`.
        :return: the namespace of (dense, dropout) for the hidden layers.
        """

        def hidden(n: int, c: SimpleNamespace) -> SimpleNamespace:
            layer = SimpleNamespace(
                dense=mx.gluon.nn.Dense(units=c.dim, activation=c.activation),
                dropout=mx.gluon.nn.Dropout(c.dropout))

            with self.name_scope():
                self.__setattr__('hidden_%d' % n, layer.dense)
                self.__setattr__('hidden_%d_dropout' % n, layer.dropout)

            return layer

        return [hidden(i, config) for i, config in enumerate(configs)]


class FFNNModel(NLPModel):
    """
    :class:`FFNNModel` implements a Feed-Forward Neural Network (FFNN) consisting of
    an input layer, n-gram convolution layers (optional), hidden layers (optional), and an output layer.
    This network accepts one input layer and returns one output layer.
    """

    def __init__(self,
                 input_config: SimpleNamespace,
                 output_config: SimpleNamespace,
                 fuse_conv_config: Optional[SimpleNamespace] = None,
                 ngram_conv_config: Optional[SimpleNamespace] = None,
                 hidden_configs: Optional[Sequence[SimpleNamespace]] = None,
                 **kwargs):
        """
        :param input_config: the configuration for the input layer; see :meth:`FFNNModel.namespace_input_layer`.
        :param output_config: the configuration for the output layer; see :meth:`FFNNModel.namespace_output_layer`.
        :param fuse_conv_config: the configuration for the fuse convolution layer; see :meth:`FFNNModel.namespace_fuse_conv_layer`.
        :param ngram_conv_config: the configuration for the n-gram convolution layer; see :meth:`FFNNModel.namespace_ngram_conv_layer`.
        :param hidden_configs: the configurations for the hidden layers; see :meth:`FFNNModel.namespace_hidden_layer`.
        """
        super().__init__(**kwargs)

        # initialize
        self._input = self.init_input_layer(input_config)

        if fuse_conv_config:
            self._fuse_conv = self.init_fuse_conv_layer(
                fuse_conv_config, input_config.col)
            col = fuse_conv_config.filters
        else:
            self._fuse_conv = None
            col = input_config.col

        self._ngram_convs = self.init_ngram_conv_layer(
            ngram_conv_config, input_config.row, col) if ngram_conv_config else None
        self._hiddens = self.init_hidden_layers(
            hidden_configs) if hidden_configs else None
        self._output = self.init_output_layer(output_config)

    def forward(self, x: NDArray) -> NDArray:
        # input layer
        x = self._input.dropout(x)

        # dimensionality reduction layer
        if self._fuse_conv:
            x = x.reshape((0, 1, x.shape[1], x.shape[2]))
            x = self._fuse_conv.dropout(self._fuse_conv.conv(x))

        # convolution layer
        if self._ngram_convs:
            x = mx.nd.transpose(
                x, (0, 3, 2, 1)) if self._fuse_conv else x.reshape(
                (0, 1, x.shape[1], x.shape[2]))
            t = [
                c.dropout(
                    c.pool(
                        c.conv(x))) if c.pool else c.dropout(
                    c.conv(x).reshape(
                        (0, -1))) for c in self._ngram_convs]
            x = nd.concat(*t, dim=1)

        # hidden layers
        if self._hiddens:
            for h in self._hiddens:
                x = h.dense(x)
                x = h.dropout(x)

        # output layer
        output = self._output.dense(x)
        return output
