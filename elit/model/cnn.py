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
from typing import Sequence, Optional, List

from mxnet import gluon
from types import SimpleNamespace

__author__ = "Gary Lai"


class CNNModel(gluon.HybridBlock):
    """
    :class:`NLPModel` is an abstract class providing helper methods to implement a neural network model.
    """

    def __init__(self, input_config: SimpleNamespace, output_config: SimpleNamespace,
                 fuse_conv_config: SimpleNamespace=None, ngram_conv_config: SimpleNamespace=None, hidden_configs: SimpleNamespace=None):
        super().__init__()
        self.input_layer = self._init_input_layer(input_config)
        self.output_layer = self._init_output_layer(output_config)
        self.fuse_conv_layer = self._init_fuse_conv_layer(fuse_conv_config)
        self.ngram_conv_layers = self._init_ngram_conv_layers(ngram_conv_config)
        self.hidden_layers = self._init_hidden_layers(hidden_configs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        # x: batch_size, window size, features

        # input layer
        x = self.input_layer.dropout(x)

        # dimensionality reduction layer
        if self.fuse_conv_layer is not None:
            x = F.reshape(x, (0, 1, self.input_layer.row, self.input_layer.col))
            x = self.fuse_conv_layer.dropout(self.fuse_conv_layer.conv(x))

        # convolution layer
        if self.ngram_conv_layers is not None:
            if self.fuse_conv_layer is not None:
                x = F.transpose(x, (0, 3, 2, 1))
            else:
                x = F.reshape(x, (0, 1, self.input_layer.row, self.input_layer.col))
            c = [layer.dropout(layer.pool(layer.conv(x))) if layer.pool else layer.dropout(layer.conv(x).reshape((0, -1))) for layer in self.ngram_conv_layers]
            x = F.concat(*c, dim=1)

        # hidden layers
        if self.hidden_layers is not None:
            for layer in self.hidden_layers:
                x = layer.dense(x)
                x = layer.dropout(x)

        # output layer
        x = self.output_layer.dense(x)
        return x

    # ======================================== create_layers ========================================

    def _init_input_layer(self, config: SimpleNamespace) -> SimpleNamespace:
        """
        :param config: the output of :meth:`FFNNModel.namespace_input_layer`.
        :return: the namespace of (dropout) for the input layer.
        """
        layer = SimpleNamespace(
            col=config.col,
            row=config.row,
            dropout=gluon.nn.Dropout(config.dropout)
        )

        with self.name_scope():
            self.__setattr__(layer.dropout.name, layer.dropout)

        return layer

    def _init_output_layer(self, config: SimpleNamespace) -> SimpleNamespace:
        """
        :param config: the output of :meth:`FFNNModel.namespace_output_layer`.
        :return: the namespace of (dense) for the output layer.
        """
        layer = SimpleNamespace(
            dense=gluon.nn.Dense(
                units=config.num_class,
                flatten=config.flatten
            )
        )

        with self.name_scope():
            self.__setattr__(layer.dense.name, layer.dense)

        return layer

    def _init_fuse_conv_layer(self, config: SimpleNamespace) -> Optional[SimpleNamespace]:
        """
        :param config: the output of :meth:`FFNNModel.namespace_input_conv_layer`.
        :param input_col: the column dimension of the input matrix.
        :return: the namespace of (conv, dropout) for the fuse convolution layer.
        """
        if config is None:
            return None
        layer = SimpleNamespace(
            conv=gluon.nn.Conv2D(
                channels=config.filters,
                kernel_size=(1, self.input_layer.col),
                strides=(1, self.input_layer.col),
                activation=config.activation),
            dropout=gluon.nn.Dropout(config.dropout))

        with self.name_scope():
            self.__setattr__(layer.conv.name, layer.conv)
            self.__setattr__(layer.dropout.name, layer.dropout)

        return layer

    def _init_ngram_conv_layers(self, config: SimpleNamespace) -> Optional[List[SimpleNamespace]]:

        if config is None:
            return None

        def conv(ngram):
            return gluon.nn.Conv2D(
                channels=config.filters,
                kernel_size=(ngram, self.input_layer.col if self.fuse_conv_layer is None else self.fuse_conv_layer._channels),
                strides=(1, self.input_layer.col),
                activation=config.activation)

        def pool(ngram):
            pool_model = gluon.nn.MaxPool2D if config.pool == 'max' else gluon.nn.AvgPool2D
            return pool_model(pool_size=(ngram, 1), strides=(ngram, 1))

        def layers(ngram):
            layer = SimpleNamespace(
                conv=conv(ngram),
                dropout=gluon.nn.Dropout(config.dropout),
                pool=pool(ngram) if config.pool is not None else None
            )

            with self.name_scope():
                self.__setattr__(layer.conv.name, layer.conv)
                self.__setattr__(layer.dropout.name, layer.dropout)
                if config.pool is not None:
                    self.__setattr__(layer.pool.name, layer.pool)

            return layer

        return [layers(ngram) for ngram in config.ngrams]

    def _init_hidden_layers(self, configs: Optional[Sequence[SimpleNamespace]]) -> Optional[List[SimpleNamespace]]:
        """
        :param configs: the sequence of outputs of :meth:`FFNNModel.namespace_hidden_layers`.
        :return: the namespace of (dense, dropout) for the hidden layers.
        """

        if configs is None:
            return None

        def hidden(c: SimpleNamespace) -> SimpleNamespace:
            layer = SimpleNamespace(
                dense=gluon.nn.Dense(units=c.dim,
                                     activation=c.activation),
                dropout=gluon.nn.Dropout(c.dropout))

            with self.name_scope():
                self.__setattr__('hidden_{}'.format(layer.dense.name), layer.dense)
                self.__setattr__('hidden_{}'.format(layer.dropout.name), layer.dropout)

            return layer

        return [hidden(config) for config in configs]