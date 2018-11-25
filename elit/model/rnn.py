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
from mxnet import gluon
from mxnet.gluon import Block
from types import SimpleNamespace

__author__ = "Gary Lai"


class RNNModel(Block):

    def __init__(self, rnn_config: SimpleNamespace, output_config: SimpleNamespace):
        super().__init__()
        self.rnn_layer = self._init_rnn_layer(rnn_config)
        self.output_layer = self._init_output_layer(output_config)

    def forward(self, X, state, *args):
        Y, state = self.rnn_layer.rnn(X, state)
        output = self.output_layer.dense(Y)
        return output, state

    def begin_state(self, batch_size, ctx, *args, **kwargs):
        # 1 * batch_size * hidden_size
        return self.rnn_layer.rnn.begin_state(batch_size=batch_size, ctx=ctx, *args, **kwargs)

    def detach(self, hidden):
        if isinstance(hidden, (tuple, list)):
            hidden = [self.detach(h) for h in hidden]
        else:
            hidden = hidden.detach()
        return hidden

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
                layout=config.layout,
                dropout=config.dropout,
                bidirectional=config.bidirectional,
                i2h_weight_initializer=config.i2h_weight_initializer,
                h2h_weight_initializer=config.h2h_weight_initializer,
                i2h_bias_initializer=config.i2h_bias_initializer,
                h2h_bias_initializer=config.h2h_bias_initializer,
                input_size=config.input_size,
            ),
            clip=config.clip
        )

        with self.name_scope():
            self.__setattr__(layer.rnn.name, layer.rnn)

        return layer

    def _init_output_layer(self, config: SimpleNamespace) -> SimpleNamespace:
        """
        :param config: the output of :meth:`RNNModel.output_layer`.
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

