#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2016 Timothy Dozat
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
import tensorflow as tf

from elit.dev.biaffineparser.common import leaky_relu, linear, tanh, gate


class LSTMCell(object):
    """
    A vanilla LSTM cell with three gates; the value of the output gate is computed from input from the lower layer
    and the previous hidden state, with no input from the current cell state. This is to allow all activations--the
    recurrent activation and the gate activations--to be computable with one matrix multiplication, making the
    network more efficient. Conceptually, using the cell state is probably undesireable anyway since the scale of the
    cell state can get more extreme over time, meaning that in a network that did use cell states to compute the
    output gate, the value of the output gate would be primarily determined by the input/previous hidden state in the
    first few steps but later on would be dominated by extreme values in the cell state.
    """

    def __init__(self, input_size, output_size) -> None:
        """
        Create a cell
        :param input_size:
        :param output_size:
        """
        super().__init__()
        self.output_size = output_size
        self.input_size = input_size if input_size is not None else self.output_size
        self.forget_bias = 0
        self.recur_func = leaky_relu

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            cell_tm1, hidden_tm1 = tf.split(state, 2, axis=1)
            input_list = [inputs, hidden_tm1]
            lin = linear(input_list,
                         self.output_size,
                         add_bias=True,
                         n_splits=4)
            cell_act, input_act, forget_act, output_act = lin

            cell_tilde_t = tanh(cell_act)
            input_gate = gate(input_act)
            forget_gate = gate(forget_act - self.forget_bias)
            output_gate = gate(output_act)
            cell_t = input_gate * cell_tilde_t + (1 - forget_gate) * cell_tm1
            hidden_tilde_t = self.recur_func(cell_t)
            hidden_t = hidden_tilde_t * output_gate

            return hidden_t, tf.concat([cell_t, hidden_t], 1)

    def zero_state(self, batch_size, dtype):
        """
        Initial states
        :param batch_size:
        :param dtype:
        :return:
        """
        zero_state = tf.get_variable('Zero_state',
                                     shape=self.state_size,
                                     dtype=dtype,
                                     initializer=tf.zeros_initializer())
        state = tf.reshape(tf.tile(zero_state, tf.stack([batch_size])), tf.stack([batch_size, self.state_size]))
        state.set_shape([None, self.state_size])
        return state

    @property
    def state_size(self):
        return self.output_size * 2
