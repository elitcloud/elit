# ========================================================================
# Copyright 2017 Emory University
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
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Tuple, Union

import mxnet as mx
import numpy as np

from elit.components.template.state import NLPState

__author__ = 'Jinho D. Choi'


class NLPModel(metaclass=ABCMeta):
    def __init__(self, batch_size: int=128):
        # label
        self.index_map: Dict[str, int] = {}
        self.labels: List[str] = []

        # module
        self.batch_size = batch_size
        self.mxmod: mx.module.Module = None

    # ============================== Label ==============================

    def get_label_index(self, label: str) -> int:
        """
        :return: the index of the label.
        """
        return self.index_map.get(label, -1)

    def get_label(self, index: Union[int, np.int]) -> str:
        """
        :param index: the index of the label to be returned.
        :return: the index'th label.
        """
        return self.labels[index]

    def add_label(self, label: str) -> int:
        """
        :return: the index of the label.
          Add a label to this map if not exist already.
        """
        idx = self.get_label_index(label)
        if idx < 0:
            idx = len(self.labels)
            self.index_map[label] = idx
            self.labels.append(label)
        return idx

    # ============================== Module ==============================

    def data_iter(self, xs: np.array, ys: np.array=None) -> mx.io.DataIter:
        batch_size = len(xs) if len(xs) < self.batch_size else self.batch_size
        return mx.io.NDArrayIter(data={'data': xs}, label={'softmax_label': ys}, batch_size=batch_size)

    def bind(self, xs: np.array, ys: np.array=None, for_training: bool=True) -> mx.io.DataIter:
        dat: mx.io.NDArrayIter = self.data_iter(xs, ys)
        self.mxmod.bind(data_shapes=dat.provide_data, label_shapes=dat.provide_label, for_training=for_training,
                        force_rebind=True)
        return dat

    def train(self, dat: mx.io.NDArrayIter, num_epoch: int=1):
        for epoch in range(num_epoch):
            for i, batch in enumerate(dat):
                self.mxmod.forward_backward(batch)
                self.mxmod.update()

            # sync aux params across devices
            arg_params, aux_params = self.mxmod.get_params()
            self.mxmod.set_params(arg_params, aux_params)

            # end of 1 epoch, reset the data-iter for another epoch
            dat.reset()

    def predict(self, dat: mx.io.DataIter) -> np.array:
        return self.mxmod.predict(dat)

    # def predict(self, xs: np.array) -> np.array:
    #     batch_size = self.module.data_shapes[0][1][0]
    #     pad = batch_size - len(xs)
    #     if pad > 0: xs = np.vstack((xs, np.zeros((pad, len(xs[0])))))
    #     ys = self.module.predict(mx.io.NDArrayIter(xs, batch_size=batch_size))
    #     return ys[:len[xs]] if pad > 0 else ys

    # ============================== Neural Networks ==============================

    @abstractmethod
    def x(self, state: NLPState) -> np.array:
        """ :return: the feature vector for the current state. """

    @classmethod
    def create_ffnn(cls, hidden: List[Tuple[int, str, float]], input_dropout: float=0, output_size: int=2,
                    context: mx.context.Context = mx.cpu()) -> mx.mod.Module:
        net = mx.sym.Variable('data')
        if input_dropout > 0: net = mx.sym.Dropout(net, p=input_dropout)

        for i, (num_hidden, act_type, dropout) in enumerate(hidden, 1):
            net = mx.sym.FullyConnected(net, num_hidden=num_hidden, name='fc' + str(i))
            if act_type: net = mx.sym.Activation(net, act_type=act_type, name=act_type + str(i))
            if dropout > 0: net = mx.sym.Dropout(net, p=dropout)

        net = mx.sym.FullyConnected(net, num_hidden=output_size, name='fc' + str(len(hidden) + 1))
        net = mx.sym.SoftmaxOutput(net, name='softmax')

        return mx.mod.Module(symbol=net, context=context)


