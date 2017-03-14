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
from elit.components.template.state import NLPState
from typing import Dict, List, Tuple, Union
from abc import ABCMeta, abstractmethod
import numpy as np
import mxnet as mx

__author__ = 'Jinho D. Choi'


class NLPModel(metaclass=ABCMeta):
    def __init__(self):
        self.index_map: Dict[str, int] = {}
        self.labels: List[str] = []
        self.mxmod: mx.module.BaseModule = None

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

    @abstractmethod
    def train(self, train_data: mx.io.NDArrayIter, num_epoch: int = 1):
        for epoch in range(num_epoch):
            for data_batch in train_data:
                self.mxmod.forward_backward(data_batch)
                self.mxmod.update()

            # sync aux params across devices
            arg_params, aux_params = self.mxmod.get_params()
            self.mxmod.set_params(arg_params, aux_params)

            # end of 1 epoch, reset the data-iter for another epoch
            train_data.reset()

    @abstractmethod
    def predict(self, xs: np.array) -> np.array:
        batch_size = self.module.data_shapes[0][1][0]
        pad = batch_size - len(xs)
        if pad > 0: xs = np.vstack((xs, np.zeros((pad, len(xs[0])))))
        ys = self.module.predict(mx.io.NDArrayIter(xs, batch_size=batch_size))
        return ys[:len[xs]] if pad > 0 else ys

    def init_train(self, train_data: mx.io.NDArrayIter,
                    kvstore: Union[str, mx.kvstore.KVStore] = 'local',
                    optimizer: Union[str, mx.optimizer.Optimizer] = 'sgd',
                    optimizer_params=(('learning_rate', 0.01),),
                    initializer: mx.initializer.Initializer = mx.initializer.Uniform(0.01),
                    arg_params=None, aux_params=None,
                    allow_missing: bool = False, force_rebind: bool = False, force_init: bool = False):
        self.mxmod.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label, for_training=True,
                        force_rebind=force_rebind)
        self.mxmod.init_params(initializer=initializer, arg_params=arg_params, aux_params=aux_params,
                               allow_missing=allow_missing, force_init=force_init)
        self.mxmod.init_optimizer(kvstore=kvstore, optimizer=optimizer, optimizer_params=optimizer_params)

    def fit(self, module: mx.module.BaseModule, train_data: mx.io.NDArrayIter, eval_data: mx.io.NDArrayIter=None,
            kvstore: Union[str, mx.kvstore.KVStore] ='local',
            optimizer: Union[str, mx.optimizer.Optimizer]='sgd',
            optimizer_params=(('learning_rate', 0.01),),
            initializer: mx.initializer.Initializer=mx.initializer.Uniform(0.01),
            arg_params=None, aux_params=None,
            allow_missing: bool=False, force_rebind: bool=False, force_init: bool=False, num_epoch: int=1):
        module.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label, for_training=True,
                    force_rebind=force_rebind)
        module.init_params(initializer=initializer, arg_params=arg_params, aux_params=aux_params,
                           allow_missing=allow_missing, force_init=force_init)
        module.init_optimizer(kvstore=kvstore, optimizer=optimizer, optimizer_params=optimizer_params)

        ################################################################################
        # training loop
        ################################################################################
        for epoch in range(num_epoch):
            for data_batch in train_data:
                module.forward_backward(data_batch)
                module.update()

            # sync aux params across devices
            arg_params, aux_params = module.get_params()
            module.set_params(arg_params, aux_params)

            # end of 1 epoch, reset the data-iter for another epoch
            train_data.reset()

    # ============================== Predict ==============================

    @abstractmethod
    def create_feature_vector(self, state: NLPState) -> np.array:
        """
        :param state: the current processing state.
        :return: the feature vector representing the current state.
        """

    @abstractmethod
    def predict(self, x: np.array) -> int:
        """
        :param x: the feature vector.
        :return: the ID of the best predicted label.
        """

    def predict_label(self, x: np.array) -> str:
        """
        :param x: the feature vector.
        :return: the best predicted label.
        """
        return self.labels[self.predict(x)]


def FeedForwardNeuralNetwork(hidden_layers: List[Tuple[int, str]], output_layer: Tuple[int, str],
                             context: mx.context.Context) -> mx.mod.Module:
    net = mx.sym.Variable('data')

    for i, num_hidden, act_type in enumerate(hidden_layers, 1):
        net = mx.sym.FullyConnected(net, name='fc'+str(i), num_hidden=num_hidden)
        net = mx.sym.Activation(net, name=act_type+str(i), act_type=act_type)

    net = mx.sym.FullyConnected(net, name='fc'+str(len(hidden_layers)+1), num_hidden=output_layer[0])
    net = mx.sym.SoftmaxOutput(net, name='softmax')

    return mx.mod.Module(symbol=net, context=context)