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
import logging
import time
from abc import ABCMeta, abstractmethod
from random import shuffle
from typing import Dict, List, Tuple, Union

import mxnet as mx
import numpy as np

from elit.components.template.lexicon import NLPLexicon
from elit.components.template.state import NLPState
from elit.structure import NLPGraph

__author__ = 'Jinho D. Choi'


class NLPModel(metaclass=ABCMeta):
    def __init__(self, lexicon: NLPLexicon):
        self.lexicon: NLPLexicon = lexicon
        self.mxmod: mx.module.Module = None

        # label
        self.index_map: Dict[str, int] = {}
        self.labels: List[str] = []

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

    # ============================== State ==============================

    @abstractmethod
    def create_state(self, graph: NLPGraph, save_gold: bool = False) -> NLPState:
        """
        :param graph: the input graph.
        :param save_gold: if True, save gold tags from the input graph.
        :return: the wrapper state for the input graph.
        """

    # ============================== Feature ==============================

    @abstractmethod
    def x(self, state: NLPState) -> np.array:
        """ :return: the feature vector for the current state. """

    def feature_vectors(self, states: List[NLPState]) -> np.array:
        xs = [self.x(state) for state in states]
        return np.vstack(xs)

    def train_instances(self, states: List[NLPState]) -> Tuple[np.array, np.array]:
        xs, ys = zip(*[(self.x(state), state.gold_label) for state in states])
        return np.vstack(xs), np.array([self.add_label(y) for y in ys])

    # ============================== Module ==============================

    def bind(self, xs: np.array, ys: np.array=None, for_training: bool=True, batch_size=128) -> mx.io.DataIter:
        dat: mx.io.NDArrayIter = self.data_iter(xs, ys, batch_size)
        self.mxmod.bind(data_shapes=dat.provide_data, label_shapes=dat.provide_label, for_training=for_training,
                        force_rebind=True)
        return dat

    def fit(self, dat: mx.io.NDArrayIter, num_epoch: int=1):
        for epoch in range(num_epoch):
            for i, batch in enumerate(dat):
                self.mxmod.forward_backward(batch)
                self.mxmod.update()

            # sync aux params across devices
            arg_params, aux_params = self.mxmod.get_params()
            self.mxmod.set_params(arg_params, aux_params)

            # end of 1 epoch, reset the data-iter for another epoch
            dat.reset()

        return self.mxmod.predict(dat)

    def train(self, trn_graphs: List[NLPGraph], dev_graphs: List[NLPGraph], num_steps=20000, batch_size=128,
              initializer: mx.initializer.Initializer = mx.initializer.Uniform(0.01),
              arg_params=None, aux_params=None,
              allow_missing: bool=False, force_init: bool=False,
              kvstore: Union[str, mx.kvstore.KVStore] = 'local',
              optimizer: Union[str, mx.optimizer.Optimizer] = 'sgd',
              optimizer_params=(('learning_rate', 0.01),)):
        trn_states: List[NLPState] = [self.create_state(graph, save_gold=True) for graph in trn_graphs]
        dev_states: List[NLPState] = [self.create_state(graph, save_gold=True) for graph in dev_graphs]
        complete: int = 0

        for step in range(1, num_steps+1):
            st = time.time()
            shuffle(trn_states)
            xs, ys = self.train_instances(trn_states)
            dat: mx.io.NDArrayIter = self.bind(xs, ys, batch_size=batch_size)

            if step == 1:
                self.mxmod.init_params(
                    initializer=initializer, arg_params=arg_params, aux_params=aux_params,
                    allow_missing=allow_missing, force_init=force_init)
                self.mxmod.init_optimizer(
                    kvstore=kvstore, optimizer=optimizer, optimizer_params=optimizer_params)

            ys = self.fit(dat)
            # print([np.argmax(ys[i]) for i in range(10)])

            for i, state in enumerate(trn_states):
                label = self.get_label(np.argmax(ys[i]))
                state.process(label, ys[i])
                if state.terminate:
                    state.reset()
                    complete += 1

            et = time.time()
            logging.info('%4d: labels = %6d, complete = %d, time = %d' % (step, len(self.labels), complete, et-st))

            # if count > len(trn_graphs) / 2:
            #     n = self.evaluate(dev_states)
            #     s = 100.0 * n[1] / n[0]
            #     count = 0
            #     print(s)

    def evaluate(self, states: List[NLPState]):
        while states:
            xs, ys = self.train_instances(states)
            dat: mx.io.NDArrayIter = self.bind(xs)
            yhats = self.mxmod.predict(dat)

            for i, state in enumerate(states):
                state.process(self.get_label(np.argmax(ys[i])), ys[i])

            states = [state for state in states if not state.terminate]

    # ============================== Neural Networks ==============================

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

    # ============================== Helpers ==============================

    @classmethod
    def data_iter(cls, xs: np.array, ys: np.array=None, batch_size=128) -> mx.io.DataIter:
        batch_size = len(xs) if len(xs) < batch_size else batch_size
        return mx.io.NDArrayIter(data={'data': xs}, label={'softmax_label': ys}, batch_size=batch_size)
