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
from typing import Dict, List, Tuple, Union, Callable

import mxnet as mx
import numpy as np

from elit.components.template.lexicon import NLPLexicon
from elit.components.template.state import NLPState
from elit.structure import NLPGraph

__author__ = 'Jinho D. Choi'


class NLPModel(metaclass=ABCMeta):
    def __init__(self, mxmod: mx.module.Module, create_state: Callable[[NLPGraph, NLPLexicon, bool], NLPState]):
        # module
        self.mxmod: mx.module.Module = mxmod
        self.create_state = create_state

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

    @property
    def label_size(self):
        return len(self.labels)

    # ============================== Feature ==============================

    @abstractmethod
    def x(self, state: NLPState) -> np.array:
        """ :return: the feature vector for the current state. """

    def feature_vectors(self, states: List[NLPState]) -> np.array:
        xs = [self.x(state) for state in states]
        return np.vstack(xs)

    def train_instances(self, states: List[NLPState]) -> Tuple[np.array, np.array]:
        xs, ys = zip(*[(self.x(state), state.gold) for state in states])
        return np.vstack(xs), np.array([self.add_label(y) for y in ys])

    # ============================== Module ==============================

    def bind(self, data: np.array, label: np.array=None, batch_size=32,
             for_training: bool=True, force_rebind=True) -> mx.io.DataIter:
        batches: mx.io.NDArrayIter = self.data_iter(data, label, batch_size)
        label_shapes = None if label is None else batches.provide_label
        self.mxmod.bind(data_shapes=batches.provide_data, label_shapes=label_shapes, for_training=for_training,
                        force_rebind=force_rebind)
        return batches

    def fit(self, batches: mx.io.DataIter, num_epoch: int=1) -> np.array:
        for epoch in range(num_epoch):
            for batch in batches:
                self.mxmod.forward_backward(batch)
                self.mxmod.update()

            # sync aux params across devices
            arg_params, aux_params = self.mxmod.get_params()
            self.mxmod.set_params(arg_params, aux_params)

            # end of 1 epoch, reset the data-iter for another epoch
            batches.reset()

        return self.predict(batches)

    def predict(self, batches: mx.io.DataIter) -> np.array:
        return self.mxmod.predict(batches).asnumpy()
        #return ys[:, range(self.label_size)] if ys.shape[1] > self.label_size else ys

    def train(self, lexicon: NLPLexicon, trn_graphs: List[NLPGraph], dev_graphs: List[NLPGraph],
              num_steps=20000, batch_size=32,
              initializer: mx.initializer.Initializer = mx.initializer.Normal(0.01),
              arg_params=None, aux_params=None,
              allow_missing: bool=False, force_init: bool=False,
              kvstore: Union[str, mx.kvstore.KVStore] = 'local',
              optimizer: Union[str, mx.optimizer.Optimizer] = 'sgd',
              optimizer_params=(('learning_rate', 0.01),)):
        trn_states: List[NLPState] = [self.create_state(graph, lexicon, save_gold=True) for graph in trn_graphs]
        dev_states: List[NLPState] = [self.create_state(graph, lexicon, save_gold=True) for graph in dev_graphs]
        complete: int = 0

        for step in range(1, num_steps+1):
            st = time.time()
            shuffle(trn_states)
            xs, ys = self.train_instances(trn_states)
            batches = self.bind(xs, ys, batch_size=batch_size)

            if step == 1:
                self.mxmod.init_params(
                    initializer=initializer, arg_params=arg_params, aux_params=aux_params,
                    allow_missing=allow_missing, force_init=force_init)
                self.mxmod.init_optimizer(
                    kvstore=kvstore, optimizer=optimizer, optimizer_params=optimizer_params)

            predictions = self.fit(batches)
            correct = 0

            for state, y, yhats in zip(trn_states, ys, predictions):
                yh = np.argmax(yhats if len(yhats) == self.label_size else yhats[:self.label_size])
                state.process(self.get_label(yh), yhats)
                if y == yh: correct += 1
                if state.terminate:
                    state.reset()
                    complete += 1

            if complete > len(trn_graphs) / 5:
                dev_eval = '%6.4f' % self.evaluate(dev_states, batch_size=batch_size)
                complete = 0
            else:
                dev_eval = 'N/A'

            tt = time.time() - st
            trn_acc = correct / len(ys)
            logging.info('%6d: trn-acc = %6.4f, dev-eval = %6s, time = %d' % (step, trn_acc, dev_eval, tt))

    def evaluate(self, states: List[NLPState], batch_size=128):
        for state in states: state.reset()
        backup = states

        while states:
            xs = self.feature_vectors(states)
            batches: mx.io.NDArrayIter = self.bind(xs, batch_size=batch_size, for_training=False)
            predictions = self.predict(batches)

            for state, yhats in zip(states, predictions):
                yh = np.argmax(yhats if len(yhats) == self.label_size else yhats[:self.label_size])
                state.process(self.get_label(yh), yhats)

            states = [state for state in states if not state.terminate]

        stats = np.array([0, 0])
        acc = 0

        for state in backup:
            acc = state.eval(stats)

        return acc

    # ============================== Helper ==============================

    @classmethod
    def data_iter(cls, data: np.array, label: np.array=None, batch_size=32) -> mx.io.DataIter:
        batch_size = len(data) if len(data) < batch_size else batch_size
        return mx.io.NDArrayIter(data=data, label=label, batch_size=batch_size, shuffle=False)
