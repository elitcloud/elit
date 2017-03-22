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
from elit.components.template.lexicon import NLPLexicon
from elit.components.template.model import NLPModel
from elit.components.template.state import NLPState
from elit.structure import NLPGraph

from abc import ABCMeta, abstractmethod
from typing import List, Union, Tuple
from random import shuffle
import mxnet as mx
import numpy as np
import logging
import time

__author__ = 'Jinho D. Choi'


class NLPComponent(metaclass=ABCMeta):
    def __init__(self, lexicon: NLPLexicon, model: NLPModel):
        self.lexicon: NLPLexicon = lexicon
        self.model: NLPModel = model

    @abstractmethod
    def create_state(self, graph: NLPGraph, save_gold: bool=False) -> NLPState:
        """
        :param graph: the input graph.
        :param save_gold: if True, save gold tags from the input graph.
        :return: the wrapper state for the input graph.
        """

    def train_instances(self, states: List[NLPState]) -> Tuple[np.array, np.array]:
        xs, ys = zip(*[(self.model.x(state), state.gold_label) for state in states])
        return np.vstack(xs), np.array([self.model.add_label(y) for y in ys])

    def feature_vectors(self, states: List[NLPState]) -> np.array:
        xs = [self.model.x(state) for state in states]
        return np.vstack(xs)

    def train(self, trn_graphs: List[NLPGraph], dev_graphs: List[NLPGraph], num_steps=1000,
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
            print(ys[:20])
            print(self.model.labels)

            dat: mx.io.NDArrayIter = self.model.bind(xs, ys)

            if step == 1:
                self.model.mxmod.init_params(
                    initializer=initializer, arg_params=arg_params, aux_params=aux_params,
                    allow_missing=allow_missing, force_init=force_init)
                self.model.mxmod.init_optimizer(
                    kvstore=kvstore, optimizer=optimizer, optimizer_params=optimizer_params)

            self.model.train(dat)
            dat.reset()
            ys = self.model.predict(dat)
            print([np.argmax(ys[i]) for i in range(10)])

            for i, state in enumerate(trn_states):
                state.process(self.model.get_label(np.argmax(ys[i])), ys[i])
                if state.terminate:
                    state.reset()
                    complete += 1

            et = time.time()
            logging.info('%4d: labels = %6d, complete = %d, time = %d' % (step, len(self.model.labels), complete, et-st))

            # if count > len(trn_graphs) / 2:
            #     n = self.evaluate(dev_states)
            #     s = 100.0 * n[1] / n[0]
            #     count = 0
            #     print(s)

    def decode(self, graphs: List[NLPGraph]):
        states = [self.create_state(graph) for graph in graphs]

        while states:
            xs = self.train_instances(states)
            ys = self.model.predict(xs)
            for i, state in enumerate(states):
                state.process((ys[i], self.model))
            states = [state for state in states if not state.terminate]

    def evaluate(self, states: List[NLPState]):
        while states:
            xs, ys = self.train_instances(states)
            dat: mx.io.NDArrayIter = self.model.bind(xs)
            yhats = self.model.predict(dat)

            for i, state in enumerate(states):
                state.process(self.model.get_label(np.argmax(ys[i])), ys[i])

            states = [state for state in states if not state.terminate]



