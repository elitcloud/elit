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
from abc import ABCMeta
from abc import abstractmethod
from typing import Callable
from typing import Generator
from typing import Tuple
from typing import List
import numpy as np
from elit.components.template.lexicon import NLPLexicon
from elit.components.template.model import NLPModel
from elit.components.template.state import NLPState
from elit.structure import NLPGraph

__author__ = 'Jinho D. Choi'


class NLPTrain(metaclass=ABCMeta):
    def __init__(self, lexicon: NLPLexicon, model: NLPModel=None):
        self.lexicon: NLPLexicon = lexicon
        self.model: NLPModel = model if model else self.init_model()

    @abstractmethod
    def init_state(self, graph: NLPGraph) -> NLPState:
        """ :return: an NLP state for the input graph. """

    @abstractmethod
    def init_model(self) -> NLPModel:
        """ :return: an NLP model to be trained. """

    def feature_vectors(self, states: List[NLPState]) -> np.array:
        xs = [state.x for state in states]
        return np.vstack(xs)

    def train(self, trn_graphs: List[NLPGraph], dev_graphs: List[NLPGraph], epochs=1000):
        trn_states = [self.init_state(graph) for graph in trn_graphs]
        dev_states = [self.init_state(graph) for graph in dev_graphs]

        for epoch in range(1, epochs+1):
            xs = self.feature_vectors(trn_states)
            self.model.train(xs)
            ys = self.model.predict(xs)
            for i, state in enumerate(trn_states):
                state.process((ys[i], self.model))
                if state.terminate: state.reset()




        train_instances = []

        for graph in graphs:
            state = self.init_state(graph)
            train_instances.extend(state.generate_train_instances(model, prev_model))





        component = self.init_component(prev_model, NLPFlag.DECODE)

    def decode(self, graphs: List[NLPGraph]):
        states = [self.init_state(graph) for graph in graphs]

        while states:
            xs = self.feature_vectors(states)
            ys = self.model.predict(xs)
            for i, state in enumerate(states):
                state.process((ys[i], self.model))
            states = [state for state in states if not state.terminate]

    def evaluate(self, graphs: List[NLPGraph]):
        states = [self.init_state(graph) for graph in graphs]

        while states:
            xs = self.feature_vectors(states)



            ys = self.model.predict(xs)
            for i, state in enumerate(states):
                state.process((ys[i], self.model))
            states = [state for state in states if not state.terminate]



