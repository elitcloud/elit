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

import numpy as np

from elit.components.template.model import NLPModel
from elit.reader import TSVReader
from elit.structure import *

__author__ = 'Jinho D. Choi'





class NLPComponent:
    def __init__(self, init_state: Callable[[NLPGraph], NLPState], model: NLPModel):
        """
        :param init_state: a function to initialize the processing state.
        :param model: a statistical model.
        """
        self.init_state: Callable[[NLPGraph], NLPState] = init_state
        self.model: NLPModel = model
        self.eval_counts: np.array = None

    def train(self, graph: NLPGraph, prev_model: NLPModel=None) -> List[Tuple[str, np.array]]:
        """
        :param graph: the input graph.
        :param prev_model: a pre-trained model used for data aggregation.
        :return: a list of training instances generated for the input graph.
        """
        state: NLPState = self.init_state(graph)
        if not state.save_tags(): return []
        instances = [(y, x) for y, x in state.generate_train_instances(self.model, prev_model)]
        self._evaluate(state)
        return instances

    def evaluate(self, graph: NLPGraph) -> np.array:
        """
        :param graph: the input graph.
        :return: counts (e.g., correct, total) for the evaluation
        """
        state: NLPState = self.init_state(graph)
        if state.save_tags():
            state.decode(self.model)
            self._evaluate(state)

    def decode(self, graph: NLPGraph):
        """
        :param graph: the input graph
        """
        state: NLPState = self.init_state(graph)
        state.decode(self.model)

    def reset_evaluator(self):
        self.eval_counts.fill(0)

    def _evaluate(self, state: NLPState):
        counts = state.counts()
        if self.eval_counts: self.eval_counts += counts
        else: self.eval_counts = counts










def read_graphs(reader: TSVReader, filename: str, lexicon: NLPLexicon=None) -> List[NLPGraph]:
    """
    :param reader: the tsv-reader.
    :param filename: the name of the file containing graphs.
    :param lexicon: consisting of word embeddings, ambiguity classese, etc.
    :return: the list of graphs.
    """
    reader.open(filename)
    graphs: List[NLPGraph] = reader.next_all()
    reader.close()
    if lexicon:
        for graph in graphs: lexicon.init(graph)
    return graphs

