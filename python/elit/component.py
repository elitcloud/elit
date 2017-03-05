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
import numpy as np
from typing import Tuple
from typing import Generator
from typing import Callable
from abc import ABCMeta
from abc import abstractmethod
from elit.structure import *
from elit.model import NLPModel
from elit.reader import TSVReader

__author__ = 'Jinho D. Choi'


class NLPState(metaclass=ABCMeta):
    def __init__(self, graph: NLPGraph):
        """
        :param graph: the input graph.
        """
        self.graph: NLPGraph = graph

    # ============================== Transition ==============================

    def generate_train_instances(self, model: NLPModel, prev_model: NLPModel=None) -> Generator[Tuple[str, np.array]]:
        """
        :param model: the model to be trained.
        :param prev_model: a pre-trained model used for data aggregation.
        :return: a generator of training instances [(y, x), ...].
        """
        while not self.terminate:
            gold = self.gold_label
            x = model.create_feature_vector(self)
            y = prev_model.predict_label(x) if prev_model else gold
            self.next(y)
            yield gold, x

    def decode(self, model: NLPModel):
        """
        :param model: the model to make a predict.
        """
        while not self.terminate:
            x = model.create_feature_vector(self)
            y = model.predict_label(x)
            self.next(y)

    @abstractmethod
    def next(self, label: str):
        """
        :param label: the next transition.
          Perform the next transition given the label.
        """

    @abstractmethod
    def terminate(self) -> bool:
        """
        :return: True if no more state can be processed; otherwise, False.
        """

    # ============================== Oracle ==============================

    @abstractmethod
    def clear_oracle(self) -> bool:
        """
        :return: True if gold labels are saved; otherwise, False.
          Save and remove the gold labels from the input graph.
        """

    @abstractmethod
    def reset_oracle(self) -> np.array:
        """
        :return: counts (e.g., correct, total) for evaluation.
          Put the gold labels back to the input graph.
        """

    @property
    @abstractmethod
    def gold_label(self) -> str:
        """
        :return: the gold label for the current state if exists; otherwise, None.
        """

    # ============================== Node ==============================

    def get_node(self, index: int, window: int=0, relation: Relation=None, root: bool=False) -> Union[NLPNode, None]:
        """
        :param index: the index of the anchor node.
        :param window: the context window to the anchor node.
        :param relation: the relation to the (index+window)'th node.
        :param root: if True, the root (nodes[0]) is returned when the condition is met; otherwise, None.
        :return: the relation(index+window)'th node in the graph if exists; otherwise, None.
        """
        index += window
        begin = 0 if root else 1
        node: NLPNode = self.graph.nodes[index] if begin <= index < len(self.graph) else None

        if node and relation:
            # 1st order
            if relation == Relation.PARENT:
                return node.parent
            if relation == Relation.LEFTMOST_CHILD:
                return node.get_leftmost_child()
            if relation == Relation.RIGHTMOST_CHILD:
                return node.get_rightmost_child()
            if relation == Relation.LEFT_NEAREST_CHILD:
                return node.get_left_nearest_child()
            if relation == Relation.RIGHT_NEAREST_CHILD:
                return node.get_right_nearest_child()
            if relation == Relation.LEFTMOST_SIBLING:
                return node.get_leftmost_sibling()
            if relation == Relation.RIGHTMOST_SIBLING:
                return node.get_rightmost_sibling()
            if relation == Relation.LEFT_NEAREST_SIBLING:
                return node.get_left_nearest_sibling()
            if relation == Relation.RIGHT_NEAREST_SIBLING:
                return node.get_right_nearest_sibling()

            # 2nd order
            if relation == Relation.GRANDPARENT:
                return node.grandparent
            if relation == Relation.SND_LEFTMOST_CHILD:
                return node.get_leftmost_child(1)
            if relation == Relation.SND_RIGHTMOST_CHILD:
                return node.get_rightmost_child(1)
            if relation == Relation.SND_LEFT_NEAREST_CHILD:
                return node.get_left_nearest_child(1)
            if relation == Relation.SND_RIGHT_NEAREST_CHILD:
                return node.get_right_nearest_child(1)
            if relation == Relation.SND_LEFTMOST_SIBLING:
                return node.get_leftmost_sibling(1)
            if relation == Relation.SND_RIGHTMOST_SIBLING:
                return node.get_rightmost_sibling(1)
            if relation == Relation.SND_LEFT_NEAREST_SIBLING:
                return node.get_left_nearest_sibling(1)
            if relation == Relation.SND_RIGHT_NEAREST_SIBLING:
                return node.get_right_nearest_sibling(1)

        return node

    def is_first(self, node: NLPNode) -> bool:
        """
        :param node: the node to be compared
        :return: True if the node is the first node in the graph; otherwise, False
        """
        return len(self.graph) > 1 and self.graph.nodes[1] == node

    def is_last(self, node: NLPNode) -> bool:
        """
        :param node: the node to be compared
        :return: True if the node is the last node in the graph; otherwise, False
        """
        return self.graph.nodes[-1] == node


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
        if not state.clear_oracle(): return []
        instances = [(y, x) for y, x in state.generate_train_instances(self.model, prev_model)]
        self._evaluate(state)
        return instances

    def evaluate(self, graph: NLPGraph) -> np.array:
        """
        :param graph: the input graph.
        :return: counts (e.g., correct, total) for the evaluation
        """
        state: NLPState = self.init_state(graph)
        if state.clear_oracle():
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
        counts = state.reset_oracle()
        if self.eval_counts: self.eval_counts += counts
        else: self.eval_counts = counts





class NLPTrainer:
    def __init__(self, init_state: Callable[[NLPGraph], NLPState],
                 trn_graphs: List[NLPGraph], dev_graphs: List[NLPGraph], aggregate: bool=False):
        """
        :param init_state: a function to initialize the processing state.
        :param trn_graphs: the list of graphs for training.
        :param dev_graphs: the list of graphs for validation.
        :param aggregate: if True, data aggregation is applied during training.
        """
        self.init_state: Callable[[NLPGraph], NLPState] = init_state
        self.trn_graphs: List[NLPGraph] = trn_graphs
        self.dev_graphs: List[NLPGraph] = dev_graphs
        self.aggregate = aggregate

    @abstractmethod
    def init_component(self, model: NLPModel=None) -> NLPComponent:
        """
        :return: the NLP components.
        """

    def train(self, graphs: List[NLPGraph], prev_model: NLPModel=None):
        model = NLPModel()
        train_instances = []

        for graph in graphs:
            state = self.init_state(graph)
            train_instances.extend(state.generate_train_instances(model, prev_model))





        component = self.init_component(prev_model, NLPFlag.DECODE)


class NLPLexicon:
    def __init__(self, word_vectors: Dict[str, np.array]=None, ambiguity_classes: Dict[str, np.array]=None):
        self.word_vectors = word_vectors
        self.ambiguity_classes = ambiguity_classes

        self.default_word_vector: np.array = np.zeros(len(next(iter(word_vectors.values())))) if word_vectors else None
        self.default_ambiguity_class: np.array = np.zeros(len(next(iter(ambiguity_classes.values()))))\
            if ambiguity_classes else None

    def populate(self, graph: NLPGraph):
        for node in graph:
            if self.word_vectors:
                node.word_vector = self.word_vectors.get(node.word, self.default_word_vector)
            if self.ambiguity_classes:
                node.ambiguity_class = self.ambiguity_classes.get(node.word, self.default_ambiguity_class)


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
        for graph in graphs: lexicon.populate(graph)
    return graphs
