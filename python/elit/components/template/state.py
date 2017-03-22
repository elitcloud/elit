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
from typing import Union, Tuple, List

import numpy as np

from elit.components.template.lexicon import NLPLexicon
from elit.structure import NLPGraph, NLPNode, Relation

__author__ = 'Jinho D. Choi'


class NLPState(metaclass=ABCMeta):
    def __init__(self, lexicon: NLPLexicon, graph: NLPGraph):
        self.lexicon: NLPLexicon = lexicon
        self.graph: NLPGraph = graph
        lexicon.init(graph)

    @abstractmethod
    def reset(self):
        """ Reset to the initial state and remove all tags for training. """

    # ============================== Oracle ==============================

    @property
    @abstractmethod
    def gold_label(self) -> str:
        """ :return: the gold label for the current state if exists; otherwise, None. """

    @abstractmethod
    def eval_counts(self) -> np.array:
        """ :return: [total, count1, count2, ...] if golds exist; otherwise, []. """

    # ============================== Transition ==============================

    @abstractmethod
    def process(self, label: str, scores: np.array=None):
        """
        :param label: the label for the current state.
        :param scores: prediction scores.
         Perform the next transition given the label.
        """

    @abstractmethod
    def terminate(self) -> bool:
        """ :return: True if no more state can be processed; otherwise, False. """

    # ============================== Feature ==============================

    @abstractmethod
    def features(self, node: NLPNode) -> List[np.array]:
        """
        :return: features extracted from lexicons.
        """

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
