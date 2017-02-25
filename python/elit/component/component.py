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
import mxnet as mx
import numpy as np
from typing import Tuple
from typing import Callable
from abc import ABCMeta
from abc import abstractmethod
from enum import Enum
from elit.structure import *
from elit.model import NLPModel

__author__ = 'Jinho D. Choi'


class NLPState(metaclass=ABCMeta):
    def __init__(self, graph: NLPGraph, model: NLPModel, flag: NLPFlag=NLPFlag.DECODE):
        self.graph: NLPGraph = graph
        self.model: NLPModel = model
        self.flag: int = flag

    def __next__(self):
        if self.terminate: raise StopIteration
        return self.next()

    def __iter__(self):
        return self

    def next(self) -> Union[Tuple[int, np.array], None]:
        """
        :return: None if decoding; otherwise, the tuple of (label, feature vector).
        """
        x = self.feature_vector

        if self.flag == NLPFlag.TRAIN:
            label = self.gold_label
            self.perform(label)
            return self.model.get_label_index(label), x
        elif self.flag == NLPFlag.DAGGER:
            label = self.gold_label
            y = self.model.predict
            self.perform(self.model.labels[y])
            return self.model.get_label_index(label), x
        else:  # evaluate or decode
            y = self.model.predict
            self.perform(self.model.labels[y])
            return None

    @abstractmethod
    def perform(self, label: str):
        """
        :param label: the gold or predicted label for the current state.
        """

    @abstractmethod
    def terminate(self) -> bool:
        """
        :return: True if no more state can be processed; otherwise, False.
        """

    # ============================== Oracle ==============================

    @property
    @abstractmethod
    def feature_vector(self) -> np.array:
        """
        :return: the feature vector representing the current state.
        """

    @abstractmethod
    def save_oracle(self) -> bool:
        """
        :return: True if gold labels are saved; otherwise, False.
        Save and remove gold labels from the input graph.
        """

    @property
    @abstractmethod
    def gold_label(self) -> str:
        """
        :return: the gold label for the current state if exists; otherwise, None.
        """

    @abstractmethod
    def reset_oracle(self):
        """ Put the gold labels back to the input graph. """

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


class NLPComponent(metaclass=ABCMeta):
    def __init__(self, flag: NLPFlag=NLPFlag.DECODE):
        self.flag: NLPFlag = flag
        if self.flag_train_or_dagger:
            self.YX: List[Tuple[int, np.array]] = []   # feature vectors

    @abstractmethod
    def init_state(self, graph: NLPGraph) -> NLPState:
        """
        :return: the initial processing state of this component.
        """

    def clear_train_instances(self):
        """ Removes currently stored training instances. """
        del self.X[:]
        del self.Y[:]

    def process(self, graph: NLPGraph) -> NLPState:
        """
        :param graph: the input graph.
        """
        state: NLPState = self.init_state(graph)
        if not self.flag_decode and not state.save_oracle(): return state

        if self.flag_train_or_dagger:
            self.YX.extend([(y, x) for y, x in state])
        elif self.flag == NLPFlag.EVALUATE:
            self.evaluate()








    # ============================== Flag ==============================

    @property
    def flag_train_or_dagger(self):
        return self.flag == NLPFlag.TRAIN or self.flag == NLPFlag.DAGGER

    @abstractmethod
    def evaluate(self):
        """
        Evaluate
        """




#mxnet.module.module.Module


class NLPFlag(Enum):
    TRAIN    = 0
    DECODE   = 1
    EVALUATE = 2
    DAGGER   = 3


class Relation(Enum):
    PARENT                    = 'p'
    LEFTMOST_CHILD            = 'lmc'
    RIGHTMOST_CHILD           = 'rmc'
    LEFT_NEAREST_CHILD        = 'lnc'
    RIGHT_NEAREST_CHILD       = 'rnc'
    LEFTMOST_SIBLING          = 'lms'
    RIGHTMOST_SIBLING         = 'rms'
    LEFT_NEAREST_SIBLING      = 'lns'
    RIGHT_NEAREST_SIBLING     = 'rns'

    GRANDPARENT               = 'gp'
    SND_LEFTMOST_CHILD        = 'lmc2'
    SND_RIGHTMOST_CHILD       = 'rmc2'
    SND_LEFT_NEAREST_CHILD    = 'lnc2'
    SND_RIGHT_NEAREST_CHILD   = 'rnc2'
    SND_LEFTMOST_SIBLING      = 'lms2'
    SND_RIGHTMOST_SIBLING     = 'rms2'
    SND_LEFT_NEAREST_SIBLING  = 'lns2'
    SND_RIGHT_NEAREST_SIBLING = 'rns2'
