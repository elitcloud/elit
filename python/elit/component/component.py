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
from enum import Enum
from elit.structure import *
__author__ = 'Jinho D. Choi'


class Relation(Enum):
    parent              = 'p'
    leftmost_child      = 'lmc'
    rightmost_child     = 'rmc'
    left_nearest_child  = 'lnc'
    right_nearest_child = 'rnc'








class NLPState(metaclass=ABCMeta):
    def __init__(self, graph: NLPGraph):
        self.graph = graph

    def get_node(self, index: int, window: int, root: bool=False) -> NLPNode:
        """
        :param index:
        :param window:
        :param root: if True, the root (nodes[0]) is returned when index+window == 0; otherwise, None.
        :return: the index+window'th node in the graph if exists; otherwise, None.
        """
        index += window
        begin = 0 if root else 1
        return self.graph.nodes[index] if begin <= index < len(self.graph) else None




    @abstractmethod
    def save_oracle(self) -> bool:
        """
        :return: True if gold labels are saved; otherwise, False.
        Save and remove gold labels from the input graph.
        """
        return

    @abstractmethod
    def get_oracle(self) -> str:
        """
        :return: the gold label for the current state if exists; otherwise, None.
        """
        return

    @abstractmethod
    def reset_oracle(self):
        """Put the gold labels back to the input graph."""
        pass







class NLPComponent(metaclass=ABCMeta):
    @abstractmethod
    def process(self, graph: NLPGraph):
        """
        :param graph: the input graph.
        """
        return


