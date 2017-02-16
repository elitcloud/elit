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
from elit.structure import *
__author__ = 'Jinho D. Choi'


class NLPState(metaclass=ABCMeta):
    def __init__(self, graph: NLPGraph):
        self.graph = graph

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
