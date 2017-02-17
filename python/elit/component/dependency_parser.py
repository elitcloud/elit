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
from typing import TypeVar
from typing import List
from elit.component.component import *

__author__ = 'Jinho D. Choi'


class DEPState(NLPState):
    def __init__(self, graph: NLPGraph):
        super().__init__(graph)
        self.stack: List[int] = [0]
        self.inter: List[int] = []
        self.input: int = 1

    # ============================== TRANSITION ==============================

    def next(self, transition):
        w_i =

    def get_stack(self):


    def shift(self):
        self.stack.extend(reversed(self.inter))
        self.stack.append(self.input)
        del self.inter[:]
        self.input += 1

    def reduce(self):
        self.stack.pop()

    def skip(self):
        self.inter.append(self.stack.pop())

    def is_terminate(self):
        return self.input >= len(self.graph)

    # ============================== Node ==============================

    def get_node(self, ):




class TreeParser():
    TRANS_LEFT = 'l'
    TRANS_RIGHT = 'r'
    TRANS_NO = 'n'
    TRANS_SHIFT = 's'
    TRANS_REDUCE = 'r'
    TRANS_PASS = 'p'

    def parse(self, graph: NLPGraph):
        """
        :param graph: a dependency tree to be parsed.
        """

        while input < len(graph):
            i = stack[-1]
            j = input
