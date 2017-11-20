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
from fasttext.model import WordVectorModel
from gensim.models import KeyedVectors

from elit.dev.structure import NLPGraph
from elit.dev.template.lexicon import NLPLexiconMapper
from elit.dev.template.state import NLPState

__author__ = 'Jinho D. Choi'

# transitions
LEFT_ARC  = 'L'
RIGHT_ARC = 'R'
NO_ARC    = 'N'
SHIFT     = 'S'
REDUCE    = 'R'
PASS      = 'P'


class DEPLexicon(NLPLexiconMapper):
    def __init__(self, w2v: KeyedVectors=None, f2v: WordVectorModel=None):
        super().__init__(w2v, f2v)


class DEPState(NLPState):
    def __init__(self, graph: NLPGraph, lexicon: DEPLexicon, save_gold=False):
        super().__init__(graph)
        self.lex: DEPLexicon = lexicon

        # reset
        self.stack: List[int] = [0]
        self.inter: List[int] = []
        self.input: int = 1

    # ============================== Node ==============================

    def get_stack(self, window: int=0, relation: Relation=None) -> NLPNode:
        """
        :param window: the context window to the top of the stack.
        :param relation: the relation to the (top+window)'th node.
        :return: relation(top+window)'th node if exists; otherwise, None.
        """
        return self.get_node(index=self.stack[-1], window=window, relation=relation, root=True)

    def get_input(self, window: int=0, relation: Relation=None) -> NLPNode:
        """
        :param window: the context window to the front of the input.
        :param relation: the relation to the (input+window)'th node.
        :return: relation(input+window)'th node if exists; otherwise, None.
        """
        return self.get_node(index=self.input, window=window, relation=relation, root=True)

    # ============================== Transition ==============================

    def next(self, label: str):
        """
        :param label: arc + transition + y
        """
        s = self.get_stack()
        i = self.get_input()
        a = label[0]  # arc
        t = label[1]  # transition

        if a == LEFT_ARC:
            s.set_parent(i, label[2:])
            if t == REDUCE: self.reduce()
            else: self.passes()
        elif a == RIGHT_ARC:
            i.set_parent(s, label[2:])
            if t == SHIFT: self.shift()
            else: self.passes()
        else:
            if t == SHIFT: self.shift()
            elif t == REDUCE: self.reduce()
            else: self.passes()

    def shift(self):
        self.stack.extend(reversed(self.inter))
        self.stack.append(self.input)
        del self.inter[:]
        self.input += 1

    def reduce(self):
        self.stack.pop()

    def passes(self):
        self.inter.append(self.stack.pop())

    def terminate(self):
        return self.input >= len(self.graph)

    # ============================== Node ==============================


class DEPParser(NLPComponent):
    def init_state(self, graph: NLPGraph) -> NLPState:
        return DEPState(graph)


