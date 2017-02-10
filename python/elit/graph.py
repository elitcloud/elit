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
from igraph import *
__author__ = 'Jinho D. Choi'


ATTR_WORD = 'w'
ATTR_LEMMA = 'l'
ATTR_POS = 'p'
ATTR_NAME = 'n'
ATTR_FEATS = 'f'

FIELD_ROOT = '@#r$%'
FIELD_BLANK = '_'
DELIM_FEAT = '|'
DELIM_FEAT_KV = '='
DELIM_ARC = ';'
DELIM_ARC_NL = ':'


class NLPGraph(Graph):
    def __init__(self, edges=None):
        super(NLPGraph, self).__init__(directed=True, edges=edges)



class NLPGraph2:
    """
    :param nodes: dependency nodes.
    :type nodes: Tuple[NLPNode]
    A root node is automatically added as the first element in the graph.
    The iterator skips the root node and starts with the first "real" node.
    The length of the graph is the number of "real" nodes.
    """
    def __init__(self, nodes=None):
        graph = Graph(directed=True)

        if nodes:
            graph.add_vertices(len(nodes)+1)


        self.graph = graph



        self.nodes = [Fields(node_id=0, word=FIELD_ROOT, lemma=FIELD_ROOT, pos=FIELD_ROOT, nament=FIELD_ROOT)]
        self.nodes.extend(nodes)

    def __next__(self):
        if self._idx >= len(self.nodes):
            raise StopIteration

        node = self.nodes[self._idx]
        self._idx += 1
        return node

    def __iter__(self):
        self._idx = 1
        return self

    def __str__(self):
        return '\n'.join(map(str, self))

    def __len__(self):
        return len(self.nodes) - 1




    '''
    def __str__(self):
        def field(f):
            return f if f else FIELD_BLANK

        token_id = str(self.node_id)
        word = field(self.word)
        lemma = field(self.lemma)
        pos = field(self.pos)
        nament = field(self.nament)
        feats = DELIM_FEAT.join(DELIM_FEAT_KV.join((k, v)) for k, v in self.feats.items()) if self.feats else FIELD_BLANK
        head_id = str(self.head.node.node_id) if self.head and self.head.node else FIELD_BLANK
        deprel = field(self.head.label) if self.head else FIELD_BLANK
        snd_heads = DELIM_ARC.join(str(arc) for arc in self.snd_heads)
        return '\t'.join((token_id, word, lemma, pos, feats, head_id, deprel, snd_heads, nament))

    def __lt__(self, other):
        return self.node_id < other.token_id
    '''