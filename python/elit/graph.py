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
from itertools import islice
from igraph import *
__author__ = 'Jinho D. Choi'


# attributes
WORD   = 'w'
LEMMA  = 'm'
POS    = 'p'
NAMENT = 'n'
FEATS  = 'f'
LABEL  = 'l'
TYPE   = 't'

# edge types
EDGE_TYPE_PRIMARY   = 0
EDGE_TYPE_SECONDARY = 1

# fields
BLANK_FIELD = '_'
ROOT_TAG = '@#r$%'

# delimiters
DELIM_FEAT    = '|'
DELIM_FEAT_KV = '='
DELIM_ARC     = ';'
DELIM_ARC_NL  = ':'


class NLPGraph(Graph):
    def __init__(self):
        super(NLPGraph, self).__init__(directed=True)

    def __next__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopIteration

    def __iter__(self):
        self._iter = islice(self.vs, 1, len(self.vs))
        return self

    def __str__(self):
        return '\n'.join(self.vertex_to_tsv(v) for v in self)

    def get_primary_head(self, vertex):
        head_id = next((self.es[i].source for i in self.incident(vertex, mode=IN) if
                        self.es[i][TYPE] == EDGE_TYPE_PRIMARY), None)
        return 

        self.incident(vertex, mode=IN)




def vertex_to_tsv(self, vertex):
    token_id = str(vertex.index)
    word     = vertex[WORD] or BLANK_FIELD
    lemma    = vertex[LEMMA] or BLANK_FIELD
    pos      = vertex[POS] or BLANK_FIELD
    nament   = vertex[NAMENT] or BLANK_FIELD
    feats    = DELIM_FEAT.join(
               DELIM_FEAT_KV.join((k, v)) for k, v in vertex[FEATS].items()) if vertex[FEATS] else BLANK_FIELD


    return '\t'.join((token_id, word, lemma, pos, feats, nament))
    #return '\t'.join((token_id, word, lemma, pos, feats, head_id, deprel, snd_heads, nament))