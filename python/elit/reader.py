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
from elit.graph import *
import re
__author__ = 'Jinho D. Choi'

_TAB      = re.compile('\t')
_FEATS    = re.compile('\\|')
_FEATS_KV = re.compile(DELIM_FEAT_KV)
_ARC      = re.compile(DELIM_ARC)
_ARC_NL   = re.compile(DELIM_ARC_NL)


class TSVReader:
    """
    :param filename: the name of a TSV file.
    :type  filename: str
    :param word_index: the column index of word forms.
    :type  word_index: int
    :param lemma_index: the column index of lemma.
    :type  lemma_index: int
    :param pos_index: the column index of part-of-speech tags.
    :type  pos_index: int
    :param feats_index: the column index of extra features.
    :type  feats_index: int
    :param head_id_index: the column index of primary head IDs.
    :type  head_id_index: int
    :param deprel_index: the column index of primary dependency labels.
    :type  deprel_index: int
    :param snd_heads_index: the column index of secondary dependency heads.
    :type  snd_heads_index: int
    :param nament_index: the column index of of named entity tags.
    :type  nament_index: int
    """

    def __init__(self, filename=None, word_index=-1, lemma_index=-1, pos_index=-1, feats_index=-1, head_id_index=-1,
                 deprel_index=-1, snd_heads_index=-1, nament_index=-1):
        if filename:
            self.fin = self.open(filename)

        self.word_index      = word_index
        self.lemma_index     = lemma_index
        self.pos_index       = pos_index
        self.feats_index     = feats_index
        self.head_id_index   = head_id_index
        self.deprel_index    = deprel_index
        self.snd_heads_index = snd_heads_index
        self.nament_index    = nament_index

    def __next__(self):
        graph = self.next()
        if graph:
            return graph
        else:
            raise StopIteration

    def __iter__(self):
        return self

    def open(self, filename):
        """
        :param filename: the name of a TSV file.
        :type  filename: str
        """
        self.fin = open(filename, buffering=200)
        return self.fin

    def close(self):
        self.fin.close()

    def next(self):
        tsv = []

        for line in self.fin:
            line = line.strip()
            if line:
                tsv.append(_TAB.split(line))
            elif tsv:
                break

        if tsv:
            return self.tsv_to_graph(tsv)
        else:
            return None

    def next_all(self):
        return [graph for graph in self]

    def tsv_to_graph(self, tsv):
        """
        :param tsv: each row represents a token, each column represents a field.
        :type  tsv: List[List[str]]
        """
        def get_field(row, index):
            return ROOT_TAG if row is None else row[index] if index >= 0 else None

        def get_feats(row):
            if self.feats_index >= 0:
                f = row[self.feats_index]
                if f == BLANK_FIELD:
                    return None
                return {feat[0]: feat[1] for feat in map(_FEATS_KV.split, _FEATS.split(f))}
            return None

        def set_fields(vertex, row=None):
            vertex[WORD]   = get_field(row, self.word_index)
            vertex[LEMMA]  = get_field(row, self.lemma_index)
            vertex[POS]    = get_field(row, self.pos_index)
            vertex[NAMENT] = get_field(row, self.nament_index)
            vertex[FEATS]  = get_feats(row) if row else {}

        g = NLPGraph()
        g.add_vertices(len(tsv) + 1)

        # initialize fields
        set_fields(g.vs[0])  # root
        for i, t in enumerate(tsv, 1):
            set_fields(g.vs[i], t)

        # initialize primary dependency heads
        if self.head_id_index >= 0:
            es = [(int(t[self.head_id_index]), i) for i, t in enumerate(tsv, 1)]
            ls = [t[self.deprel_index] for t in tsv]
            ts = [EDGE_TYPE_PRIMARY for _ in tsv]

            if self.snd_heads_index >= 0:
                for i, t in enumerate(tsv, 1):
                    arcs = t[self.snd_heads_index]
                    if arcs != BLANK_FIELD:
                        for arc in _ARC.split(arcs):
                            p = _ARC_NL.split(arc)
                            es.append((int(p[0]), i))
                            ls.append(p[1])
                            ts.append(EDGE_TYPE_SECONDARY)

            g.add_edges(es)
            g.es[LABEL] = ls
            g.es[TYPE]  = ts

        return g


fn = '/Users/jdchoi/Documents/Data/english/ontonotes.ddg1'
reader = TSVReader(filename=fn, word_index=1, lemma_index=2, pos_index=3, feats_index=4, head_id_index=5, deprel_index=6, snd_heads_index=7, nament_index=8)
import time
st = time.time()
count = 0
for g in reader:
    for i,v in enumerate(g,1):
        print(str(i)+' '+ str(next(g.es[e].source for e in g.incident(v, mode=IN) if g.es[e][TYPE] == EDGE_TYPE_PRIMARY)))
    break
et = time.time()
print(et-st)
