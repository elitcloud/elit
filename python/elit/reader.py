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
import time
__author__ = 'Jinho D. Choi'

RE_TAB = re.compile('\t')
RE_FEATS = re.compile('\\|')
RE_FEATS_KV = re.compile(DELIM_FEAT_KV)
RE_ARC = re.compile(DELIM_ARC)
RE_ARC_NL = re.compile(DELIM_ARC_NL)


class TSVReader:
    def __init__(self, fin=None, word_index=-1, lemma_index=-1, pos_index=-1, feats_index=-1, head_id_index=-1, deprel_index=-1, snd_heads_index=-1, nament_index=-1):
        self.fin = fin
        self.word_index = word_index
        self.lemma_index = lemma_index
        self.pos_index = pos_index
        self.feats_index = feats_index
        self.head_id_index = head_id_index
        self.deprel_index = deprel_index
        self.snd_heads_index = snd_heads_index
        self.nament_index = nament_index

    def __next__(self):
        graph = self.next()
        if graph:
            return graph
        else:
            raise StopIteration

    def __iter__(self):
        return self

    # fin : file input-stream
    def open(self, fin):
        self.fin = fin

    def close(self):
        self.fin.close()

    def next(self):
        tokens = list()

        for line in self.fin:
            line = line.strip()
            if line:
                tokens.append(RE_TAB.split(line))
            elif tokens:
                break

        if tokens:
            return self.tsv_to_graph(tokens)
        else:
            return None

    def next_all(self):
        return [graph for graph in self]

    def tsv_to_graph(self, tsv):
        def set_fields(t, v)

        def to_field(t, index):
            return t[index] if index >= 0 else None

        def to_feats(t):
            if self.feats_index < 0:
                return None

            f = t[self.feats_index]
            if f == FIELD_BLANK:
                return None

            return {p[0]: p[1] for p in map(RE_FEATS_KV.split, RE_FEATS.split(f))}

        if self.head_id_index >= 0:
            es = [(int(t[self.head_id_index]), i+1) for i,t in enumerate(tsv)]
            g = NLPGraph(edges=es)
        else:
            g = NLPGraph()
            g.add_vertices(len(tsv) + 1)



        v = g.vs[0]
        v['word'] = FIELD_ROOT
        v['lemma'] = FIELD_ROOT
        v['pos'] = FIELD_ROOT
        v['nament'] = FIELD_ROOT
        v['feats'] = {}


        for i, t in enumerate(tsv):
            v = g.vs[i+1]
            v['word'] = to_field(t, self.word_index)
            v['lemma'] = to_field(t, self.lemma_index)
            v['pos'] = to_field(t, self.pos_index)
            v['nament'] = to_field(t, self.nament_index)
            v['feats'] = to_feats(t)

        return g


filename = '/Users/jdchoi/Documents/Data/experiments/general-en/trn/ontonotes_nw.trn'
reader = TSVReader(fin=open(filename), word_index=1, lemma_index=2, pos_index=3, feats_index=4, head_id_index=5, deprel_index=6, snd_heads_index=7, nament_index=8)

st = time.time()
for g in reader:
    for v in g.vs:

        s = v['fields'].word
        s = v['fields'].lemma
        s = v['fields'].pos
        s = v['fields'].nament
        '''
        s = v['word']
        s = v['lemma']
        s = v['pos']
        s = v['nament']
        '''

et = time.time()
print(et-st)

'''

def tsv_to_ddg(tsv, node_id_index=0, word_index=1, lemma_index=2, pos_index=3, feats_index=4, head_id_index=5, deprel_index=6, snd_heads_index=7, name_index=8):
    """
    :param tsv: list of fields from a TSV file.
    :param word_index: column index of the word form in the TSV file.
    :param lemma_index: column index of the lemma in the TSV file.
    :param pos_index: column index of the part-of-speech tag in the TSV file.
    :param feats_index: column index of the extra features in the TSV file.
    :param head_id_index: column index of the primary head ID in the TSV file.
    :param deprel_index: column index of the primary dependency label in the TSV file.
    :param snd_heads_index: column index of the secondary heads in the TSV file.
    :param name_index: column index of the named entity tag in the TSV file.
    :return: the field if exists; otherwise, None.
    """



    def to_field(t, index):
        return t[index] if 0 <= index < len(t) else None

    def to_feats(t):
        if 0 <= feats_index < len(t):
            f = t[feats_index]
            if f == FIELD_BLANK:
                return None
            l = RE_FEATS.split(f)
            return {p[0]:p[1] for p in map(RE_FEATS_KV.split, RE_FEATS.split(f))}

        return None

    def to_arc(graph, head_id, label):
        return NLPArc(graph.nodes[head_id], label)

    def to_head(graph, node_index):
        t = tsv[node_index]
        if 0 <= head_id_index < len(t):
            graph.nodes[node_index + 1].head = to_arc(graph, int(t[head_id_index]), to_field(t, deprel_index))

    def to_snd_heads(graph, node_index):
        t = tsv[node_index]
        if 0 <= snd_heads_index < len(t):
            f = t[snd_heads_index]
            if f == FIELD_BLANK:
                return None
            return [to_arc(graph, int(p[0]), p[1]) for p in map(RE_ARC_NL.split, RE_ARC.split(f))]
        return None

    g = NLPGraph()
    g.add_vertices(len(tsv)+1)

    if is_range(node_id_index) and is_range(head_id_index):
        edges = [(int(t[head_id_index]), int(t[node_id_index])) for t in tsv]
        g = NLPGraph(edges=edges)




        for i, t in enumerate(tsv):
            node_id = i+1
            head_id = int(t[head_id_index])





    for i in range(1, len(g)):





    g = NLPGraph([NLPNode(node_id=i + 1, word=to_field(t, word_index), lemma=to_field(t, lemma_index), pos=to_field(t, pos_index), feats=to_feats(t), nament=to_field(t, name_index)) for i, t in enumerate(tsv)])

    for i in range(1, len(g)):
        to_head(g, i)
        to_snd_heads(g, i)


    def is_range(index):
        return 0 <= index < len(tsv[0])

    if is_range(head_id_index):
        edges = [(int(t[head_id_index]), i+1) for i,t in enumerate(tsv)]
        g = NLPGraph(edges=edges)
    else:
        g = NLPGraph()
        g.add_vertices(len(tsv)+1)

    if is_range(word_index):
        g.vs['word'] = [t[word_index] for t in tsv]

    return g

'''
