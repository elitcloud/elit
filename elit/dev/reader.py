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
import re

from elit.dev.structure import DELIM_ARC, DELIM_ARC_KV, DELIM_FEAT_KV, BLANK, ROOT_TAG, \
    NLPToken, NLPGraph

__author__ = 'Jinho D. Choi'

_TAB = re.compile('\t')
_FEATS = re.compile('\\|')
_FEATS_KV = re.compile(DELIM_FEAT_KV)
_ARC = re.compile(DELIM_ARC)
_ARC_KV = re.compile(DELIM_ARC_KV)


class TSVReader:
    def __init__(self, form_index=-1, lemma_index=-1, pos_index=-1,
                 feats_index=-1, head_index=-1, deprel_index=-1,
                 sheads_index=-1, nament_index=-1):
        """

        :param form_index: the column index of form forms.1
        :type form_index: int
        :param lemma_index: the column index of lemma.
        :type lemma_index: int
        :param pos_index: the column index of part-of-speech tags.
        :type pos_index: int
        :param feats_index: the column index of extra features.
        :type feats_index: int
        :param head_index: the column index of primary head IDs.
        :type head_index: int
        :param deprel_index: the column index of primary dependency sys_labels.
        :type deprel_index: int
        :param sheads_index: the column index of secondary dependency heads.
        :type sheads_index: int
        :param nament_index: the column index of of named entity tags.
        :type nament_index: int
        """
        self.form_index = form_index
        self.lemma_index = lemma_index
        self.pos_index = pos_index
        self.feats_index = feats_index
        self.head_index = head_index
        self.deprel_index = deprel_index
        self.sheads_index = sheads_index
        self.nament_index = nament_index
        self.ins = None

    def __next__(self):
        graph = self.next
        if graph:
            return graph
        else:
            raise StopIteration

    def __iter__(self):
        return self

    @classmethod
    def create_reader(cls, reader):
        """

        :type reader: TSVReader
        :return: a reader adapting the configuration (not the input stream) from the other reader.
        :rtype: TSVReader
        """
        return TSVReader(form_index=reader.form_index,
                         lemma_index=reader.lemma_index,
                         pos_index=reader.pos_index,
                         feats_index=reader.feats_index,
                         head_index=reader.head_index,
                         deprel_index=reader.deprel_index,
                         sheads_index=reader.sheads_index,
                         nament_index=reader.nament_index)

    def open(self, filename):
        """

        :param filename: file
        :type filename: str
        :return: file
        :rtype: TextIOWrapper[str]
        """
        self.ins = open(filename)
        return self.ins

    def close(self):
        """
        Close file
        """
        self.ins.close()

    @property
    def next(self):
        """

        :return: get next graph
        :rtype: NLPGraph
        """
        tsv = []

        for line in self.ins:
            line = line.strip()
            if line:
                tsv.append(_TAB.split(line))
            elif tsv:
                break

        return self.tsv_to_graph(tsv) if tsv else None

    @property
    def next_all(self):
        """

        :return: List of NLPGraph
        :rtype: List[NLPGraph]
        """
        return [graph for graph in self]

    def tsv_to_graph(self, tsv):
        """

        :param tsv: each row represents a token, each column represents a field.
        :type tsv: List[List[str]]
        """

        def get_field(row, index):
            """

            :param row:
            :type row: List[str]
            :param index:
            :type index: int
            :return: ROOT_TAG
            :rtype: str
            """
            return ROOT_TAG if row is None else row[index] if index >= 0 else None

        def get_feats(row):
            """

            :param row:
            :type row: List[str]
            :return: feats
            :rtype: Union[Dict[str, str], None]
            """
            if self.feats_index >= 0:
                feats = row[self.feats_index]
                if feats == BLANK:
                    return None
                return {feat[0]: feat[1] for feat in map(_FEATS_KV.split, _FEATS.split(feats))}
            return None

        def init_node(i):
            """

            :param i:
            :type i: int
            :return: NLPNode
            :rtype: NLPToken
            """
            row = tsv[i]
            token_id = i + 1
            form = get_field(row, self.form_index)
            lemma = get_field(row, self.lemma_index)
            pos = get_field(row, self.pos_index)
            nament = get_field(row, self.nament_index)
            feats = get_feats(row) if row else None
            return NLPToken(token_id=token_id,
                            form=form,
                            lemma=lemma,
                            pos=pos,
                            nament=nament,
                            feats=feats)

        graph = NLPGraph([init_node(i) for i in range(len(tsv))])

        if self.head_index >= 0:
            for i, node in enumerate(graph):
                _tsv = tsv[i]
                node.set_parent(graph.nodes[int(_tsv[self.head_index])], _tsv[self.deprel_index])

                if self.sheads_index >= 0 and _tsv[self.sheads_index] != BLANK:
                    for arc in map(_ARC_KV.split, _ARC.split(_tsv[self.sheads_index])):
                        node.add_secondary_parent(graph.nodes[int(arc[0])], arc[1])

        return graph
