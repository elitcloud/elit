# ========================================================================
# Copyright 2018 Emory University
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
import glob
import logging

from elit.util import DEPREL, TOK, Sentence, group_states

__author__ = "Gary Lai"


def tsv(filepath):
    return filepath + '.tsv'


def read_tsv(filepath, cols, create_state=None):
    """
    Reads data from TSV files specified by the filepath.
    :param filepath: the path to a file (e.g., train.tsv) or multiple files (e.g., folder/*.tsv).
    :type filepath: str
    :param cols: a dictionary containing the column index of each field.
    :type cols: dict
    :param create_state: a function that takes a document and returns a state.
    :type create_state: Document -> elit.nlp.component.NLPState
    :return: a list of states containing documents, where each document is a list of sentences.
    :rtype: list of elit.nlp.component.NLPState
    """

    def create_dict():
        return {k: [] for k in cols.keys()}

    def aux(filename):
        fin = open(filename)
        d = create_dict()
        wc = 0

        for line in fin:
            l = line.split()
            if l:
                for k, v in cols.items():
                    # (head ID, deprel)
                    f = (int(l[v[0]]) - 1, l[v[1]]) if k == DEPREL else l[v]
                    d[k].append(f)
            elif d[TOK]:
                sentences.append(Sentence(d))
                wc += len(sentences[-1])
                d = create_dict()

        return wc

    sentences = []
    word_count = 0
    for file in glob.glob(filepath):
        word_count += aux(file)
    states = group_states(sentences, create_state)
    logging.info('Read: %s (sc = %d, wc = %d, grp = %d)' % (
        filepath, len(sentences), word_count, len(states)))
    return states
