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
import bisect
import glob
import json
import logging
from typing import List, Dict, Callable, Sequence

from elit.state import NLPState
from elit.util.structure import Sentence, TOK, Document, to_gold

__author__ = "Jinho D. Choi, Gary Lai"


# ======================================== Readers ========================================

def tsv_reader(filepath: str, cols: Dict[str, int], key: str = None) -> List[Document]:
    documents = []
    wc = sc = 0

    if TOK not in cols:
        raise ValueError('The column index of "%s" must be specified' % TOK)

    if key is not None:
        if key in cols:
            cols = cols.copy()
            cols[to_gold(key)] = cols.pop(key)
        else:
            raise ValueError('Key mismatch: %s is not a key in %s' % (key, str(cols)))

    logging.info('Reading')
    logging.info('- filepath: %s' % filepath)

    for filename in glob.glob(filepath):
        fin = open(filename)
        sentences = []
        fields = {k: [] for k in cols.keys()}

        for line in fin:
            if line.startswith('#'): continue
            l = line.split()

            if l:
                for k, v in fields.items():
                    v.append(l[cols[k]])
            elif len(fields[TOK]) > 0:
                wc += len(fields[TOK])
                sentences.append(Sentence(fields))
                fields = {k: [] for k in cols.keys()}

        if len(fields[TOK]) > 0:
            wc += len(fields[TOK])
            sentences.append(Sentence(fields))

        fin.close()
        sc += len(sentences)
        documents.append(Document(sen=sentences))

    logging.info('- dc = %d, sc = %d, wc = %d' % (len(documents), sc, wc))
    return documents


def json_reader(filepath: str) -> List[Document]:
    # TODO: update this to accept any format (see tsv_reader)
    documents = []
    dc = wc = sc = 0

    logging.info('Reading')
    logging.info('- filepath: %s' % filepath)

    for filename in glob.glob('{}/*.json'.format(filepath)):
        assert filename.endswith('.json')
        with open(filename) as f:
            docs = json.load(f)
            for doc in docs:
                sentences = []
                for sen in doc['sen']:
                    wc += len(sen['tok'])
                    sentences.append(Sentence(sen))
                sc += len(sentences)
                documents.append(Document(sen=sentences))
            dc += len(documents)
    logging.info('- dc = %d, sc = %d, wc = %d' % (dc, sc, wc))
    return documents


def pkl(filepath):
    return filepath + '.pkl'


def gln(filepath):
    return filepath + '.gln'


def group_states(docs: Sequence[Document], create_state: Callable[[Document], NLPState], maxlen: int = -1) -> List[NLPState]:
    """
    Groups sentences into documents such that each document consists of multiple sentences and the total number of words
    across all sentences within a document is close to the specified maximum length.
    :param docs: a list of documents.
    :param create_state: a function that takes a document and returns a state.
    :param maxlen: the maximum number of words; if max_len < 0, it is inferred by the length of the longest sentence.
    :return: list of states, where each state roughly consists of the max_len number of words.
    """

    def aux(i):
        ls = d[keys[i]]
        t = ls.pop()
        document.sentences.append(t)
        if not ls: del keys[i]
        return len(t)

    # key = length, value = list of sentences with the key length
    d = {}

    for doc in docs:
        for sen in doc.sentences:
            d.setdefault(len(sen), []).append(sen)

    keys = sorted(list(d.keys()))
    if maxlen < 0:
        maxlen = keys[-1]

    states = []
    document = Document()
    wc = maxlen - aux(-1)

    while keys:
        idx = bisect.bisect_left(keys, wc)
        if idx >= len(keys) or keys[idx] > wc:
            idx -= 1
        if idx < 0:
            states.append(create_state(document))
            document = Document()
            wc = maxlen - aux(-1)
        else:
            wc -= aux(idx)

    if document:
        states.append(create_state(document))

    return states
