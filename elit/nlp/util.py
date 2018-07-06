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
import bisect
import glob
import logging
import time
from random import shuffle

import numpy as np
from mxnet import gluon, autograd

from elit.nlp.structure import DEPREL, TOKEN, Sentence, Document

__author__ = 'Jinho D. Choi'


X_FST = np.array([1, 0]).astype('float32')  # the first word
X_LST = np.array([0, 1]).astype('float32')  # the last word
X_ANY = np.array([0, 0]).astype('float32')  # any other word


def get_loc_embeddings(document):
    """
    :return: the position embedding of the (self.tok_id + window)'th word.
    :rtype: numpy.array
    """
    def aux(sentence):
        size = len(sentence)
        return [X_FST if i == 0 else X_LST if i+1 == size else X_ANY for i in range(size)]

    return [aux(s) for s in document], X_ANY


def get_embeddings(vsm, document, key=TOKEN):
    """
    :param vsm: a vector space model.
    :type vsm: elit.nlp.lexicon.VectorSpaceModel
    :param document: a document.
    :type document: elit.nlp.structure.Document
    :param key: the key to each sentence.
    :type key: str
    :return:
    """
    return [vsm.get_list(s[key]) for s in document], vsm.zero


def x_extract(tok_id, window, size, emb, zero):
    """
    :param window: the context window.
    :type window: int
    :param emb: the list of embeddings.
    :type emb: numpy.array
    :param zero: the vector for zero-padding.
    :type zero: numpy.array
    :return: the (self.tok_id + window)'th embedding if exists; otherwise, the zero-padded embedding.
    """
    i = tok_id + window
    return emb[i] if 0 <= i < size else zero




def group_states(sentences, create_state=None, max_len=-1):
    """
    Groups sentences into documents such that each document consists of multiple sentences and the total number of words
    across all sentences within a document is close to the specified maximum length.
    :param sentences: list of sentences.
    :type sentences: list of elit.util.structure.Sentence
    :param create_state: a function that takes a document and returns a state.
    :type create_state: Document -> elit.nlp.component.NLPState
    :param max_len: the maximum number of words; if max_len < 0, it is inferred by the length of the longest sentence.
    :type max_len: int
    :return: list of states, where each state roughly consists of the max_len number of words.
    :rtype: list of elit.nlp.NLPState
    """
    def dummy(doc):
        return doc

    def aux(i):
        ls = d[keys[i]]
        t = ls.pop()
        document.append(t)
        if not ls: del keys[i]
        return len(t)

    # key = length, value = list of sentences with the key length
    d = {}
    for s in sentences: d.setdefault(len(s), []).append(s)
    keys = sorted(list(d.keys()))
    if max_len < 0: max_len = keys[-1]

    states = []
    document = Document()
    wc = max_len - aux(-1)
    if create_state is None: create_state = dummy

    while keys:
        idx = bisect.bisect_left(keys, wc)
        if idx >= len(keys) or keys[idx] > wc:
            idx -= 1
        if idx < 0:
            states.append(create_state(document))
            document = Document()
            wc = max_len - aux(-1)
        else:
            wc -= aux(idx)

    if document: states.append(create_state(document))
    return states