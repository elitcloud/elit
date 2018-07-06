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

import numpy as np
from elit.nlp.structure import TOKEN

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
        return [X_FST if i == 0 else X_LST if i + 1 == size else X_ANY for i in range(size)]

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


