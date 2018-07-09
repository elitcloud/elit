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

__author__ = 'Jinho D. Choi'

SENTENCE_ID = 'sid'

TOKEN = 'tok'
LEMMA = 'lemma'
OFFSET = 'offset'

POS = 'pos'
NER = 'ner'
DEPREL = 'deprel'
COREF = 'coref'
SENTIMENT = 'sentiment'

OUT = '-out'


class Sentence(dict):
    def __init__(self, d=None):
        """
        :param d: a dictionary containing fields for the sentence.
        :type d: dict
        """
        super().__init__()
        if d is not None:
            self.update(d)

    def __len__(self):
        return len(self[TOKEN])


class Document(list):
    def __init__(self, l=None):
        """
        :param l: a list containing sentences.
        :type l: list of Sentence
        """
        super().__init__()
        if l is not None:
            self.extend(l)


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
    for s in sentences:
        d.setdefault(len(s), []).append(s)
    keys = sorted(list(d.keys()))
    if max_len < 0:
        max_len = keys[-1]

    states = []
    document = Document()
    wc = max_len - aux(-1)
    if create_state is None:
        create_state = dummy

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

    if document:
        states.append(create_state(document))
    return states
