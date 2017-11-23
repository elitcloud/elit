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

TOKENS = 'tokens'
OFFSETS = 'offsets'
SENTIMENT = 'sentiment'

# part-of-speech tagging
POS = 'pos'

# named entity recognition
NER = 'ner'


class Sentence(dict):
    def __init__(self, d=None):
        """
        :param d: a dictionary containing fields for the sentence.
        :type d: dict
        """
        super(Sentence, self).__init__()
        if d is not None: self.update(d)

    def __len__(self):
        return len(self[TOKENS])


def group_sentences(sentences, max_len=-1):
    """
    :param sentences: list of sentences.
    :type sentences: list of elit.util.structure.Sentence
    :param max_len: the maximum number of words in each document to be returned.
                    If max_len < 0, it is inferred by the longest sentence.
    :type max_len: int
    :return: list of documents, where each document is a list of sentences that contain the max_len number of words.
    :rtype: list of list of elit.util.structure.Sentence
    """
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

    document = []
    document_list = []
    wc = max_len - aux(-1)

    while keys:
        idx = bisect.bisect_left(keys, wc)
        if idx >= len(keys) or keys[idx] > wc:
            idx -= 1
        if idx < 0:
            document_list.append(document)
            document = []
            wc = max_len - aux(-1)
        else:
            wc -= aux(idx)

    if document: document_list.append(document)
    return document_list

