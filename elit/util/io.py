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
import codecs
import glob
import json
import logging
import os
from typing import List, Dict, Callable, Sequence, Any, Tuple

from elit.state import NLPState
from elit.util.structure import Sentence, TOK, Document, to_gold, SEN_ID, DOC_ID
from elit.util.vsm import LabelMap

__author__ = "Jinho D. Choi, Gary Lai"


def tsv_cols(tsv_heads: list) -> Dict[str, int]:
    d = {}
    for head in tsv_heads:
        d[head[0]] = head[1]
    return d


# ======================================== Readers =======================


def tsv_reader(tsv_directory: str,
               cols: Dict[str, int],
               key: str = None) -> Tuple[List[Document], LabelMap]:
    documents = []
    wc = sc = 0
    label_map = LabelMap()

    if TOK not in cols:
        raise ValueError('The column index of "%s" must be specified' % TOK)

    if key is not None:
        if key in cols:
            cols = cols.copy()
            cols[to_gold(key)] = cols.pop(key)
        else:
            raise ValueError('Key mismatch: %s is not a key in %s' % (key, str(cols)))

    logging.info('Reading tsv from:')
    logging.info('- directory: %s' % tsv_directory)

    for filename in glob.glob('{}/*.tsv'.format(tsv_directory)):
        # avoid reading unexpected files, such as hidden files.
        if not os.path.isfile(filename):
            continue
        logging.info('  - file: %s' % filename)

        sentences = []
        sid = 0
        fields = {k: [] for k in cols.keys()}

        fin = open(filename)
        for line in fin:
            if line.startswith('#'):
                continue
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

        [[label_map.add(i) for i in sent[to_gold(key)]] for sent in sentences]
        [sent.update({SEN_ID: i}) for i, sent in enumerate(sentences)]
        sc += len(sentences)
        documents.append(Document(sen=sentences))

    [sent.update({DOC_ID: i}) for i, sent in enumerate(documents)]
    logging.info('- dc = %d, sc = %d, wc = %d' % (len(documents), sc, wc))
    return documents, label_map


def json_reader(filepath: str,
                cols: Any = None,
                key: str = None) -> Tuple[List[Document], LabelMap]:
    # TODO: update this to accept any format (see tsv_reader)
    documents = []
    dc = wc = sc = 0
    label_map = LabelMap()

    logging.info('Reading json file from: ')
    if not os.path.isfile(filepath) and filepath.endswith('.json'):
        raise ValueError("{} is not a valid format".format(filepath))
    logging.info('- filepath: %s' % filepath)

    with open(filepath) as f:
        docs = json.load(f)
        for i, doc in enumerate(docs):
            sentences = []
            for sen in doc['sen']:
                wc += len(sen['tok'])
                if key is not None:
                    sen = sen.copy()
                    sen[to_gold(key)] = sen.pop(key)
                sentences.append(Sentence(sen))
            sc += len(sentences)
            [[label_map.add(i) for i in sent[to_gold(key)]] for sent in sentences]
            document = Document(sen=sentences)
            document[DOC_ID] = i
            documents.append(document)
            dc += len(documents)
    logging.info('- dc = %d, sc = %d, wc = %d' % (dc, sc, wc))
    return documents, label_map


def pkl(filepath):
    return filepath + '.pkl'


def gln(filepath):
    return filepath + '.gln'


def params(filepath):
    return filepath + '.params'


def group_states(docs: Sequence[Document], create_state: Callable[[
                                                                      Document], NLPState], maxlen: int = -1) -> List[
    NLPState]:
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
        if not ls:
            del keys[i]
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


def read_word_set(filename):
    """
    :param filename: the name of the file containing one key per line.
    :return: a set containing all keys in the file.
    """
    fin = codecs.open(filename, mode='r', encoding='utf-8')
    s = set(line.strip() for line in fin)
    logging.info('Init: %s (keys = %d)' % (filename, len(s)))
    return s


def read_concat_word_dict(filename):
    """
    :param filename: the name of the file containing one key per line.
    :return: a dictionary whose key is the concatenated word and value is the list of split points.
    """

    def key_value(line):
        l = [i for i, c in enumerate(line) if c == ' ']
        l = [i - o for o, i in enumerate(l)]
        line = line.replace(' ', '')
        l.append(len(line))
        return line, l

    fin = codecs.open(filename, mode='r', encoding='utf-8')
    d = dict(key_value(line.strip()) for line in fin)
    logging.info('Init: %s (keys = %d)' % (filename, len(d)))
    return d
