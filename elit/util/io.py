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
import sys

import bisect
import glob
import json
import logging
import re
from typing import List, Dict, Sequence, Any, Tuple, Set

import codecs
import os

from elit.dataset import LabelMap
from elit.structure import Sentence, TOK, Document, to_gold, SID, DOC_ID

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
        [sent.update({SID: i}) for i, sent in enumerate(sentences)]
        sc += len(sentences)
        documents.append(Document(sens=sentences))

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
            for sen in doc['sens']:
                wc += len(sen['tok'])
                if key is not None:
                    sen = sen.copy()
                    sen[to_gold(key)] = sen.pop(key)
                sentences.append(Sentence(sen))
            sc += len(sentences)
            [[label_map.add(i) for i in sent[to_gold(key)]] for sent in sentences]
            document = Document(sens=sentences)
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


def bucket_sentences(data: Sequence[Document], maxlen: int = -1) -> List[Document]:
    """
    :param data: a list of documents or a list of sentences.
    :param maxlen: the maximum length of each sentence; if max_len < 0, it is inferred by the length of the longest sentence.
    :return: the list of documents, where the number of tokens across all sentences in each document is roughly ``maxlen``.
    """

    def aux(i):
        ls = d[keys[i]]
        t = ls.pop()
        document.add_sentence(t)
        if not ls:
            del keys[i]
        return len(t)

    # key = length, value = list of sentences with the key length
    d = {}

    for doc in data:
        for sen in doc.sentences:
            d.setdefault(len(sen), []).append(sen)

    keys = sorted(list(d.keys()))
    if maxlen < 0:
        maxlen = keys[-1]

    documents = []
    document = Document()
    wc = maxlen - aux(-1)

    while keys:
        idx = bisect.bisect_left(keys, wc)
        if idx >= len(keys) or keys[idx] > wc:
            idx -= 1
        if idx < 0:
            documents.append(document)
            document = Document()
            wc = maxlen - aux(-1)
        else:
            wc -= aux(idx)

    if document:
        documents.append(document)
    return documents


def read_word_set(filename) -> Set[str]:
    """
    :param filename: the name of the file containing one key per line.
    :return: a set containing all keys in the file.
    """
    fin = codecs.open(filename, mode='r', encoding='utf-8')
    s = set(line.strip() for line in fin)
    logging.info('Init: %s (keys = %d)' % (filename, len(s)))
    return s


def read_word_dict(filename: str) -> Dict[str, str]:
    """
    :param filename: the name of the file containing one 'key value' pair per line.
    :return: a dictionary containing all key-value pairs in the file.
    """
    fin = codecs.open(filename, mode='r', encoding='utf-8')
    d = dict(line.strip().split() for line in fin)
    logging.info('Init: %s (keys = %d)' % (filename, len(d)))
    return d


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


class NoIndent(object):
    def __init__(self, value):
        self.value = value


class NoIndentEncoder(json.JSONEncoder):
    REGEX = re.compile(r'@@@(\d+)@@@')

    def __init__(self, *args, **kwargs):
        super(NoIndentEncoder, self).__init__(*args, **kwargs)
        self.kwargs = dict(kwargs)
        del self.kwargs['indent']
        self._replacements = {}

    def default(self, o):
        if isinstance(o, NoIndent):
            key = len(self._replacements)
            self._replacements[key] = json.dumps(o.value, **self.kwargs)
            return "@@@%d@@@" % key
        else:
            return super(NoIndentEncoder, self).default(o)

    def encode(self, o):
        result = super(NoIndentEncoder, self).encode(o)
        out = []
        m = self.REGEX.search(result)
        while m:
            key = int(m.group(1))
            out.append(result[:m.start(0) - 1])
            out.append(self._replacements[key])
            result = result[m.end(0) + 1:]
            m = self.REGEX.search(result)
        return ''.join(out)


def set_logger(filename: str = None,
               level: int = logging.INFO,
               formatter: logging.Formatter = None):
    log = logging.getLogger()
    log.setLevel(level)
    ch = logging.StreamHandler(sys.stdout) if filename is None else logging.FileHandler(filename)
    if formatter is not None:
        ch.setFormatter(formatter)
    log.addHandler(ch)
