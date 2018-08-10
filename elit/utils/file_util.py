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
import json
from typing import List, Dict

from elit.structure import Sentence, TOK, Document
from elit.util import to_gold

__author__ = "Jinho D. Choi, Gary Lai"


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

    print('Reading: dc = %d, sc = %d, wc = %d' % (len(documents), sc, wc))
    return documents


def json_reader(filepath: str) -> List[Document]:
    # TODO: to be filled
    documents = []
    dc = wc = sc = 0

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
    print('Reading: dc = %d, sc = %d, wc = %d' % (dc, sc, wc))
    return documents


def pkl(filepath):
    return filepath + '.pkl'


def gln(filepath):
    return filepath + '.gln'
