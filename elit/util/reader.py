# ========================================================================
# Copyright 2018 ELIT
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
import logging
import os
from typing import List, Dict, Any, Tuple

from elit.dataset import LabelMap
from elit.structure import Sentence, TOK, Document, to_gold, SID, DOC_ID

__author__ = "Gary Lai"


def tsv_cols(tsv_heads: list) -> Dict[str, int]:
    d = {}
    for head in tsv_heads:
        d[head[0]] = head[1]
    return d


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
