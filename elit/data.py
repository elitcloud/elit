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
from types import SimpleNamespace
from typing import List

import numpy as np
from gluonnlp.data import batchify, FixedBucketSampler
from mxnet import nd
from mxnet.gluon.data import Dataset, DataLoader
from tqdm import tqdm

from elit.util.io import tsv_reader
from elit.util.structure import Document, TOK, POS, DOC_ID, to_gold
from elit.util.vsm import LabelMap, init_vsm

__author__ = "Gary Lai"

batchify_fn = batchify.Tuple(batchify.Pad(), batchify.Pad())


class SentDataset(Dataset):

    def __init__(self, vsms: List[SimpleNamespace],
                 key,
                 docs,
                 label_map: LabelMap,
                 label: bool = True,
                 display: bool = False):
        self.vsms = vsms
        self.key = key
        self.label_map = label_map
        self.label = label
        self._data = []
        self.display = display
        self.init_data(docs)

    def __getitem__(self, idx):
        return self._data[idx]

    def __len__(self):
        return len(self._data)

    def extract_sen(self, sen):
        return nd.array(
            [np.concatenate(i) for i in zip(*[vsm.model.embedding_list(sen.tokens) for vsm in self.vsms])]) \
            .reshape(0, -1)

    def extract_labels(self, sen):
        return nd.array([self.label_map.cid(label) for label in sen[to_gold(self.key)]])

    def init_data(self, docs: [Document]):
        for doc in tqdm(docs, disable=self.display):
            for sen in tqdm(doc, desc="loading doc: {}".format(doc[DOC_ID]), leave=False, disable=self.display):
                self._data.append((self.extract_sen(sen), self.extract_labels(sen)))


if __name__ == '__main__':
    vsm_path = [['fasttext', '/home/glai2/Documents/elit/sample-fasttext.bin']]
    vsms = [init_vsm(n) for n in vsm_path]
    docs, lm = tsv_reader(tsv_directory='/home/glai2/Documents/wsj-pos/trn', cols={TOK: 0, POS: 1}, key=POS)
    # docs, lm = tsv_reader(tsv_directory='/Users/gary/Documents/research/elit/elit/tests/resources/tsv/pos/trn', cols={TOK: 0, POS: 1}, key='pos')

    dataset = SentDataset(docs=docs, label_map=lm, vsms=vsms, key='pos')
    dataset_lengths = list(map(lambda x: float(len(x[1])), dataset))
    batch_sampler = FixedBucketSampler(dataset_lengths,
                                       batch_size=64,
                                       num_buckets=10,
                                       ratio=0.5,
                                       shuffle=True)
    dataloader = DataLoader(dataset=dataset,
                            batch_sampler=batch_sampler,
                            batchify_fn=batchify_fn)

    print(batch_sampler.stats())

    for data, label in dataloader:
        print('data: {} \nshape: {}'.format(data, data.shape))
        print('label: {} \nshape: {}'.format(label, label.shape))
        break
