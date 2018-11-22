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
from typing import List, Union, Sequence, Tuple

import abc
import numpy as np
from gluonnlp.data.batchify import batchify
from mxnet import nd, gluon
from mxnet.ndarray import NDArray
from tqdm import tqdm

from elit.nlp.embedding import Embedding, TokenEmbedding
from elit.structure import to_gold, Sentence, Document

__author__ = "Gary Lai"


class LabelMap(object):
    """
    :class:`LabelMap` provides the mapping between string labels and their unique integer IDs.
    """

    def __init__(self):
        self.index_map = {}
        self.labels = {}

    def __len__(self):
        return len(self.labels)

    def __str__(self):
        return str(self.index_map)

    def add(self, label: str) -> int:
        """
        :param label: the label.
        :return: the class ID of the label.

        Adds the label to this map and assigns its class ID if not already exists.
        """
        idx = self.cid(label)
        if idx < 0:
            idx = len(self.labels)
            self.index_map[label] = idx
            self.labels[idx] = label
        return idx

    def get(self, cid: int) -> str:
        """
        :param cid: the class ID.
        :return: the label corresponding to the class ID.
        """
        return self.labels.get(cid, None)

    def cid(self, label: str) -> int:
        """
        :param label: the label.
        :return: the class ID of the label if exists; otherwise, -1.
        """
        return self.index_map.get(label, -1)

    def num_class(self) -> int:
        return len(self.labels)

    def argmax(self, scores: Union[np.ndarray, NDArray]) -> str:
        """
        :param scores: the prediction scores of all labels.
        :return: the label with the maximum score.
        """
        if self.__len__() < len(scores):
            scores = scores[:self.__len__()]
        n = nd.argmax(scores, axis=0).asscalar() if isinstance(scores, NDArray) else np.argmax(scores)
        return self.get(int(n))


class Dataset(gluon.data.Dataset):

    def __init__(self, docs: Sequence[Document], embs: List[Union[Embedding, TokenEmbedding]], key: str, label_map: LabelMap, label: bool = True, transform=None):
        self._data = []
        self.embs = embs
        self.key = key
        self.label_map = label_map
        self.label = label
        self.transform = transform
        self.docs = docs
        self.init_data()

    def __getitem__(self, idx):
        if self.transform is not None:
            return self.transform(self._data[idx])
        else:
            return self._data[idx]

    def __len__(self):
        return len(self._data)

    @abc.abstractmethod
    def extract(self, did: int, sid: int, sen: Sentence):
        raise NotImplementedError

    def extract_sen(self, sen):
        return nd.array([np.concatenate(i) for i in zip(*[emb.emb_list(sen.tokens) for emb in self.embs])]).reshape(0, -1)

    def init_data(self):
        for did, doc in enumerate(tqdm(self.docs, leave=False)):
            for sid, sen in enumerate(tqdm(doc.sentences, leave=False)):
                self.extract(did, sid, sen)


class TokensDataset(Dataset):

    def __init__(self, docs: Sequence[Document], embs: List[Embedding], key: str, label_map: LabelMap, feature_windows: Tuple, label: bool = True, transform=None):
        self.embs = embs
        self.feature_windows = feature_windows
        self.pad = nd.zeros(sum([emb.dim for emb in self.embs]))
        super().__init__(docs=docs, embs=embs, key=key, label_map=label_map, label=label, transform=transform)

    def extract_x(self, i, w):
        return nd.stack(*[w[i + win] if 0 <= (i + win) < len(w) else self.pad for win in self.feature_windows])

    def extract_y(self, label: str):
        return self.label_map.cid(label)

    def extract(self, did, sid, sen, **kwargs):
        w = self.extract_sen(sen)
        if self.label:
            for i, label in enumerate(sen[to_gold(self.key)]):
                self._data.append((self.extract_x(i, w), self.extract_y(label)))
        else:
            for i in range(len(sen)):
                self._data.append((self.extract_x(i, w), -1))


class SequencesDataset(Dataset):

    def __init__(self, docs: Sequence[Document], embs: List[Embedding], key: str, label_map: LabelMap, label: bool = True, transform=None):
        super().__init__(docs=docs, embs=embs, key=key, label_map=label_map, label=label, transform=transform)

    def extract_labels(self, sen):
        if self.label:
            return nd.array([self.label_map.cid(l) for l in sen[to_gold(self.key)]])
        else:
            return nd.array([-1 for _ in range(len(sen))])

    def extract(self, did, sid, sen):
        self._data.append((did, sid, self.extract_sen(sen), self.extract_labels(sen)))


sequence_batchify_fn = batchify.Tuple((batchify.Stack(), batchify.Stack(), batchify.Pad(), batchify.Pad(pad_val=-1)))
# stack doc idx
# stack sen idx
# pad sen
# pad label
