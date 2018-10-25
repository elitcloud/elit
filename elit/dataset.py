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
import abc

import numpy as np
from typing import List, Union

from mxnet import nd, gluon
from mxnet.ndarray import NDArray
from tqdm import tqdm
from types import SimpleNamespace

from elit.structure import to_gold, DOC_ID, Document

__author__ = "Gary Lai"


class LabelMap(object):
    """
    :class:`LabelMap` provides the mapping between string labels and their unique integer IDs.
    """

    def __init__(self):
        self.index_map = {}
        self.labels = []

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
            self.labels.append(label)
        return idx

    def get(self, cid: int) -> str:
        """
        :param cid: the class ID.
        :return: the label corresponding to the class ID.
        """
        return self.labels[cid]

    def cid(self, label: str) -> int:
        """
        :param label: the label.
        :return: the class ID of the label if exists; otherwise, -1.
        """
        return self.index_map.get(label, -1)

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

    def __init__(self, vsms: List[SimpleNamespace], key: str, docs: List[Document], label_map: LabelMap, transform=None):
        self._data = []
        self.vsms = vsms
        self.key = key
        self.label_map = label_map
        self.transform = transform
        self.init_data(docs)

    def __getitem__(self, idx):
        x, y = self._data[idx]
        if self.transform is not None:
            return self.transform(x, y)
        else:
            return x, y

    def __len__(self):
        return len(self._data)

    @abc.abstractmethod
    def extract(self, sen):
        raise NotImplementedError

    def extract_sen(self, sen):
        return nd.array([np.concatenate(i) for i in zip(*[vsm.model.embedding_list(sen.tokens) for vsm in self.vsms])]).reshape(0, -1)

    def init_data(self, docs: List[Document]):
        for doc in tqdm(docs):
            for sen in tqdm(doc, desc="loading doc: {}".format(doc[DOC_ID]), leave=False):
                self.extract(sen)


class TokensDataset(Dataset):

    def __init__(self, vsms: List[SimpleNamespace], key: str, docs: List[Document], label_map: LabelMap, feature_windows: List, transform=None):
        super().__init__(vsms, key, docs, label_map, transform)
        self.feature_windows = feature_windows
        self.pad = nd.zeros(sum([vsm.model.dim for vsm in self.vsms]))

    def extract_x(self, idx, w):
        return nd.stack(*[w[idx + win] if 0 <= (idx + win) < len(w) else self.pad for win in self.feature_windows])

    def extract_y(self, label: str):
        return self.label_map.cid(label)

    def extract(self, sen, **kwargs):
        w = self.extract_sen(sen)
        for idx, label in enumerate(sen[to_gold(self.key)]):
            self._data.append((self.extract_x(idx, w), self.extract_y(label)))


class SequencesDataset(Dataset):

    def __init__(self, vsms: List[SimpleNamespace], key: str, docs: List[Document], label_map: LabelMap):
        super().__init__(vsms, key, docs, label_map)

    def extract_labels(self, sen):
        return nd.array([self.label_map.cid(label) for label in sen[to_gold(self.key)]])

    def extract(self, sen):
        self._data.append((self.extract_sen(sen), self.extract_labels(sen)))
