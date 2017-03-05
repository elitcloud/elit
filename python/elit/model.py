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
from abc import ABCMeta
from abc import abstractmethod
from typing import Dict
from typing import List
import numpy as np
from elit.component import NLPState

__author__ = 'Jinho D. Choi'


class NLPModel(metaclass=ABCMeta):
    def __init__(self):
        self.index_map: Dict[str, int] = {}
        self.labels: List[str] = []

    # ============================== Label ==============================

    def get_label_index(self, label: str) -> int:
        """
        :return: the index of the label.
        """
        return self.index_map.get(label, -1)

    def get_label(self, index: int) -> str:
        """
        :param index: the index of the label to be returned.
        :return: the index'th label.
        """
        return self.labels[index]

    def add_label(self, label: str) -> int:
        """
        :return: the index of the label.
          Add a label to this map if not exist already.
        """
        idx = self.get_label_index(label)
        if idx < 0:
            idx = len(self.labels)
            self.index_map[label] = idx
            self.labels.append(label)
        return idx

    # ============================== Predict ==============================

    @abstractmethod
    def create_feature_vector(self, state: NLPState) -> np.array:
        """
        :param state: the current processing state.
        :return: the feature vector representing the current state.
        """

    @abstractmethod
    def predict(self, x: np.array) -> int:
        """
        :param x: the feature vector.
        :return: the ID of the best predicted label.
        """

    def predict_label(self, x: np.array) -> str:
        """
        :param x: the feature vector.
        :return: the best predicted label.
        """
        return self.labels[self.predict(x)]



#mxnet.module.module.Module

