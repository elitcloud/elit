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
from typing import Any, Tuple

from elit.structure import Document

__author__ = 'Jinho D. Choi'


class EvalMetric(abc.ABC):
    @abc.abstractmethod
    def update(self, document: Document):
        """
        Updates this evaluation metric given
        """
        pass

    @abc.abstractmethod
    def get(self) -> Any:
        """
        :return: the evaluated score.
        """
        pass


class Accuracy(EvalMetric):
    def __init__(self):
        self.correct = 0
        self.total = 0

    @abc.abstractmethod
    def update(self, document: Document):
        pass

    def get(self) -> float:
        """
        :return: accuracy in percentage.
        """
        return 100.0 * self.correct / self.total


class F1(EvalMetric):
    def __init__(self):
        super().__init__()
        self.correct = 0
        self.p_total = 0
        self.r_total = 0

    @abc.abstractmethod
    def update(self, document: Document):
        pass

    def precision(self) -> float:
        return 100.0 * self.correct / self.p_total if self.p_total > 0 else 0

    def recall(self) -> float:
        return 100.0 * self.correct / self.r_total if self.r_total > 0 else 0

    def f1(self):
        p = self.precision()
        r = self.recall()
        return 2 * p * r / (p + r) if p + r > 0 else 0

    def get(self) -> float:
        """
        :return: (F1 score, prediction, recall)
        """
        return self.f1()


