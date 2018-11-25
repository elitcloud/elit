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
from typing import Tuple

import abc
from mxnet import metric

from elit.structure import BILOU

__author__ = 'Jinho D. Choi'


class EvalMetric(metric.EvalMetric):
  
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

    @abc.abstractmethod
    def update(self, labels, preds):
        pass


class F1(EvalMetric):
  
    def __init__(self):
        super().__init__(name='f1score')
        self.correct = 0.0
        self.p_total = 0.0
        self.r_total = 0.0

    def precision(self) -> float:
        return self.correct / self.p_total if self.p_total > 0.0 else 0.0

    def recall(self) -> float:
        return self.correct / self.r_total if self.r_total > 0.0 else 0.0

    def f1(self) -> float:
        p = self.precision()
        r = self.recall()
        return 2 * p * r / (p + r) if p + r > 0.0 else 0.0

    def reset(self):
        self.correct = 0.0
        self.p_total = 0.0
        self.r_total = 0.0

    def get(self) -> Tuple[str, float]:
        return self.name, self.f1()

    @abc.abstractmethod
    def update(self, labels, preds):
        pass


class ChunkF1(F1):

    def update(self, labels, preds):
        gold = BILOU.to_chunks(labels)
        pred = BILOU.to_chunks(preds)
        self.correct += len(set.intersection(set(gold), set(pred)))
        self.p_total += len(pred)
        self.r_total += len(gold)
