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
import abc

__author__ = 'Jinho D. Choi'


class Metric(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def reset(self):
        """
        Resets all counts to 0.
        """
        pass

    @abc.abstractmethod
    def get(self):
        """
        :return: the evaluated score.
        """
        return


class Accuracy(Metric):
    def __init__(self):
        self.correct = 0
        self.total = 0

    def reset(self):
        self.correct = 0
        self.total = 0

    def get(self):
        """
        :rtype: float
        """
        return 100.0 * self.correct / self.total


class F1(Metric):
    def __init__(self):
        self.correct = 0
        self.p_total = 0
        self.r_total = 0

    def reset(self):
        self.correct = 0
        self.p_total = 0
        self.r_total = 0

    def get(self):
        """
        :return: (F1 score, prediction, recall)
        :rtype: (float, float, float)
        """
        p = 100.0 * self.correct / self.p_total
        r = 100.0 * self.correct / self.r_total
        f1 = 2 * p * r / (p + r)
        return f1, p, r
