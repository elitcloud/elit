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
from typing import List

from elit.component import NLPComponent
from elit.eval import Accuracy, F1
from elit.nlp.doc_classifier import SentenceClassificationBatchState
from elit.structure import SENTI_NEUTRAL, SENTI_POSITIVE, SENTI_NEGATIVE

__author__ = 'Jinho D. Choi'


class SentimentState(SentenceClassificationBatchState):
    def eval(self, metric: List[Accuracy, Accuracy, F1, F1, F1]):
        eval_sentiment(metric, self.gold, [s[self.key] for s in self.document])


class SentimentAnalyzer(NLPComponent):
    def __init__(self, ctx, vsm):
        """
        :param ctx:
        :type ctx: mx.
        :param vsm:
        """
        super().__init__(ctx)
        self.vsm = vsm

        # to be initialized
        self.label_map = None


def eval_sentiment(metric: List[Accuracy,
                                Accuracy,
                                F1,
                                F1,
                                F1],
                   gold_labels: List[str],
                   pred_labels: List[str]):
    for i, gold in enumerate(gold_labels):
        pred = pred_labels[i]

        # accuracy: multi-class
        m: Accuracy = metric[0]
        m.total += 1
        if gold == pred:
            m.correct += 1

        # accuracy: binary
        if gold != SENTI_NEUTRAL:
            m: Accuracy = metric[1]
            m.total += 1
            if gold[0] == pred[0]:
                m.correct += 1

        # f1: positive, negative, neutral
        m: F1 = metric[2] if gold[0] == SENTI_POSITIVE else metric[3] if gold[0] == SENTI_NEGATIVE else metric[4]
        m.r_total += 1

        m: F1 = metric[2] if pred[0] == SENTI_POSITIVE else metric[3] if pred[0] == SENTI_NEGATIVE else metric[4]
        m.p_total += 1

        if gold[0] == pred[0]:
            m.correct += 1
