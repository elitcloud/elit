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
from elit.component import NLPState

__author__ = 'Jinho D. Choi'

class SentiState(NLPState):
    def __init__(self, document, vsm, label_map, maxlen):
        """
        POSState inherits the left-to-right one-pass (LR1P) decoding strategy from ForwardState.
        :param document: an input document.
        :type document: elit.structure.Document
        :param vsm: a vector space model for word embeddings.
        :type vsm: elit.vsm.VectorSpaceModel
        :param label_map: the mapping between class labels and their unique IDs.
        :type label_map: elit.vsm.LabelMap
        """
        super().__init__(document)
        self.emb = vsm.document_matrix(document.tokens, maxlen)
        self.label_map = label_map

    def reset(self):
        """
        Nothing to reset.
        """
        pass

    def process(self, output):
        self.outputs

    def eval(self, metric):
        """
        :param metric: the accuracy metric.
        :type metric: elit.util.Accuracy
        """
        autos = self.labels

        for i, sentence in enumerate(self.document):
            gold = sentence[POS]
            auto = autos[i]
            metric.correct += len([1 for g, p in zip(gold, auto) if g == p])
            metric.total += len(gold)

    @property
    def x(self):
        """
        :return: the n * d matrix where n = # of feature_windows and d = sum(vsm_list) + position emb + label emb
        """
        t = len(self.document.get_sentence(self.sen_id))
        l = ([x_extract(self.tok_id, w, t, emb[self.sen_id], zero) for w in self.feature_windows] for emb, zero in self.embs)
        n = np.column_stack(l)
        return n