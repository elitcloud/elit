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

__author__ = 'Jinho D. Choi'


class NLPState(abc.ABC):
    def __init__(self, document):
        """
        NLPState provides a generic template to define a decoding strategy.
        :param document: an input document.
        :type document: elit.structure.Document
        """
        self.document = document
        self.outputs = None

    @abc.abstractmethod
    def reset(self):
        """
        Resets to the initial state.
        """
        pass

    @abc.abstractmethod
    def process(self, output):
        """
        Applies predicted output to the current state, and moves onto the next state.
        :param output: the predicted output (e.g., the output layer of a neural network).
        :type output: numpy.array
        """
        pass

    @abc.abstractmethod
    def has_next(self):
        """
        :return: True if there exists a next state to be processed; otherwise, False.
        :rtype: bool
        """
        return

    @abc.abstractmethod
    def finalize(self):
        """
        Saves all predicted outputs (self.outputs) and inferred labels (self.labels) to the input document (self.document).
        """
        pass

    @abc.abstractmethod
    def eval(self, metric):
        """
        Updates the evaluation metric by comparing the gold-standard labels (if available) and the inferred labels (self.labels).
        :param metric: the evaluation metric.
        :type metric: elit.util.EvalMetric
        """
        pass

    @property
    @abc.abstractmethod
    def labels(self):
        """
        :return: the labels for the input document inferred from `self.outputs`.
        """
        return

    @property
    @abc.abstractmethod
    def x(self):
        """
        :return: the feature vector (or matrix) extracted from the current state.
        :rtype: numpy.array
        """
        return

    @property
    @abc.abstractmethod
    def y(self):
        """
        :return: the class ID of the gold-standard label for the current state (training only).
        :rtype: int
        """
        return


class OPLRState(NLPState):
    def __init__(self, document, label_map, padout, key, key_out=None):
        """
        LR1PState defines the one-pass left-to-right (OPLR) decoding strategy.
        :param document: an input document.
        :type document: elit.structure.Document
        :param label_map: collects class labels during training and maps them to unique IDs.
        :type label_map: elit.vsm.LabelMap
        :param padout: a vector whose dimension is the number of class labels, where all values are 0.
        :type padout: numpy.array
        :param key: the key to each sentence in the input document where the inferred labels (self.labels) are saved.
        :type key: str
        :param key_out: the key to each sentence in the input document where the predicted outputs (self.outputs) are saved.
        :type key_out: str
        """
        super().__init__(document)

        self.label_map = label_map
        self.padout = padout

        self.key = key
        self.key_out = key_out if key_out else key + '-out'

        self.sen_id = 0    # sentence ID
        self.tok_id = 0    # token ID
        self.reset()

    @abc.abstractmethod
    def eval(self, metric):
        pass

    @property
    @abc.abstractmethod
    def x(self):
        return

    def reset(self):
        if self.outputs is None:
            self.outputs = [[self.padout] * len(s) for s in self.document]
        else:
            for i, s in enumerate(self.document):
                self.outputs[i] = [self.padout] * len(s)

        self.sen_id = 0
        self.tok_id = 0

    def process(self, output):
        # apply the output to the current state
        self.outputs[self.sen_id][self.tok_id] = output

        # move onto the next state
        self.tok_id += 1
        if self.tok_id == len(self.document.get_sentence(self.sen_id)):
            self.sen_id += 1
            self.tok_id = 0

    def has_next(self):
        return 0 <= self.sen_id < len(self.document)

    def finalize(self):
        """
        Saves the predicted outputs (self.outputs) and the inferred labels (self.labels) to the input document after decoding.
        """
        for i, labels in enumerate(self.labels):
            d = self.document.get_sentence(i)
            d[self.key] = labels
            d[self.key_out] = self.outputs[i]

    @property
    def labels(self):
        """
        :rtype: list of (list of str)
        """
        def aux(scores):
            if size < len(scores): scores = scores[:size]
            return self.label_map.get(np.argmax(scores))

        size = len(self.label_map)
        return [[aux(o) for o in output] for output in self.outputs]

    @property
    def y(self):
        label = self.document.get_sentence(self.sen_id)[self.key][self.tok_id]
        return self.label_map.add(label)

    def _get_label_embeddings(self):
        """
        :return: (self.output, self.padout)
        """
        return self.outputs, self.padout


