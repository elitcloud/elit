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
import numpy as np

__author__ = 'Jinho D. Choi'


class NLPState(metaclass=abc.ABCMeta):
    def __init__(self, document):
        """
        NLPState defines a decoding strategy to process the input document.
        :param document: the input document.
        :type document: elit.nlp.structure.Document
        """
        self.document = document
        self.output = None

    @abc.abstractmethod
    def reset(self):
        """
        Resets to the initial state.
        """
        pass

    @abc.abstractmethod
    def process(self, output):
        """
        Applies the output to the current state, and moves onto the next state.
        :param output: the prediction output of the current state.
        :type output: numpy.array
        """
        pass

    @abc.abstractmethod
    def has_next(self):
        """
        :return: True if there exists the next state to be processed; otherwise, False.
        :rtype: bool
        """
        return

    @abc.abstractmethod
    def get_labels(self):
        """
        :return: the predicted labels for the input document.
        :rtype: depend on the task
        """
        return

    @abc.abstractmethod
    def set_labels(self, key):
        """
        Sets predicted labels to the input document using the key.
        :param key: the key to indicate where the predicted labels are stored in.
        :param key: str
        """
        pass

    @abc.abstractmethod
    def eval(self, metric):
        """
        Updates the evaluation metric using the gold-standard labels in document and predicted labels from get_labels.
        :param metric: the evaluation metric.
        :type metric: elit.util.metric.Metric
        """
        pass


class ForwardState(NLPState, metaclass=abc.ABCMeta):
    def __init__(self, document, label_map, zero_output):
        """
        ForwardState defines the one-pass, left-to-right decoding strategy.
        :param document: the input document.
        :type document: elit.nlp.structure.Document
        :param label_map: the mapping between class labels and their unique IDs.
        :type label_map: elit.nlp.util.LabelMap
        :param zero_output: a vector of size `num_class` where all values are 0; used to zero pad label embeddings.
        :type zero_output: numpy.array
        """
        super().__init__(document)
        self.label_map = label_map
        self.zero_output = zero_output
        self.sen_id = 0
        self.tok_id = 0
        self.reset()

    def reset(self):
        self.output = [[self.zero_output] * len(s) for s in self.document]
        self.sen_id = 0
        self.tok_id = 0

    def process(self, output):
        # apply the output to the current state
        self.output[self.sen_id][self.tok_id] = output

        # move onto the next state
        self.tok_id += 1
        if self.tok_id == len(self.document[self.sen_id]):
            self.sen_id += 1
            self.tok_id = 0

    def has_next(self):
        return 0 <= self.sen_id < len(self.document)

    def get_labels(self):
        """
        :rtype: list of (list of str)
        """
        def aux(scores):
            if size < len(scores): scores = scores[:size]
            return self.label_map.get(np.argmax(scores))

        size = len(self.label_map)
        return [[aux(o) for o in output] for output in self.output]

    def set_labels(self, key):
        for i, labels in enumerate(self.get_labels()):
            self.document[i][key] = labels
