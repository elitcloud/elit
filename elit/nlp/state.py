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


class NLPState(metaclass=abc.ABCMeta):
    def __init__(self, document):
        """
        NLPState implements a decoding strategy to process the input document.
        :param document: the input document.
        :type document: list of elit.util.structure.Sentence
        """
        self.document = document
        self.output = None

    @abc.abstractmethod
    def reset(self):
        """
        Resets to the beginning state.
        """
        pass

    @abc.abstractmethod
    def process(self, output):
        """
        Applies the output to the current state, and move onto the next state.
        :param output: the prediction output of the current state.
        :type output: numpy.array
        """
        pass

    @property
    @abc.abstractmethod
    def has_next(self):
        """
        :return: True if there exists the next state that can be processed; otherwise, False.
        :rtype: bool
        """
        return


class ForwardState(NLPState, metaclass=abc.ABCMeta):
    def __init__(self, document, zero_output):
        """
        ForwardState implements the one-pass, left-to-right decoding strategy.
        :param document: the input document.
        :type document: list of elit.util.structure.Sentence
        :param zero_output: a vector of size `num_class` where all values are 0; used to zero pad label embeddings.
        :type zero_output: numpy.array
        """
        super().__init__(document)
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

    @property
    def has_next(self):
        return 0 <= self.sen_id < len(self.document)
