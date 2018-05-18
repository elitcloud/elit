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
import os
from elitsdk.sdk import Component
from elitsdk.aux.model import Model

__author__ = "Jinho D. Choi, Gary Lai"


class Tokenizer(Component):
    def __init__(self):
        super(Tokenizer, self).__init__()
        pass

    @abc.abstractmethod
    def decode(self, input_data, offset=0, **kwargs):
        """

        :type input_data: str
        :type offset: int
        :param input_data: the input text.
        :param offset: the starting offset.
        :return: the tuple of (tokens, offsets[, custom values]*); see the comments for Tokenizer.offsets() for more details about the offsets.
        :rtype: json
        """
        pass

    def load(self, model_path, *args, **kwargs):
        """

        :param model_path:
        :param args:
        :param kwargs:
        :return:
        """
        pass

    def train(self, trn_data, dev_data, *args, **kwargs):
        """

        :param trn_data:
        :param dev_data:
        :param args:
        :param kwargs:
        :return:
        """
        pass

    def save(self, model_path, *args, **kwargs):
        """

        :param model_path:
        :param args:
        :param kwargs:
        :return:
        """
        pass

    @staticmethod
    def offsets(input_data, tokens, offset=0):
        """

        :type input_data: str
        :type tokens: list of str
        :type offset: int
        :param input_data: the input text.
        :param tokens: the list of tokens split from the input text.
        :param offset: offset of tokens
        :return: the list of (begin, end) offsets, where the begin (inclusive) and the end (exclusive) offsets indicate the caret positions of the first and the last characters of the corresponding token, respectively. e.g., text = 'Hello, world!', tokens = ['Hello', ',', 'world', '!'] -> [(0, 5), (5, 6), (7, 12), (12, 13)]
        :rtype: json
        """
        def get_offset(token):
            nonlocal end
            begin = input_data.index(token, end)
            end = begin + len(token)
            return begin + offset, end + offset

        end = 0
        return [get_offset(token) for token in tokens]


class SpaceTokenizer(Tokenizer):

    def __init__(self):
        super(SpaceTokenizer, self).__init__()

    def decode(self, input_data, offset=0, **kwargs):
        """

        :param input_data:
        :param offset:
        :param kwargs:
        :return:
        """
        tokens = input_data.split()
        return tokens, self.offsets(input_data, tokens, offset)
