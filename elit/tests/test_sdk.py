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

import pytest

from ..benchmark import Timer
from ..sdk import Component

__author__ = "Gary Lai"


class Tokenizer(Component, abc.ABC):

    def init(self):
        pass

    @abc.abstractmethod
    def decode(self, input_data, offset=0, **kwargs):
        """
        :param input_data: the input text.
        :param offset: the starting offset.
        :type input_data: str
        :type offset: int
        :return:
            the tuple of (tokens, offsets[, custom values]*);
            see the comments for Tokenizer.offsets() for more details about the offsets.
        :rtype: (list of str, list of (int, int), *args)
        """
        pass

    def load(self, model_path, **kwargs):
        pass

    def train(self, trn_data, dev_data, model_path, **kwargs):
        pass

    def save(self, model_path, **kwargs):
        pass

    @staticmethod
    def offsets(input_data, tokens, offset=0):
        """
        :param input_data: the input text.
        :param tokens: the list of tokens split from the input text.
        :param offset:
        :type input_data: str
        :type tokens: list of str
        :type offset: int
        :return:
            the list of (begin, end) offsets, where the begin (inclusive) and the end (exclusive) offsets indicate
            the caret positions of the first and the last characters of the corresponding token, respectively.
            e.g., text = 'Hello, world!', tokens = ['Hello', ',', 'world', '!'] -> [(0, 5), (5, 6), (7, 12), (12, 13)]
        :rtype list of (int, int)
        """
        def get_offset(token):
            nonlocal end
            begin = input_data.index(token, end)
            end = begin + len(token)
            return begin + offset, end + offset

        end = 0
        return [get_offset(token) for token in tokens]


class SpaceTokenizer(Tokenizer):

    def decode(self, input_data, offset=0, **kwargs):
        tokens = input_data.split()
        return tokens, self.offsets(input_data, tokens, offset)


def test_sdk():

    class TestSDK(Component):
        pass

    with pytest.raises(TypeError):
        TestSDK()


def test_abstract_class():

    class TestSDK(Component):

        def __init__(self):
            super().__init__()

        def init(self):
            super().init()

        def decode(self, input_data, **kwargs):
            super().decode(input_data, **kwargs)

        def load(self, model_path, **kwargs):
            super().load(model_path, **kwargs)

        def train(self, trn_data, dev_data, model_path, **kwargs):
            super().train(trn_data, dev_data, model_path, **kwargs)

        def save(self, model_path, **kwargs):
            super().save(model_path, **kwargs)
        
    with pytest.raises(NotImplementedError):
        test_task = TestSDK()
        test_task.decode("test")


def test_space_tokenizer():
    space_tokenizer = SpaceTokenizer()
    tokens, offset = space_tokenizer.decode("Hello, world")
    assert tokens == ['Hello,', 'world']
    assert offset == [(0, 6), (7, 12)]


def test_benchmark():
    space_tokenizer = SpaceTokenizer()
    with Timer() as time1:
        space_tokenizer.decode("Hello, world")
    with Timer() as time2:
        space_tokenizer.decode("This module implements specialized container datatypes providing alternatives to Python’s general purpose built-in containers.")

    assert isinstance(time1.runtime, float)
    assert isinstance(time2.runtime, float)
    assert time1.runtime > 0.0
    assert time2.runtime > 0.0
