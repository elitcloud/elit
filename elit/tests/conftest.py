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
import pytest
from ..tokenizer import Tokenizer, SpaceTokenizer, EnglishTokenizer
from ..segmenter import EnglishSegmenter
from ..util import Accuracy, F1

__author__ = "Gary Lai"


@pytest.fixture()
def tokenizer():
    return Tokenizer()


@pytest.fixture()
def space_tokenizer():
    return SpaceTokenizer()


@pytest.fixture()
def accuracy():
    return Accuracy()


@pytest.fixture()
def f1():
    return F1()
