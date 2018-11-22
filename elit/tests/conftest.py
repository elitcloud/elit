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
import os

import pytest

from elit.nlp.morph_analyzer import EnglishMorphAnalyzer
from elit.nlp.tokenizer import Tokenizer, SpaceTokenizer, EnglishTokenizer
from elit.util.reader import tsv_reader, json_reader


__author__ = "Gary Lai"

current_path = os.path.abspath(os.path.dirname(__file__))


@pytest.fixture()
def tokenizer():
    return Tokenizer()


@pytest.fixture()
def space_tokenizer():
    return SpaceTokenizer()


@pytest.fixture()
def english_tokenizer():
    return EnglishTokenizer()


@pytest.fixture()
def en_morph_analyzer():
    return EnglishMorphAnalyzer()


@pytest.fixture()
def tsv_reader():
    return tsv_reader


@pytest.fixture()
def json_reader():
    return json_reader