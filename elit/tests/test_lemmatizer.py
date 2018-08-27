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

__author__ = "Liyan Xu"


test_data_empty_input = [(list(), list()), ]


@pytest.mark.parametrize('input, expected', test_data_empty_input)
def test_empty_input(english_lemmatizer, input, expected):
    actual = english_lemmatizer.decode(input)
    assert actual == expected


test_data_abbreviation = [
    ([("He", "PRP"), ("is", "VBZ"), ("n't", "RB"), ("tall", "JJ")], ["he", "be", "not", "tall"])
]


@pytest.mark.parametrize('input, expected', test_data_abbreviation)
def test_abbreviation(english_lemmatizer, input, expected):
    actual = english_lemmatizer.decode(input)
    assert actual == expected


test_data_cardinal = [
    ([("He", "PRP"), ("has", "VBZ"), ("one", "CD"), ("paper", "NN")], ["he", "have", "#crd#", "paper"])
]


@pytest.mark.parametrize('input, expected', test_data_cardinal)
def test_cardinal(english_lemmatizer, input, expected):
    actual = english_lemmatizer.decode(input)
    assert actual == expected


test_data_ordinal = [
    ([("He", "PRP"), ("is", "VBZ"), ("the", "DT"), ("first", "JJ")], ["he", "be", "the", "#ord#"])
]


@pytest.mark.parametrize('input, expected', test_data_ordinal)
def test_ordinal(english_lemmatizer, input, expected):
    actual = english_lemmatizer.decode(input)
    assert actual == expected


test_data_exception = [
    ([("He", "PRP"), ("bought", "VBD"), ("a", "DT"), ("car", "NN")], ["he", "buy", "a", "car"])
]


@pytest.mark.parametrize('input, expected', test_data_exception)
def test_exception(english_lemmatizer, input, expected):
    actual = english_lemmatizer.decode(input)
    assert actual == expected


test_data_VBZ_no_replacement = [
    ([("He", "PRP"), ("likes", "VBZ"), ("flowers", "NNS")], ["he", "like", "flower"]),
    ([("He", "PRP"), ("pushes", "VBZ")], ["he", "push"])
]


@pytest.mark.parametrize('input, expected', test_data_VBZ_no_replacement)
def test_VBZ_no_replacement(english_lemmatizer, input, expected):
    actual = english_lemmatizer.decode(input)
    assert actual == expected


test_data_VBZ_with_replacement = [
    ([("He", "PRP"), ("studies", "VBZ"), ("history", "NN")], ["he", "study", "history"])
]


@pytest.mark.parametrize('input, expected', test_data_VBZ_with_replacement)
def test_VBZ_with_replacement(english_lemmatizer, input, expected):
    actual = english_lemmatizer.decode(input)
    assert actual == expected


test_data_JJR_double_consonants = [
    ([("He", "PRP"), ("has", "VBZ"), ("a", "DT"), ("bigger", "JJR"), ("car", "NN")],
     ["he", "have", "a", "big", "car"])
]


@pytest.mark.parametrize('input, expected', test_data_JJR_double_consonants)
def test_JJR_double_consonants(english_lemmatizer, input, expected):
    actual = english_lemmatizer.decode(input)
    assert actual == expected