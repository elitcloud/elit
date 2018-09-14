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
    ([("He", "PRP"), ("is", "VBZ"), ("n't", "RB"), ("tall", "JJ")], [("he",[]),("be",[]), ("not",[]), ("tall",[])])
]


@pytest.mark.parametrize('input, expected', test_data_abbreviation)
def test_abbreviation(english_lemmatizer, input, expected):
    actual = english_lemmatizer.decode(input)
    assert actual == expected


test_data_cardinal = [
    ([("He", "PRP"), ("has", "VBZ"), ("one", "CD"), ("paper", "NN")], [("he",[]), ("have",[]), ("#crd#",[]), ("paper",[])])
]


@pytest.mark.parametrize('input, expected', test_data_cardinal)
def test_cardinal(english_lemmatizer, input, expected):
    actual = english_lemmatizer.decode(input)
    assert actual == expected


test_data_ordinal = [
    ([("He", "PRP"), ("is", "VBZ"), ("the", "DT"), ("first", "JJ")], [("he",[]), ("be",[]), ("the",[]), ("#ord#",[])])
]


@pytest.mark.parametrize('input, expected', test_data_ordinal)
def test_ordinal(english_lemmatizer, input, expected):
    actual = english_lemmatizer.decode(input)
    assert actual == expected


test_data_exception = [
    ([("He", "PRP"), ("bought", "VBD"), ("a", "DT"), ("car", "NN")], [("he",[]), ("buy",[]), ("a",[]), ("car",[])])
]


@pytest.mark.parametrize('input, expected', test_data_exception)
def test_exception(english_lemmatizer, input, expected):
    actual = english_lemmatizer.decode(input)
    assert actual == expected


test_data_VBZ_no_replacement = [
    ([("He", "PRP"), ("likes", "VBZ"), ("flowers", "NNS")], [("he",[]), ("like",["s"]), ("flower",["s"])]),
    ([("He", "PRP"), ("pushes", "VBZ")], [("he",[]), ("push",["es"])])
]


@pytest.mark.parametrize('input, expected', test_data_VBZ_no_replacement)
def test_VBZ_no_replacement(english_lemmatizer, input, expected):
    actual = english_lemmatizer.decode(input)
    assert actual == expected


test_data_VBZ_with_replacement = [
    ([("He", "PRP"), ("studies", "VBZ"), ("history", "NN")], [("he",[]), ("study",["ies"]), ("history",[])])
]


@pytest.mark.parametrize('input, expected', test_data_VBZ_with_replacement)
def test_VBZ_with_replacement(english_lemmatizer, input, expected):
    actual = english_lemmatizer.decode(input)
    assert actual == expected


test_data_JJR_double_consonants = [
    ([("He", "PRP"), ("has", "VBZ"), ("a", "DT"), ("bigger", "JJR"), ("car", "NN")],
     [("he",[]), ("have",[]), ("a",[]), ("big",["er"]), ("car",[])])
]


@pytest.mark.parametrize('input, expected', test_data_JJR_double_consonants)
def test_JJR_double_consonants(english_lemmatizer, input, expected):
    actual = english_lemmatizer.decode(input)
    assert actual == expected

test_data_ify_derivation = [([
                ("fortify","VB" ), ("glorify","VB"), ("terrify","VB"),( "qualify","VB"),
                ("simplify","VB"),("fancify","VB")],[
                ("fort",["ify"]),("glory",["ify"]),("terror",["ify"]),("quality",["ify"]),("simple",["ify"]),
                ("fancy",["ify"])])]

@pytest.mark.parametrize('input, expected', test_data_ify_derivation)
def test_ify_derivation(english_lemmatizer, input, expected):
    actual = english_lemmatizer.decode(input)
    assert actual == expected

test_data_ize_derivation = [([
                ("terrorise","VB"),( "normalise","VB")],[
                ("terror",["ize"]),("normal",["ize"])])]

@pytest.mark.parametrize('input, expected', test_data_ize_derivation)
def test_ize_derivation(english_lemmatizer, input, expected):
    actual = english_lemmatizer.decode(input)
    assert actual == expected



test_data_al_derivation = [([
                ("arrival","NN" ), ("defiance","NN"), ("annoyance","NN"),( "insurance","NN"),
                ("deviance", "NN"), ("relevance", "NN"), ("pregnancy", "NN"), ("difference", "NN"),
                ("fluency", "NN"), ("accuracy", "NN"), ("assistant", "NN"), ("servant", "NN"),
                ("immigrant", "NN"), ("resident", "NN")
                             ],[
                ("arrive",["al"]),("defy",["iance"]),("annoy",["ance"]),("insure",["ance"]),
                ("deviate", ["ance"]), ("relevant", ["ance"]), ("pregnant", ["ancy"]), ("differ", ["ence","ent"]),
                ("fluent",["ency"]),("accurate",["acy"]),("assist",["ant"]),("serve",["ant"]),
                ("immigrate", ["ant"]), ("reside", ["ent"])])]

@pytest.mark.parametrize('input, expected', test_data_al_derivation)
def test_al_derivation(english_lemmatizer, input, expected):
    actual = english_lemmatizer.decode(input)
    assert actual == expected

test_data_recursive_derivation = [([
                ("accidentally","RB" ), ("beautifully","RB"), ("academically","RB"),( "gloriously","RB"),
                ("easily", "RB"), ("electronically", "RB")
                             ],[
                ("accident",["ly","al"]),('beauty', ['ly', 'iful']),('academic', ['ally']), ('glory', ['ly', 'ious']), ('eas', ['ily', 'y']), ('electronic', ['ally'])])]

@pytest.mark.parametrize('input, expected', test_data_recursive_derivation)
def test_revursive_derivation(english_lemmatizer, input, expected):
    actual = english_lemmatizer.decode(input)
    assert actual == expected

test_data_36_derivation = [([
                ("carrier","NN" ), ("cashier","NN"), ("financier","NN"),( "writer","NN"),
                ("reader", "NN"), ("liar", "NN"), ("actor", "NN"), ("childhood", "NN"),
                ("baptism", "NN"), ("capitalism", "NN"), ("artist", "NN"), ("agronomist", "NN"),
                ("readability", "NN")
                             ],[
                ("carry",["ier"]),("cash",["ier"]),("finance",["ier"]),("write",["er"]),
                ("read", ["er"]), ("lie", ["ar"]), ("act", ["or"]), ("child", ["hood"]),
                ("baptise",["ism"]),("capital",["ism"]),("art",["ist"]),("agronomy",["ist"]),
                ('read', ['ility', 'able'])])]
@pytest.mark.parametrize('input, expected', test_data_36_derivation)
def test_36_derivation(english_lemmatizer, input, expected):
    actual = english_lemmatizer.decode(input)
    assert actual == expected

test_data_64_derivation = [([
                ("development","NN" ), ("happiness","NN"), ("friendship","NN"),( "diagnosis","NN"),
                ("pronunciation", "NN"), ("verification", "NN"), ("admiration", "NN"), ("loyalty", "NN"),
                ("certifiable", "JJ"), ("writable", "JJ"), ("academical", "JJ"), ("adherent", "JJ"),
                ("boyish", "JJ")
                             ],[
                ("develop",["ment"]),("happy",["iness"]),("friend",["ship"]),("diagnose",["sis"]),
                ("pronounce", ["unciation"]), ('verity', ['ication', 'ify']), ("admire", ["ation"]), ("loyal", ["ty"]),
                ('cert', ['iable', 'ify']),("write",["able"]),("academy",["ical"]),("adhere",["ent"]),
                ('boy', ['ish'])])]
@pytest.mark.parametrize('input, expected', test_data_64_derivation)
def test_64_derivation(english_lemmatizer, input, expected):
    actual = english_lemmatizer.decode(input)
    assert actual == expected




