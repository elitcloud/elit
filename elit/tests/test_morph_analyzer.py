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

from elit.morph_analyzer import MorphTag

__author__ = "Jinho D. Choi"

data_inflection = [
    ([
         ('studies', 'VBZ'),
         ('pushes', 'VBZ'),
         ('takes', 'VBZ'),
         ('lying', 'VBG'),
         ('feeling', 'VBG'),
         ('taking', 'VBG'),
         ('running', 'VBG'),
         ('denied', 'VBD'),
         ('entered', 'VBD'),
         ('zipped', 'VBD'),
         ('heard', 'VBD'),
         ('fallen', 'VBN'),
         ('written', 'VBN'),
         ('drawn', 'VBN'),
         ('clung', 'VBN'),
         ('studies', 'NNS'),
         ('crosses', 'NNS'),
         ('areas', 'NNS'),
         ('men', 'NNS'),
         ('vertebrae', 'NNS'),
         ('foci', 'NNS'),
         ('optima', 'NNS'),
         ('easier', 'JJR'),
         ('larger', 'JJR'),
         ('smaller', 'JJR'),
         ('bigger', 'JJR'),
         ('easiest', 'JJS'),
         ('largest', 'JJS'),
         ('smallest', 'JJS'),
         ('biggest', 'JJS'),
         ('earlier', 'RBR'),
         ('sooner', 'RBR'),
         ('larger', 'RBR'),
         ('earliest', 'RBS'),
         ('soonest', 'RBS'),
         ('largest', 'RBS'),
     ],
     [
         [[('study', 'VB'), ('ies', MorphTag.TPS)]],
         [[('push', 'VB'), ('es', MorphTag.TPS)]],
         [[('take', 'VB'), ('s', MorphTag.TPS)]],
         [[('lie', 'VB'), ('ying', MorphTag.GER)]],
         [[('feel', 'VB'), ('ing', MorphTag.GER)]],
         [[('take', 'VB'), ('ing', MorphTag.GER)]],
         [[('run', 'VB'), ('ing', MorphTag.GER)]],
         [[('deny', 'VB'), ('ied', MorphTag.PAS)]],
         [[('enter', 'VB'), ('ed', MorphTag.PAS)]],
         [[('zip', 'VB'), ('ed', MorphTag.PAS)]],
         [[('hear', 'VB'), ('d', MorphTag.PAS)]],
         [[('fall', 'VB'), ('en', MorphTag.PAS)]],
         [[('write', 'VB'), ('en', MorphTag.PAS)]],
         [[('draw', 'VB'), ('n', MorphTag.PAS)]],
         [[('cling', 'VB'), ('ung', MorphTag.PAS)]],
         [[('study', 'NN'), ('ies', MorphTag.PLU)]],
         [[('cross', 'NN'), ('es', MorphTag.PLU)]],
         [[('area', 'NN'), ('s', MorphTag.PLU)]],
         [[('man', 'NN'), ('men', MorphTag.PLU)]],
         [[('vertebra', 'NN'), ('ae', MorphTag.PLU)]],
         [[('focus', 'NN'), ('i', MorphTag.PLU)]],
         [[('optimum', 'NN'), ('a', MorphTag.PLU)]],
         [[('easy', 'JJ'), ('ier', MorphTag.COM)]],
         [[('large', 'JJ'), ('er', MorphTag.COM)]],
         [[('small', 'JJ'), ('er', MorphTag.COM)]],
         [[('big', 'JJ'), ('er', MorphTag.COM)]],
         [[('easy', 'JJ'), ('iest', MorphTag.SUP)]],
         [[('large', 'JJ'), ('est', MorphTag.SUP)]],
         [[('small', 'JJ'), ('est', MorphTag.SUP)]],
         [[('big', 'JJ'), ('est', MorphTag.SUP)]],
         [[('early', 'RB'), ('ier', MorphTag.COM)]],
         [[('soon', 'RB'), ('er', MorphTag.COM)]],
         [[('large', 'RB'), ('er', MorphTag.COM)]],
         [[('early', 'RB'), ('iest', MorphTag.SUP)]],
         [[('soon', 'RB'), ('est', MorphTag.SUP)]],
         [[('large', 'RB'), ('est', MorphTag.SUP)]],
     ])
]


@pytest.mark.parametrize('input, expected', data_inflection)
def test_inflection(en_morph_analyzer, input, expected):
    actual = [en_morph_analyzer.analyze(token, pos) for token, pos in input]
    assert actual == expected




    # test_data_empty_input = [(list(), list()), ]
    #
    #
    # @pytest.mark.parametrize('input, expected', test_data_empty_input)
    # def test_empty_input(english_lemmatizer, input, expected):
    #     actual = english_lemmatizer.decode(input)
    #     assert actual == expected


    # test_data_cardinal = [
    #     ([("He", "PRP"), ("has", "VBZ"), ("one", "CD"), ("paper", "NN")], ["he", "have", "#crd#", "paper"])
    # ]
    #
    #
    # @pytest.mark.parametrize('input, expected', test_data_cardinal)
    # def test_cardinal(english_lemmatizer, input, expected):
    #     actual = english_lemmatizer.decode(input)
    #     assert actual == expected
    #
    #
    # test_data_ordinal = [
    #     ([("He", "PRP"), ("is", "VBZ"), ("the", "DT"), ("first", "JJ")], ["he", "be", "the", "#ord#"])
    # ]
    #
    #
    # @pytest.mark.parametrize('input, expected', test_data_ordinal)
    # def test_ordinal(english_lemmatizer, input, expected):
    #     actual = english_lemmatizer.decode(input)
    #     assert actual == expected
    #
    #
    # test_data_exception = [
    #     ([("He", "PRP"), ("bought", "VBD"), ("a", "DT"), ("car", "NN")], ["he", "buy", "a", "car"])
    # ]
    #
    #
    # @pytest.mark.parametrize('input, expected', test_data_exception)
    # def test_exception(english_lemmatizer, input, expected):
    #     actual = english_lemmatizer.decode(input)
    #     assert actual == expected
    #
    #
    # test_data_VBZ_no_replacement = [
    #     ([("He", "PRP"), ("likes", "VBZ"), ("flowers", "NNS")], ["he", "like", "flower"]),
    #     ([("He", "PRP"), ("pushes", "VBZ")], ["he", "push"])
    # ]
    #
    #
    # @pytest.mark.parametrize('input, expected', test_data_VBZ_no_replacement)
    # def test_VBZ_no_replacement(english_lemmatizer, input, expected):
    #     actual = english_lemmatizer.decode(input)
    #     assert actual == expected
    #
    #
    # test_data_VBZ_with_replacement = [
    #     ([("He", "PRP"), ("studies", "VBZ"), ("history", "NN")], ["he", "study", "history"])
    # ]
    #
    #
    # @pytest.mark.parametrize('input, expected', test_data_VBZ_with_replacement)
    # def test_VBZ_with_replacement(english_lemmatizer, input, expected):
    #     actual = english_lemmatizer.decode(input)
    #     assert actual == expected
    #
    #
    # test_data_JJR_double_consonants = [
    #     ([("He", "PRP"), ("has", "VBZ"), ("a", "DT"), ("bigger", "JJR"), ("car", "NN")],
    #      ["he", "have", "a", "big", "car"])
    # ]
    #
    #
    # @pytest.mark.parametrize('input, expected', test_data_JJR_double_consonants)
    # def test_JJR_double_consonants(english_lemmatizer, input, expected):
    #     actual = english_lemmatizer.decode(input)
    #     assert actual == expected
