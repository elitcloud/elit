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

from elit.morph_analyzer import MorphTag, extract_suffix
from elit.morph_analyzer import EnglishMorphAnalyzer as EM

__author__ = "Jinho D. Choi"


data_extract_suffix = [
    ([
        (('best', 'good'), ''),
        (('bit', 'bite'), '-e'),
        (('fed', 'feed'), '-e-'),
        (('pled', 'plead'), '-a-'),
        (('studies', 'study'), '+ies'),
        (('quizzes', 'quiz'), '+es'),
        (('begirt', 'begird'), '+t'),
        (('bit', 'beat'), '+i+'),
        (('bound', 'bind'), '+ou+'),
    ])
]


@pytest.mark.parametrize('data', data_extract_suffix)
def test_data_extract_suffix(data):
    input, expected = zip(*data)
    actual = tuple(extract_suffix(token, lemma) for token, lemma in input)
    assert actual == expected


data_irregular = [
    ([
        (('ai', 'VBP'), [[('be', EM.V)]]),
        (('ai', None), [[('ai', EM.N)]]),
        (('was', None), [[('be', EM.V), ('', MorphTag.I_3PS), ('', MorphTag.I_PAS)]]),
        (("'d", None), [[('have', EM.V), ('+d', MorphTag.I_PAS)], [('would', EM.M)]]),
    ])
]


@pytest.mark.parametrize('data', data_irregular)
def test_irregular(en_morph_analyzer, data):
    input, expected = zip(*data)
    actual = tuple(en_morph_analyzer.analyze(token, pos) for token, pos in input)
    assert actual == expected


data_base = [
    ([
        (('study', 'VB'), [[('study', 'V')]]),
        (('bound', 'JJ'), [[('bound', 'J')]]),
        (('Jinho', 'NNP'), [[('jinho', 'N')]]),
        (('study', None), [[('study', 'V')], [('study', 'N')]]),
        (('bound', None), [[('bound', 'V')], [('bind', 'V'), ('+ou+', MorphTag.I_PAS)], [('bound', 'N')], [('bound', 'J')]]),
     ])
]


@pytest.mark.parametrize('data', data_base)
def test_base(en_morph_analyzer, data):
    input, expected = zip(*data)
    actual = tuple(en_morph_analyzer.analyze(token, pos) for token, pos in input)
    assert actual == expected


data_inflection = [
    ([
         # verb: 3rd-person singular
         (('studies', 'VBZ'), [[('study', EM.V), ('+ies', MorphTag.I_3PS)]]),
         (('pushes', 'VBZ'), [[('push', EM.V), ('+es', MorphTag.I_3PS)]]),
         (('takes', 'VBZ'), [[('take', EM.V), ('+s', MorphTag.I_3PS)]]),
         # verb: gerund
         (('lying', 'VBG'), [[('lie', EM.V), ('+ying', MorphTag.I_GER)]]),
         (('feeling', 'VBG'), [[('feel', EM.V), ('+ing', MorphTag.I_GER)]]),
         (('taking', 'VBG'), [[('take', EM.V), ('+ing', MorphTag.I_GER)]]),
         (('running', 'VBG'), [[('run', EM.V), ('+ing', MorphTag.I_GER)]]),
         # verb: past (participle)
         (('denied', 'VBD'), [[('deny', EM.V), ('+ied', MorphTag.I_PAS)]]),
         (('entered', 'VBD'), [[('enter', EM.V), ('+ed', MorphTag.I_PAS)]]),
         (('zipped', 'VBD'), [[('zip', EM.V), ('+ed', MorphTag.I_PAS)]]),
         (('heard', 'VBD'), [[('hear', EM.V), ('+d', MorphTag.I_PAS)]]),
         (('fallen', 'VBN'), [[('fall', EM.V), ('+en', MorphTag.I_PAS)]]),
         (('written', 'VBN'), [[('write', EM.V), ('+en', MorphTag.I_PAS)]]),
         (('drawn', 'VBN'), [[('draw', EM.V), ('+n', MorphTag.I_PAS)]]),
         (('clung', 'VBN'), [[('cling', EM.V), ('+ung', MorphTag.I_PAS)]]),
         # verb: irregular
         (('bit', 'VBD'), [[('bite', EM.V), ('-e', MorphTag.I_PAS)]]),
         (('bites', 'VBZ'), [[('bite', EM.V), ('+s', MorphTag.I_3PS)]]),
         (('biting', 'VBG'), [[('bite', EM.V), ('+ing', MorphTag.I_GER)]]),
         (('bitted', 'VBD'), [[('bit', EM.V), ('+ed', MorphTag.I_PAS)]]),
         (('bitten', 'VBN'), [[('bite', EM.V), ('+en', MorphTag.I_PAS)]]),
         (('bitting', 'VBG'), [[('bit', EM.V), ('+ing', MorphTag.I_GER)]]),
         (('chivvies', 'VBZ'), [[('chivy', EM.V), ('+ies', MorphTag.I_3PS)]]),
         (('took', 'VBD'), [[('take', EM.V), ('+ook', MorphTag.I_PAS)]]),
         (('slept', 'VBD'), [[('sleep', EM.V), ('+pt', MorphTag.I_PAS)]]),
         (('spoken', 'VBN'), [[('speak', EM.V), ('+oken', MorphTag.I_PAS)]]),
         (('woken', 'VBN'), [[('wake', EM.V), ('+oken', MorphTag.I_PAS)]]),
         # noun: plural
         (('studies', 'NNS'), [[('study', EM.N), ('+ies', MorphTag.I_PLU)]]),
         (('crosses', 'NNS'), [[('cross', EM.N), ('+es', MorphTag.I_PLU)]]),
         (('areas', 'NNS'), [[('area', EM.N), ('+s', MorphTag.I_PLU)]]),
         (('men', 'NNS'), [[('man', EM.N), ('+men', MorphTag.I_PLU)]]),
         (('vertebrae', 'NNS'), [[('vertebra', EM.N), ('+ae', MorphTag.I_PLU)]]),
         (('foci', 'NNS'), [[('focus', EM.N), ('+i', MorphTag.I_PLU)]]),
         (('optima', 'NNS'), [[('optimum', EM.N), ('+a', MorphTag.I_PLU)]]),
         # noun: irregular
         (('indices', 'NNS'), [[('index', EM.N), ('+ices', MorphTag.I_PLU)]]),
         (('wolves', 'NNS'), [[('wolf', EM.N), ('+ves', MorphTag.I_PLU)]]),
         (('knives', 'NNS'), [[('knife', EM.N), ('+ves', MorphTag.I_PLU)]]),
         (('quizzes', 'NNS'), [[('quiz', EM.N), ('+es', MorphTag.I_PLU)]]),
         # adjective: comparative
         (('easier', 'JJR'), [[('easy', EM.J), ('+ier', MorphTag.I_COM)]]),
         (('larger', 'JJR'), [[('large', EM.J), ('+er', MorphTag.I_COM)]]),
         (('smaller', 'JJR'), [[('small', EM.J), ('+er', MorphTag.I_COM)]]),
         (('bigger', 'JJR'), [[('big', EM.J), ('+er', MorphTag.I_COM)]]),
         # adjective: superlative
         (('easiest', 'JJS'), [[('easy', EM.J), ('+iest', MorphTag.I_SUP)]]),
         (('largest', 'JJS'), [[('large', EM.J), ('+est', MorphTag.I_SUP)]]),
         (('smallest', 'JJS'), [[('small', EM.J), ('+est', MorphTag.I_SUP)]]),
         (('biggest', 'JJS'), [[('big', EM.J), ('+est', MorphTag.I_SUP)]]),
         # adjective: irregular
         (('cagier', 'JJR'), [[('cagey', EM.J), ('+ier', MorphTag.I_COM)]]),
         (('worse', 'JJR'), [[('bad', EM.J), ('', MorphTag.I_COM)]]),
         # adverb: comparative
         (('earlier', 'RBR'), [[('early', EM.R), ('+ier', MorphTag.I_COM)]]),
         (('sooner', 'RBR'), [[('soon', EM.R), ('+er', MorphTag.I_COM)]]),
         (('larger', 'RBR'), [[('large', EM.R), ('+er', MorphTag.I_COM)]]),
         # adverb: superlative
         (('earliest', 'RBS'), [[('early', EM.R), ('+iest', MorphTag.I_SUP)]]),
         (('soonest', 'RBS'), [[('soon', EM.R), ('+est', MorphTag.I_SUP)]]),
         (('largest', 'RBS'), [[('large', EM.R), ('+est', MorphTag.I_SUP)]]),
         # adverb: irregular
         (('further', 'RBR'), [[('far', EM.R), ('+urthe+', MorphTag.I_COM)]]),
         (('best', 'RBS'), [[('well', EM.R), ('', MorphTag.I_SUP)]]),
         (('worst', 'RBS'), [[('bad', EM.R), ('', MorphTag.I_SUP)]]),
     ])
]


@pytest.mark.parametrize('data', data_inflection)
def test_inflection(en_morph_analyzer, data):
    input, expected = zip(*data)
    actual = tuple(en_morph_analyzer.analyze(token, pos) for token, pos in input)
    assert actual == expected





