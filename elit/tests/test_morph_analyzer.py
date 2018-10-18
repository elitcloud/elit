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

from elit.morph_analyzer import extract_suffix
from elit.morph_analyzer import AffixTag as AT
from elit.morph_analyzer import EnglishMorphAnalyzer as EM

__author__ = "Jinho D. Choi"


# ========================= extract_suffix =========================

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
def test_extract_suffix(data):
    input, expected = zip(*data)
    actual = tuple(extract_suffix(token, lemma) for token, lemma in input)
    assert actual == expected


# ========================= analyze_prefix =========================

data_analyze_prefix = [
    ([
        (('repeat', AT.V), [('repeat', 'V')]),
        (('transcribe', AT.V), [('scribe', 'V'), ('tran+', 'P')]),
        (('belittle', AT.V), [('little', 'J'), ('be+', 'P')]),
        (('anemic', AT.J), [('emic', 'J'), ('an+', 'P')]),
        (('co-founder', AT.N), [('founder', 'N'), ('co+', 'P')]),
        (('super-overlook', AT.V), [('look', 'V'), ('super+', 'P'), ('over+', 'P')]),
(('deuteragonist', AT.N), [('agonist', 'N'), ('deuter+', 'P')]),
(('be-deuteragonist', AT.N), [('agonist', 'N'), ('be+', 'P'), ('deuter+', 'P')]),
    ])
]


@pytest.mark.parametrize('data', data_analyze_prefix)
def test_analyze_prefix(en_morph_analyzer, data):
    input, expected = zip(*data)
    actual = tuple(en_morph_analyzer._analyze_prefix(token, pos) for token, pos in input)
    assert actual == expected


# ========================= extract_irregular =========================

data_irregular = [
    ([
        (('ai', 'VBP'), [[('be', AT.V)]]),
        (('ai', None), [[('ai', AT.N)]]),
        (('was', None), [[('be', AT.V), ('', AT.I_3PS), ('', AT.I_PAS)]]),
        (("'d", None), [[('have', AT.V), ('+d', AT.I_PAS)], [('would', AT.M)]]),
    ])
]


@pytest.mark.parametrize('data', data_irregular)
def test_irregular(en_morph_analyzer, data):
    input, expected = zip(*data)
    actual = tuple(en_morph_analyzer.analyze(token, pos) for token, pos in input)
    assert actual == expected


data_base = [
    ([
        (('study', 'VB'), [[('study', AT.V)]]),
        (('bound', 'JJ'), [[('bound', AT.J)]]),
        (('Jinho', 'NNP'), [[('jinho', AT.N)]]),
        (('study', None), [[('study', AT.V)], [('study', AT.N)]]),
        (('bound', None), [[('bound', AT.V)], [('bind', AT.V), ('+ou+', AT.I_PAS)], [('bound', AT.N)], [('bound', AT.J)]]),
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
         (('studies', 'VBZ'), [[('study', AT.V), ('+ies', AT.I_3PS)]]),
         (('pushes', 'VBZ'), [[('push', AT.V), ('+es', AT.I_3PS)]]),
         (('takes', 'VBZ'), [[('take', AT.V), ('+s', AT.I_3PS)]]),

         # verb: gerund
         (('lying', 'VBG'), [[('lie', AT.V), ('+ying', AT.I_GER)]]),
         (('feeling', 'VBG'), [[('feel', AT.V), ('+ing', AT.I_GER)]]),
         (('taking', 'VBG'), [[('take', AT.V), ('+ing', AT.I_GER)]]),
         (('running', 'VBG'), [[('run', AT.V), ('+ing', AT.I_GER)]]),

         # verb: past (participle)
         (('denied', 'VBD'), [[('deny', AT.V), ('+ied', AT.I_PAS)]]),
         (('entered', 'VBD'), [[('enter', AT.V), ('+ed', AT.I_PAS)]]),
         (('zipped', 'VBD'), [[('zip', AT.V), ('+ed', AT.I_PAS)]]),
         (('heard', 'VBD'), [[('hear', AT.V), ('+d', AT.I_PAS)]]),
         (('fallen', 'VBN'), [[('fall', AT.V), ('+en', AT.I_PAS)]]),
         (('written', 'VBN'), [[('write', AT.V), ('+en', AT.I_PAS)]]),
         (('drawn', 'VBN'), [[('draw', AT.V), ('+n', AT.I_PAS)]]),
         (('clung', 'VBN'), [[('cling', AT.V), ('+ung', AT.I_PAS)]]),

         # verb: irregular
         (('bit', 'VBD'), [[('bite', AT.V), ('-e', AT.I_PAS)]]),
         (('bites', 'VBZ'), [[('bite', AT.V), ('+s', AT.I_3PS)]]),
         (('biting', 'VBG'), [[('bite', AT.V), ('+ing', AT.I_GER)]]),
         (('bitted', 'VBD'), [[('bit', AT.V), ('+ed', AT.I_PAS)]]),
         (('bitten', 'VBN'), [[('bite', AT.V), ('+en', AT.I_PAS)]]),
         (('bitting', 'VBG'), [[('bit', AT.V), ('+ing', AT.I_GER)]]),
         (('chivvies', 'VBZ'), [[('chivy', AT.V), ('+ies', AT.I_3PS)]]),
         (('took', 'VBD'), [[('take', AT.V), ('+ook', AT.I_PAS)]]),
         (('slept', 'VBD'), [[('sleep', AT.V), ('+pt', AT.I_PAS)]]),
         (('spoken', 'VBN'), [[('speak', AT.V), ('+oken', AT.I_PAS)]]),
         (('woken', 'VBN'), [[('wake', AT.V), ('+oken', AT.I_PAS)]]),

         # noun: plural
         (('studies', 'NNS'), [[('study', AT.N), ('+ies', AT.I_PLU)]]),
         (('crosses', 'NNS'), [[('cross', AT.N), ('+es', AT.I_PLU)]]),
         (('areas', 'NNS'), [[('area', AT.N), ('+s', AT.I_PLU)]]),
         (('men', 'NNS'), [[('man', AT.N), ('+men', AT.I_PLU)]]),
         (('vertebrae', 'NNS'), [[('vertebra', AT.N), ('+ae', AT.I_PLU)]]),
         (('foci', 'NNS'), [[('focus', AT.N), ('+i', AT.I_PLU)]]),
         (('optima', 'NNS'), [[('optimum', AT.N), ('+a', AT.I_PLU)]]),

         # noun: irregular
         (('indices', 'NNS'), [[('index', AT.N), ('+ices', AT.I_PLU)]]),
         (('wolves', 'NNS'), [[('wolf', AT.N), ('+ves', AT.I_PLU)]]),
         (('knives', 'NNS'), [[('knife', AT.N), ('+ves', AT.I_PLU)]]),
         (('quizzes', 'NNS'), [[('quiz', AT.N), ('+es', AT.I_PLU)]]),

         # adjective: comparative
         (('easier', 'JJR'), [[('easy', AT.J), ('+ier', AT.I_COM)]]),
         (('larger', 'JJR'), [[('large', AT.J), ('+er', AT.I_COM)]]),
         (('smaller', 'JJR'), [[('small', AT.J), ('+er', AT.I_COM)]]),
         (('bigger', 'JJR'), [[('big', AT.J), ('+er', AT.I_COM)]]),

         # adjective: superlative
         (('easiest', 'JJS'), [[('easy', AT.J), ('+iest', AT.I_SUP)]]),
         (('largest', 'JJS'), [[('large', AT.J), ('+est', AT.I_SUP)]]),
         (('smallest', 'JJS'), [[('small', AT.J), ('+est', AT.I_SUP)]]),
         (('biggest', 'JJS'), [[('big', AT.J), ('+est', AT.I_SUP)]]),

         # adjective: irregular
         (('cagier', 'JJR'), [[('cagey', AT.J), ('+ier', AT.I_COM)]]),
         (('worse', 'JJR'), [[('bad', AT.J), ('', AT.I_COM)]]),

         # adverb: comparative
         (('earlier', 'RBR'), [[('early', AT.R), ('+ier', AT.I_COM)]]),
         (('sooner', 'RBR'), [[('soon', AT.R), ('+er', AT.I_COM)]]),
         (('larger', 'RBR'), [[('large', AT.R), ('+er', AT.I_COM)]]),

         # adverb: superlative
         (('earliest', 'RBS'), [[('early', AT.R), ('+iest', AT.I_SUP)]]),
         (('soonest', 'RBS'), [[('soon', AT.R), ('+est', AT.I_SUP)]]),
         (('largest', 'RBS'), [[('large', AT.R), ('+est', AT.I_SUP)]]),

         # adverb: irregular
         (('further', 'RBR'), [[('far', AT.R), ('+urthe+', AT.I_COM)]]),
         (('best', 'RBS'), [[('well', AT.R), ('', AT.I_SUP)]]),
         (('worst', 'RBS'), [[('bad', AT.R), ('', AT.I_SUP)]]),
     ])
]


@pytest.mark.parametrize('data', data_inflection)
def test_inflection(en_morph_analyzer, data):
    input, expected = zip(*data)
    actual = tuple(en_morph_analyzer.analyze(token, pos) for token, pos in input)
    assert actual == expected


data_analyze_derivation = [
    ([
        # verb: 'fy'
        (([('fortify', AT.V)]), [('forty', AT.N), ('+ify', AT.V_FY)]),
        (([('glorify', AT.V)]), [('glory', AT.N), ('+ify', AT.V_FY)]),
        (([('terrify', AT.V)]), [('terror', AT.N), ('+ify', AT.V_FY)]),
        (([('qualify', AT.V)]), [('quality', AT.N), ('+ify', AT.V_FY)]),
        (([('simplify', AT.V)]), [('simple', AT.J), ('+ify', AT.V_FY)]),
        (([('beautify', AT.V)]), [('beauty', AT.N), ('+ify', AT.V_FY)]),
        (([('liquefy', AT.V)]), [('liquid', AT.N), ('+efy', AT.V_FY)]),

        # verb: 'ize'
        (([('hospitalize', AT.V)]), [('hospital', AT.N), ('+ize', AT.V_IZE)]),
        (([('oxidize', AT.V)]), [('oxide', AT.N), ('+ize', AT.V_IZE)]),
        (([('theorize', AT.V)]), [('theory', AT.N), ('+ize', AT.V_IZE)]),
        (([('sterilize', AT.V)]), [('sterile', AT.J), ('+ize', AT.V_IZE)]),
        (([('crystallize', AT.V)]), [('crystal', AT.N), ('+ize', AT.V_IZE)]),
        (([('dramatize', AT.V)]), [('dramatic', AT.J), ('+ize', AT.V_IZE)]),
        (([('barbarize', AT.V)]), [('barbary', AT.N), ('+ize', AT.V_IZE)]),

        # verb: 'en'
        (([('strengthen', AT.V)]), [('strength', AT.N), ('+en', AT.V_EN)]),
        (([('brighten', AT.V)]), [('bright', AT.J), ('+en', AT.V_EN)]),

        # noun: 'age'
        (([('marriage', AT.N)]), [('marry', AT.V), ('+iage', AT.N_AGE)]),
        (([('passage', AT.N)]), [('pass', AT.V), ('+age', AT.N_AGE)]),
        (([('mileage', AT.N)]), [('mile', AT.N), ('+age', AT.N_AGE)]),

        # noun: 'al'
        (([('denial', AT.N)]), [('deny', AT.V), ('+ial', AT.N_AL)]),
        (([('approval', AT.N)]), [('approve', AT.V), ('+al', AT.N_AL)]),

        # noun: 'ance'
        (([('defiance', AT.N)]), [('defy', AT.V), ('+iance', AT.N_ANCE)]),
        (([('annoyance', AT.N)]), [('annoy', AT.V), ('+ance', AT.N_ANCE)]),
        (([('insurance', AT.N)]), [('insure', AT.V), ('+ance', AT.N_ANCE)]),
        (([('admittance', AT.N)]), [('admit', AT.V), ('+ance', AT.N_ANCE)]),
        (([('relevance', AT.N)]), [('relevant', AT.J), ('+ance', AT.N_ANCE)]),
        (([('pregnancy', AT.N)]), [('pregnant', AT.J), ('+ancy', AT.N_ANCE)]),
        (([('difference', AT.N)]), [('differ', AT.V), ('+ent', AT.J_ANT), ('+ence', AT.N_ANCE)]),
        (([('fluency', AT.N)]), [('fluent', AT.J), ('+ency', AT.N_ANCE)]),
        (([('accuracy', AT.N)]), [('accurate', AT.J), ('+cy', AT.N_ANCE)]),

        # noun: 'ant'
        (([('applicant', AT.N)]), [('apply', AT.V), ('+icant', AT.N_ANT)]),
        (([('assistant', AT.N)]), [('assist', AT.V), ('+ant', AT.N_ANT)]),
        (([('propellant', AT.N)]), [('propel', AT.V), ('+ant', AT.N_ANT)]),
        (([('servant', AT.N)]), [('serve', AT.V), ('+ant', AT.N_ANT)]),
        (([('immigrant', AT.N)]), [('immigrate', AT.V), ('+ant', AT.N_ANT)]),
        (([('dependent', AT.N)]), [('depend', AT.V), ('+ent', AT.N_ANT)]),
        (([('resident', AT.N)]), [('reside', AT.V), ('+ent', AT.N_ANT)]),

        # noun: 'dom'
        (([('freedom', AT.N)]), [('free', AT.J), ('+dom', AT.N_DOM)]),
        (([('kingdom', AT.N)]), [('king', AT.N), ('+dom', AT.N_DOM)]),

        # noun: 'ee'
        (([('employee', AT.N)]), [('employ', AT.V), ('+ee', AT.N_EE)]),
        (([('escapee', AT.N)]), [('escape', AT.V), ('+ee', AT.N_EE)]),

        # noun: 'er'
        (([('carrier', AT.N)]), [('carry', AT.V), ('+ier', AT.N_ER)]),
        (([('cashier', AT.N)]), [('cash', AT.N), ('+ier', AT.N_ER)]),
        (([('financier', AT.N)]), [('finance', AT.N), ('+ier', AT.N_ER)]),
        (([('profiteer', AT.N)]), [('profit', AT.N), ('+eer', AT.N_ER)]),
        (([('bowyer', AT.N)]), [('bow', AT.V), ('+yer', AT.N_ER)]),
        (([('lawyer', AT.N)]), [('law', AT.N), ('+yer', AT.N_ER)]),
        (([('reader', AT.N)]), [('read', AT.V), ('+er', AT.N_ER)]),
        (([('runner', AT.N)]), [('run', AT.V), ('+er', AT.N_ER)]),
        (([('writer', AT.N)]), [('write', AT.V), ('+er', AT.N_ER)]),
        (([('engineer', AT.N)]), [('engine', AT.N), ('+er', AT.N_ER)]),
        (([('hatter', AT.N)]), [('hat', AT.V), ('+er', AT.N_ER)]),
        (([('tiler', AT.N)]), [('tile', AT.V), ('+er', AT.N_ER)]),
        (([('beggar', AT.N)]), [('beg', AT.V), ('+ar', AT.N_ER)]),
        (([('liar', AT.N)]), [('lie', AT.V), ('+ar', AT.N_ER)]),
        (([('actor', AT.N)]), [('act', AT.V), ('+or', AT.N_ER)]),
        (([('abator', AT.N)]), [('abate', AT.V), ('+or', AT.N_ER)]),

        # noun: 'hood'
        (([('likelihood', AT.N)]), [('like', AT.N), ('+ly', AT.J_LY), ('+ihood', AT.N_HOOD)]),
        (([('childhood', AT.N)]), [('child', AT.N), ('+hood', AT.N_HOOD)]),

        # adjective: 'ing'
        (([('building', AT.N)]), [('build', AT.V), ('+ing', AT.N_ING)]),

        # noun: 'ism'
        (([('witticism', AT.N)]), [('wit', AT.N), ('+y', AT.J_Y), ('+icism', AT.N_ISM)]),
        (([('baptism', AT.N)]), [('baptize', AT.V), ('+ism', AT.N_ISM)]),
        (([('capitalism', AT.N)]), [('capital', AT.N), ('+ize', AT.V_IZE), ('+ism', AT.N_ISM)]),
        (([('bimetallism', AT.N)]), [('bimetal', AT.N), ('+ism', AT.N_ISM)]),

        # noun: 'ist'
        (([('apologist', AT.N)]), [('apology', AT.N), ('+ist', AT.N_IST)]),
        (([('capitalist', AT.N)]), [('capital', AT.J), ('+ist', AT.N_IST)]),
        (([('machinist', AT.N)]), [('machine', AT.N), ('+ist', AT.N_IST)]),
        (([('panellist', AT.N)]), [('panel', AT.N), ('+ist', AT.N_IST)]),
        (([('environmentalist', AT.N)]), [('environ', AT.V), ('+ment', AT.N_MENT), ('+al', AT.J_AL), ('+ist', AT.N_IST)]),

        # noun: 'ity'
        (([('capability', AT.N)]), [('capable', AT.J), ('+ility', AT.N_ITY)]),
        (([('variety', AT.N)]), [('vary', AT.V), ('+ious', AT.J_OUS), ('+ety', AT.N_ITY)]),
        (([('normality', AT.N)]), [('norm', AT.N), ('+al', AT.J_AL), ('+ity', AT.N_ITY)]),
        (([('adversity', AT.N)]), [('adverse', AT.J), ('+ity', AT.N_ITY)]),
        (([('jollity', AT.N)]), [('jolly', AT.J), ('+ity', AT.N_ITY)]),
        (([('frivolity', AT.N)]), [('frivol', AT.V), ('+ous', AT.J_OUS), ('+ity', AT.N_ITY)]),
        (([('loyalty', AT.N)]), [('loyal', AT.J), ('+ty', AT.N_ITY)]),

        # noun: 'man'
        (([('chairman', AT.N)]), [('chair', AT.V), ('+man', AT.N_MAN)]),
        (([('chairwoman', AT.N)]), [('chair', AT.V), ('+woman', AT.N_MAN)]),
        (([('chairperson', AT.N)]), [('chair', AT.V), ('+person', AT.N_MAN)]),

        # noun: 'ment'
        (([('development', AT.N)]), [('develop', AT.V), ('+ment', AT.N_MENT)]),
        (([('abridgment', AT.N)]), [('abridge', AT.V), ('+ment', AT.N_MENT)]),

        # noun: 'ness'
        (([('happiness', AT.N)]), [('happy', AT.J), ('+iness', AT.N_NESS)]),
        (([('kindness', AT.N)]), [('kind', AT.J), ('+ness', AT.N_NESS)]),
        (([('thinness', AT.N)]), [('thin', AT.J), ('+ness', AT.N_NESS)]),

        # noun: 'ship'
        (([('friendship', AT.N)]), [('friend', AT.N), ('+ship', AT.N_SHIP)]),

        # noun: 'sis'
        (([('diagnosis', AT.N)]), [('diagnose', AT.V), ('+sis', AT.N_SIS)]),
        (([('analysis', AT.N)]), [('analyze', AT.V), ('+sis', AT.N_SIS)]),

        # noun: 'tion'
        (([('verification', AT.N)]), [('verify', AT.V), ('+ication', AT.N_TION)]),
        (([('flirtation', AT.N)]), [('flirt', AT.V), ('+ation', AT.N_TION)]),
        (([('admiration', AT.N)]), [('admire', AT.V), ('+ation', AT.N_TION)]),
        (([('suspicion', AT.N)]), [('suspect', AT.V), ('+icion', AT.N_TION)]),
        (([('addition', AT.N)]), [('add', AT.V), ('+ition', AT.N_TION)]),
        (([('extension', AT.N)]), [('extend', AT.V), ('+sion', AT.N_TION)]),
        (([('decision', AT.N)]), [('decide', AT.V), ('+sion', AT.N_TION)]),
        (([('introduction', AT.N)]), [('introduce', AT.V), ('+tion', AT.N_TION)]),
        (([('resurrection', AT.N)]), [('resurrect', AT.V), ('+ion', AT.N_TION)]),
        (([('alienation', AT.N)]), [('alien', AT.V), ('+ation', AT.N_TION)]),

        # adjective: 'able'
        (([('certifiable', AT.J)]), [('cert', AT.N), ('+ify', AT.V_FY), ('+iable', AT.J_ABLE)]),
        (([('readable', AT.J)]), [('read', AT.V), ('+able', AT.J_ABLE)]),
        (([('writable', AT.J)]), [('write', AT.V), ('+able', AT.J_ABLE)]),
        (([('irritable', AT.J)]), [('irritate', AT.V), ('+able', AT.J_ABLE)]),
        (([('flammable', AT.J)]), [('flam', AT.V), ('+able', AT.J_ABLE)]),
        (([('visible', AT.J)]), [('vision', AT.N), ('+ible', AT.J_ABLE)]),

        # adjective: 'al'
        (([('influential', AT.J)]), [('influence', AT.N), ('+tial', AT.J_AL)]),
        (([('colonial', AT.J)]), [('colony', AT.N), ('+ial', AT.J_AL)]),
        (([('accidental', AT.J)]), [('accident', AT.N), ('+al', AT.J_AL)]),
        (([('visceral', AT.J)]), [('viscera', AT.N), ('+al', AT.J_AL)]),
        (([('universal', AT.J)]), [('universe', AT.N), ('+al', AT.J_AL)]),
        (([('bacterial', AT.J)]), [('bacteria', AT.N), ('+al', AT.J_AL)]),
        (([('focal', AT.J)]), [('focus', AT.N), ('+al', AT.J_AL)]),
        (([('economical', AT.J)]), [('economy', AT.N), ('+ic', AT.J_IC), ('+al', AT.J_AL)]),

        # adjective: 'ant'
        (([('applicant', AT.J)]), [('apply', AT.V), ('+icant', AT.J_ANT)]),
        (([('relaxant', AT.J)]), [('relax', AT.V), ('+ant', AT.J_ANT)]),
        (([('propellant', AT.J)]), [('propel', AT.V), ('+ant', AT.J_ANT)]),
        (([('pleasant', AT.J)]), [('please', AT.V), ('+ant', AT.J_ANT)]),
        (([('dominant', AT.J)]), [('dominate', AT.V), ('+ant', AT.J_ANT)]),
        (([('absorbent', AT.J)]), [('absorb', AT.V), ('+ent', AT.J_ANT)]),
        (([('abhorrent', AT.J)]), [('abhor', AT.V), ('+ent', AT.J_ANT)]),
        (([('adherent', AT.J)]), [('adhere', AT.V), ('+ent', AT.J_ANT)]),

        # adjective: 'ary'
        (([('cautionary', AT.J)]), [('caution', AT.V), ('+ary', AT.J_ARY)]),
        (([('imaginary', AT.J)]), [('imagine', AT.V), ('+ary', AT.J_ARY)]),
        (([('pupillary', AT.J)]), [('pupil', AT.N), ('+ary', AT.J_ARY)]),
        (([('monetary', AT.J)]), [('money', AT.N), ('+tary', AT.J_ARY)]),

        # adjective: 'ed'
        (([('diffused', AT.J)]), [('diffuse', AT.V), ('+d', AT.J_ED)]),
        (([('shrunk', AT.J)]), [('shrink', AT.V), ('+u+', AT.J_ED)]),

        # adjective: 'ful'
        (([('beautiful', AT.J)]), [('beauty', AT.N), ('+iful', AT.J_FUL)]),
        (([('thoughtful', AT.J)]), [('thought', AT.N), ('+ful', AT.J_FUL)]),
        (([('helpful', AT.J)]), [('help', AT.V), ('+ful', AT.J_FUL)]),

        # adjective: 'ic'
        (([('realistic', AT.J)]), [('real', AT.N), ('+ize', AT.V_IZE), ('+stic', AT.J_IC)]),
        (([('fantastic', AT.J)]), [('fantasy', AT.N), ('+tic', AT.J_IC)]),
        (([('diagnostic', AT.J)]), [('diagnose', AT.V), ('+sis', AT.N_SIS), ('+tic', AT.J_IC)]),
        (([('analytic', AT.J)]), [('analyze', AT.V), ('+sis', AT.N_SIS), ('+tic', AT.J_IC)]),
        (([('poetic', AT.J)]), [('poet', AT.N), ('+ic', AT.J_IC)]),
        (([('metallic', AT.J)]), [('metal', AT.N), ('+ic', AT.J_IC)]),
        (([('sophomoric', AT.J)]), [('sophomore', AT.N), ('+ic', AT.J_IC)]),

        # adjective: 'ing'
        (([('dignifying', AT.J)]), [('dignity', AT.N), ('+ify', AT.V_FY), ('+ing', AT.J_ING)]),
        (([('abiding', AT.J)]), [('abide', AT.V), ('+ing', AT.J_ING)]),

        # adjective: 'ish'
        (([('bearish', AT.J)]), [('bear', AT.V), ('+ish', AT.J_ISH)]),
        (([('ticklish', AT.J)]), [('tickle', AT.V), ('+ish', AT.J_ISH)]),
        (([('reddish', AT.J)]), [('red', AT.V), ('+ish', AT.J_ISH)]),
        (([('boyish', AT.J)]), [('boy', AT.N), ('+ish', AT.J_ISH)]),
        (([('faddish', AT.J)]), [('fade', AT.V), ('+ish', AT.J_ISH)]),
        (([('mulish', AT.J)]), [('mule', AT.N), ('+ish', AT.J_ISH)]),

        # adjective: 'ive'
        (([('talkative', AT.J)]), [('talk', AT.V), ('+ative', AT.J_IVE)]),
        (([('adjudicative', AT.J)]), [('adjudicate', AT.V), ('+ative', AT.J_IVE)]),
        (([('destructive', AT.J)]), [('destruct', AT.V), ('+ive', AT.J_IVE)]),
        (([('defensive', AT.J)]), [('defense', AT.N), ('+ive', AT.J_IVE)]),
        (([('divisive', AT.J)]), [('divide', AT.V), ('+sion', AT.N_TION), ('+ive', AT.J_IVE)]),

        # adjective: 'less'
        (([('countless', AT.J)]), [('count', AT.V), ('+less', AT.J_LESS)]),
        (([('speechless', AT.J)]), [('speech', AT.N), ('+less', AT.J_LESS)]),

        # adjective: 'like'
        (([('childlike', AT.J)]), [('child', AT.N), ('+like', AT.J_LIKE)]),

        # adjective: 'ly'
        (([('daily', AT.J)]), [('day', AT.N), ('+ily', AT.J_LY)]),
        (([('weekly', AT.J)]), [('week', AT.N), ('+ly', AT.J_LY)]),

        # adjective: 'most'
        (([('innermost', AT.J)]), [('inner', AT.J), ('+most', AT.J_MOST)]),

        # adjective: 'ous'
        (([('courteous', AT.J)]), [('court', AT.N), ('+eous', AT.J_OUS)]),
        (([('glorious', AT.J)]), [('glory', AT.V), ('+ious', AT.J_OUS)]),
        (([('wondrous', AT.J)]), [('wonder', AT.N), ('+rous', AT.J_OUS)]),
        (([('marvellous', AT.J)]), [('marvel', AT.V), ('+ous', AT.J_OUS)]),
        (([('covetous', AT.J)]), [('covet', AT.V), ('+ous', AT.J_OUS)]),
        (([('nervous', AT.J)]), [('nerve', AT.V), ('+ous', AT.J_OUS)]),
        (([('cancerous', AT.J)]), [('cancer', AT.N), ('+ous', AT.J_OUS)]),
        (([('analogous', AT.J)]), [('analogy', AT.N), ('+ous', AT.J_OUS)]),
        (([('religious', AT.J)]), [('religion', AT.N), ('+ous', AT.J_OUS)]),

        # adjective: 'some'
        (([('worrisome', AT.J)]), [('worry', AT.N), ('+isome', AT.J_SOME)]),
        (([('troublesome', AT.J)]), [('trouble', AT.N), ('+some', AT.J_SOME)]),
        (([('awesome', AT.J)]), [('awe', AT.N), ('+some', AT.J_SOME)]),
        (([('fulsome', AT.J)]), [('full', AT.J), ('+some', AT.J_SOME)]),

        # adjective: 'wise'
        (([('clockwise', AT.J)]), [('clock', AT.N), ('+wise', AT.J_WISE)]),
        (([('likewise', AT.J)]), [('like', AT.J), ('+wise', AT.J_WISE)]),

        # adjective: 'y'
        (([('clayey', AT.J)]), [('clay', AT.N), ('+ey', AT.J_Y)]),
        (([('grouchy', AT.J)]), [('grouch', AT.V), ('+y', AT.J_Y)]),
        (([('runny', AT.J)]), [('run', AT.V), ('+y', AT.J_Y)]),
        (([('rumbly', AT.J)]), [('rumble', AT.V), ('+y', AT.J_Y)]),

        # adverb: 'ly'
        (([('electronically', AT.R)]), [('electron', AT.N), ('+ic', AT.J_IC), ('+ally', AT.R_LY)]),
        (([('easily', AT.R)]), [('ease', AT.V), ('+y', AT.J_Y), ('+ily', AT.R_LY)]),
        (([('sadly', AT.R)]), [('sad', AT.J), ('+ly', AT.R_LY)]),
        (([('fully', AT.R)]), [('full', AT.J), ('+ly', AT.R_LY)]),
        (([('incredibly', AT.R)]), [('incredible', AT.J), ('+ly', AT.R_LY)]),
     ])
]


@pytest.mark.parametrize('data', data_analyze_derivation)
def test_analyze_derivation(en_morph_analyzer, data):
    input, expected = zip(*data)
    actual = tuple(en_morph_analyzer._analyze_derivation(tp) for tp in input)
    assert actual == expected


data_inflection_derivation = [
    ([
        (('ownerships', None), [[('own', AT.V), ('+er', AT.N_ER), ('+ship', AT.N_SHIP), ('+s', AT.I_PLU)]]),
        (('offensiveness', None), [[('offense', AT.N), ('+ive', AT.J_IVE), ('+ness', AT.N_NESS)]]),
        (('chairmen', None), [[('chair', AT.V), ('+man', AT.N_MAN), ('+men', AT.I_PLU)]]),
        (('girlisher', None), [[('girl', AT.N), ('+ish', AT.J_ISH), ('+er', AT.I_COM)]]),
        (('environmentalist', None), [[('environ', AT.V), ('+ment', AT.N_MENT), ('+al', AT.J_AL), ('+ist', AT.N_IST)]]),
        (('economically', None), [[('economy', AT.N), ('+ic', AT.J_IC), ('+ally', AT.R_LY)]]),
        (('beautifulliest', None), [[('beauty', AT.N), ('+iful', AT.J_FUL), ('+ly', AT.R_LY), ('+iest', AT.I_SUP)]]),
     ])
]


@pytest.mark.parametrize('data', data_inflection_derivation)
def test_data_inflection_derivation(en_morph_analyzer, data):
    input, expected = zip(*data)
    actual = tuple(en_morph_analyzer.analyze(token, pos, True) for token, pos in input)
    assert actual == expected