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
import pytest

from elit.nlp.morph_analyzer import EnglishMorphTag as MT
from elit.nlp.morph_analyzer import extract_suffix

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


# ========================= analyze_base_lookup =========================

data_analyze_base_lookup = [
    ([
        (('', 'VB'), [('', MT.VB)]),
        (('', 'VBP'), [('', MT.VB)]),
        (('', 'NN'), [('', MT.NN)]),
        (('', 'NNP'), [('', MT.NN)]),
        (('', 'JJ'), [('', MT.JJ)]),
        (('', 'RB'), [('', MT.RB)]),
        (('', 'VBZ'), None),
    ])
]


@pytest.mark.parametrize('data', data_analyze_base_lookup)
def test_analyze_base_lookup(en_morph_analyzer, data):
    input, expected = zip(*data)
    actual = tuple(en_morph_analyzer._analyze_base_lookup(token, pos) for token, pos in input)
    assert actual == expected


# ========================= analyze_inflection_lookup =========================

data_analyze_inflection_lookup = [
    ([
        (('ai', 'VBP'), [('be', MT.VB)]),
        (('was', 'VBD'), [('be', MT.VB), ('', MT.I_3PS), ('', MT.I_PST)]),
        (("'d", 'VBD'), [('have', MT.VB), ('+d', MT.I_PST)]),
        (("'d", 'MD'), [('would', MT.MD)]),
        (("ai", 'NN'), None),
    ])
]


@pytest.mark.parametrize('data', data_analyze_inflection_lookup)
def test_analyze_inflection_lookup(en_morph_analyzer, data):
    input, expected = zip(*data)
    actual = tuple(en_morph_analyzer._analyze_inflection_lookup(token, pos) for token, pos in input)
    assert actual == expected


# ========================= analyze_inflection_rules =========================

data_analyze_inflection_rules = [
    ([
        # verb: 3rd-person singular
        (('studies', 'VBZ'), [('study', MT.VB), ('+ies', MT.I_3PS)]),
        (('pushes', 'VBZ'), [('push', MT.VB), ('+es', MT.I_3PS)]),
        (('takes', 'VBZ'), [('take', MT.VB), ('+s', MT.I_3PS)]),

        # verb: gerund
        (('lying', 'VBG'), [('lie', MT.VB), ('+ying', MT.I_GRD)]),
        (('feeling', 'VBG'), [('feel', MT.VB), ('+ing', MT.I_GRD)]),
        (('taking', 'VBG'), [('take', MT.VB), ('+ing', MT.I_GRD)]),
        (('running', 'VBG'), [('run', MT.VB), ('+ing', MT.I_GRD)]),

        # verb: past (participle)
        (('denied', 'VBD'), [('deny', MT.VB), ('+ied', MT.I_PST)]),
        (('entered', 'VBD'), [('enter', MT.VB), ('+ed', MT.I_PST)]),
        (('zipped', 'VBD'), [('zip', MT.VB), ('+ed', MT.I_PST)]),
        (('heard', 'VBD'), [('hear', MT.VB), ('+d', MT.I_PST)]),
        (('fallen', 'VBN'), [('fall', MT.VB), ('+en', MT.I_PST)]),
        (('written', 'VBN'), [('write', MT.VB), ('+en', MT.I_PST)]),
        (('drawn', 'VBN'), [('draw', MT.VB), ('+n', MT.I_PST)]),
        (('clung', 'VBN'), [('cling', MT.VB), ('+ung', MT.I_PST)]),

        # verb: irregular
        (('bit', 'VBD'), [('bite', MT.VB), ('-e', MT.I_PST)]),
        (('bites', 'VBZ'), [('bite', MT.VB), ('+s', MT.I_3PS)]),
        (('biting', 'VBG'), [('bite', MT.VB), ('+ing', MT.I_GRD)]),
        (('bitted', 'VBD'), [('bit', MT.VB), ('+ed', MT.I_PST)]),
        (('bitten', 'VBN'), [('bite', MT.VB), ('+en', MT.I_PST)]),
        (('bitting', 'VBG'), [('bit', MT.VB), ('+ing', MT.I_GRD)]),
        (('bound', 'VBD'), [('bind', MT.VB), ('+ou+', MT.I_PST)]),
        (('chivvies', 'VBZ'), [('chivy', MT.VB), ('+ies', MT.I_3PS)]),
        (('took', 'VBD'), [('take', MT.VB), ('+ook', MT.I_PST)]),
        (('slept', 'VBD'), [('sleep', MT.VB), ('+pt', MT.I_PST)]),
        (('spoken', 'VBN'), [('speak', MT.VB), ('+oken', MT.I_PST)]),
        (('woken', 'VBN'), [('wake', MT.VB), ('+oken', MT.I_PST)]),

        # noun: plural
        (('studies', 'NNS'), [('study', MT.NN), ('+ies', MT.I_PLR)]),
        (('crosses', 'NNS'), [('cross', MT.NN), ('+es', MT.I_PLR)]),
        (('areas', 'NNS'), [('area', MT.NN), ('+s', MT.I_PLR)]),
        (('men', 'NNS'), [('man', MT.NN), ('+men', MT.I_PLR)]),
        (('vertebrae', 'NNS'), [('vertebra', MT.NN), ('+ae', MT.I_PLR)]),
        (('foci', 'NNS'), [('focus', MT.NN), ('+i', MT.I_PLR)]),
        (('optima', 'NNS'), [('optimum', MT.NN), ('+a', MT.I_PLR)]),

        # noun: irregular
        (('indices', 'NNS'), [('index', MT.NN), ('+ices', MT.I_PLR)]),
        (('wolves', 'NNS'), [('wolf', MT.NN), ('+ves', MT.I_PLR)]),
        (('knives', 'NNS'), [('knife', MT.NN), ('+ves', MT.I_PLR)]),
        (('quizzes', 'NNS'), [('quiz', MT.NN), ('+es', MT.I_PLR)]),

        # adjective: comparative
        (('easier', 'JJR'), [('easy', MT.JJ), ('+ier', MT.I_COM)]),
        (('larger', 'JJR'), [('large', MT.JJ), ('+er', MT.I_COM)]),
        (('smaller', 'JJR'), [('small', MT.JJ), ('+er', MT.I_COM)]),
        (('bigger', 'JJR'), [('big', MT.JJ), ('+er', MT.I_COM)]),

        # adjective: superlative
        (('easiest', 'JJS'), [('easy', MT.JJ), ('+iest', MT.I_SUP)]),
        (('largest', 'JJS'), [('large', MT.JJ), ('+est', MT.I_SUP)]),
        (('smallest', 'JJS'), [('small', MT.JJ), ('+est', MT.I_SUP)]),
        (('biggest', 'JJS'), [('big', MT.JJ), ('+est', MT.I_SUP)]),

        # adjective: irregular
        (('cagier', 'JJR'), [('cagey', MT.JJ), ('+ier', MT.I_COM)]),
        (('worse', 'JJR'), [('bad', MT.JJ), ('', MT.I_COM)]),

        # adverb: comparative
        (('earlier', 'RBR'), [('early', MT.RB), ('+ier', MT.I_COM)]),
        (('sooner', 'RBR'), [('soon', MT.RB), ('+er', MT.I_COM)]),
        (('larger', 'RBR'), [('large', MT.RB), ('+er', MT.I_COM)]),

        # adverb: superlative
        (('earliest', 'RBS'), [('early', MT.RB), ('+iest', MT.I_SUP)]),
        (('soonest', 'RBS'), [('soon', MT.RB), ('+est', MT.I_SUP)]),
        (('largest', 'RBS'), [('large', MT.RB), ('+est', MT.I_SUP)]),

        # adverb: irregular
        (('further', 'RBR'), [('far', MT.RB), ('+urthe+', MT.I_COM)]),
        (('best', 'RBS'), [('well', MT.RB), ('', MT.I_SUP)]),
        (('worst', 'RBS'), [('bad', MT.RB), ('', MT.I_SUP)]),
    ])
]


@pytest.mark.parametrize('data', data_analyze_inflection_rules)
def test_analyze_inflection_rules(en_morph_analyzer, data):
    input, expected = zip(*data)
    actual = tuple(en_morph_analyzer._analyze_inflection_rules(token, pos) for token, pos in input)
    assert actual == expected


# ========================= analyze_prefix =========================

data_analyze_prefix = [
    ([
        (('repeat', MT.VB), [('repeat', MT.VB)]),
        (('transcribe', MT.VB), [('scribe', MT.VB), ('tran+', MT.P)]),
        (('belittle', MT.VB), [('little', MT.JJ), ('be+', MT.P)]),
        (('anemic', MT.JJ), [('emic', MT.JJ), ('an+', MT.P)]),
        (('co-founder', MT.NN), [('founder', MT.NN), ('co+', MT.P)]),
        (('super-overlook', MT.VB), [('look', MT.VB), ('super+', MT.P), ('over+', MT.P)]),
        (('deuteragonist', MT.NN), [('agonist', MT.NN), ('deuter+', MT.P)]),
        (('be-deuteragonist', MT.NN), [('agonist', MT.NN), ('be+', MT.P), ('deuter+', MT.P)]),
    ])
]


@pytest.mark.parametrize('data', data_analyze_prefix)
def test_analyze_prefix(en_morph_analyzer, data):
    input, expected = zip(*data)
    actual = tuple(en_morph_analyzer._analyze_prefix(token, pos) for token, pos in input)
    assert actual == expected


# ========================= analyze_prefix =========================

data_analyze_derivation_rules = [
    ([
        # verb: 'fy'
        (('fortify', MT.VB), [('forty', MT.NN), ('+ify', MT.V_FY)]),
        (('glorify', MT.VB), [('glory', MT.NN), ('+ify', MT.V_FY)]),
        (('terrify', MT.VB), [('terror', MT.NN), ('+ify', MT.V_FY)]),
        (('qualify', MT.VB), [('quality', MT.NN), ('+ify', MT.V_FY)]),
        (('simplify', MT.VB), [('simple', MT.JJ), ('+ify', MT.V_FY)]),
        (('beautify', MT.VB), [('beauty', MT.NN), ('+ify', MT.V_FY)]),
        (('liquefy', MT.VB), [('liquid', MT.NN), ('+efy', MT.V_FY)]),

        # verb: 'ize'
        (('hospitalize', MT.VB), [('hospital', MT.NN), ('+ize', MT.V_IZE)]),
        (('oxidize', MT.VB), [('oxide', MT.NN), ('+ize', MT.V_IZE)]),
        (('theorize', MT.VB), [('theory', MT.NN), ('+ize', MT.V_IZE)]),
        (('sterilize', MT.VB), [('sterile', MT.JJ), ('+ize', MT.V_IZE)]),
        (('crystallize', MT.VB), [('crystal', MT.NN), ('+ize', MT.V_IZE)]),
        (('dramatize', MT.VB), [('drama', MT.NN), ('+tic', MT.J_IC), ('+ize', MT.V_IZE)]),
        (('barbarize', MT.VB), [('barbary', MT.NN), ('+ize', MT.V_IZE)]),

        # verb: 'en'
        (('strengthen', MT.VB), [('strength', MT.NN), ('+en', MT.V_EN)]),
        (('brighten', MT.VB), [('bright', MT.JJ), ('+en', MT.V_EN)]),

        # noun: 'age'
        (('marriage', MT.NN), [('marry', MT.VB), ('+iage', MT.N_AGE)]),
        (('passage', MT.NN), [('pass', MT.VB), ('+age', MT.N_AGE)]),
        (('mileage', MT.NN), [('mile', MT.NN), ('+age', MT.N_AGE)]),

        # noun: 'al'
        (('denial', MT.NN), [('deny', MT.VB), ('+ial', MT.N_AL)]),
        (('approval', MT.NN), [('approve', MT.VB), ('+al', MT.N_AL)]),

        # noun: 'ance'
        (('defiance', MT.NN), [('defy', MT.VB), ('+iance', MT.N_ANCE)]),
        (('annoyance', MT.NN), [('annoy', MT.VB), ('+ance', MT.N_ANCE)]),
        (('insurance', MT.NN), [('insure', MT.VB), ('+ance', MT.N_ANCE)]),
        (('admittance', MT.NN), [('admit', MT.VB), ('+ance', MT.N_ANCE)]),
        (('relevance', MT.NN), [('relevant', MT.JJ), ('+ance', MT.N_ANCE)]),
        (('pregnancy', MT.NN), [('pregnant', MT.JJ), ('+ancy', MT.N_ANCE)]),
        (('difference', MT.NN), [('differ', MT.VB), ('+ent', MT.J_ANT), ('+ence', MT.N_ANCE)]),
        (('fluency', MT.NN), [('fluent', MT.JJ), ('+ency', MT.N_ANCE)]),
        (('accuracy', MT.NN), [('accurate', MT.JJ), ('+cy', MT.N_ANCE)]),

        # noun: 'ant'
        (('applicant', MT.NN), [('apply', MT.VB), ('+icant', MT.N_ANT)]),
        (('assistant', MT.NN), [('assist', MT.VB), ('+ant', MT.N_ANT)]),
        (('propellant', MT.NN), [('propel', MT.VB), ('+ant', MT.N_ANT)]),
        (('servant', MT.NN), [('serve', MT.VB), ('+ant', MT.N_ANT)]),
        (('immigrant', MT.NN), [('immigrate', MT.VB), ('+ant', MT.N_ANT)]),
        (('dependent', MT.NN), [('depend', MT.VB), ('+ent', MT.N_ANT)]),
        (('resident', MT.NN), [('reside', MT.VB), ('+ent', MT.N_ANT)]),

        # noun: 'dom'
        (('freedom', MT.NN), [('free', MT.JJ), ('+dom', MT.N_DOM)]),
        (('kingdom', MT.NN), [('king', MT.NN), ('+dom', MT.N_DOM)]),

        # noun: 'ee'
        (('employee', MT.NN), [('employ', MT.VB), ('+ee', MT.N_EE)]),
        (('escapee', MT.NN), [('escape', MT.VB), ('+ee', MT.N_EE)]),

        # noun: 'er'
        (('carrier', MT.NN), [('carry', MT.VB), ('+ier', MT.N_ER)]),
        (('cashier', MT.NN), [('cash', MT.NN), ('+ier', MT.N_ER)]),
        (('financier', MT.NN), [('finance', MT.NN), ('+ier', MT.N_ER)]),
        (('profiteer', MT.NN), [('profit', MT.NN), ('+eer', MT.N_ER)]),
        (('bowyer', MT.NN), [('bow', MT.VB), ('+yer', MT.N_ER)]),
        (('lawyer', MT.NN), [('law', MT.NN), ('+yer', MT.N_ER)]),
        (('reader', MT.NN), [('read', MT.VB), ('+er', MT.N_ER)]),
        (('runner', MT.NN), [('run', MT.VB), ('+er', MT.N_ER)]),
        (('writer', MT.NN), [('write', MT.VB), ('+er', MT.N_ER)]),
        (('engineer', MT.NN), [('engine', MT.NN), ('+er', MT.N_ER)]),
        (('hatter', MT.NN), [('hat', MT.VB), ('+er', MT.N_ER)]),
        (('tiler', MT.NN), [('tile', MT.VB), ('+er', MT.N_ER)]),
        (('beggar', MT.NN), [('beg', MT.VB), ('+ar', MT.N_ER)]),
        (('liar', MT.NN), [('lie', MT.VB), ('+ar', MT.N_ER)]),
        (('actor', MT.NN), [('act', MT.VB), ('+or', MT.N_ER)]),
        (('abator', MT.NN), [('abate', MT.VB), ('+or', MT.N_ER)]),

        # noun: 'hood'
        (('likelihood', MT.NN), [('like', MT.NN), ('+ly', MT.J_LY), ('+ihood', MT.N_HOOD)]),
        (('childhood', MT.NN), [('child', MT.NN), ('+hood', MT.N_HOOD)]),

        # adjective: 'ing'
        (('building', MT.NN), [('build', MT.VB), ('+ing', MT.N_ING)]),

        # noun: 'ism'
        (('witticism', MT.NN), [('wit', MT.NN), ('+y', MT.J_Y), ('+icism', MT.N_ISM)]),
        (('baptism', MT.NN), [('baptize', MT.VB), ('+ism', MT.N_ISM)]),
        (('capitalism', MT.NN), [('capital', MT.NN), ('+ize', MT.V_IZE), ('+ism', MT.N_ISM)]),
        (('bimetallism', MT.NN), [('bimetal', MT.NN), ('+ism', MT.N_ISM)]),

        # noun: 'ist'
        (('apologist', MT.NN), [('apology', MT.NN), ('+ist', MT.N_IST)]),
        (('capitalist', MT.NN), [('capital', MT.JJ), ('+ist', MT.N_IST)]),
        (('machinist', MT.NN), [('machine', MT.NN), ('+ist', MT.N_IST)]),
        (('panellist', MT.NN), [('panel', MT.NN), ('+ist', MT.N_IST)]),
        (('environmentalist', MT.NN), [('environ', MT.VB), ('+ment', MT.N_MENT), ('+al', MT.J_AL), ('+ist', MT.N_IST)]),

        # noun: 'ity'
        (('capability', MT.NN), [('capable', MT.JJ), ('+ility', MT.N_ITY)]),
        (('variety', MT.NN), [('vary', MT.VB), ('+ious', MT.J_OUS), ('+ety', MT.N_ITY)]),
        (('normality', MT.NN), [('norm', MT.NN), ('+al', MT.J_AL), ('+ity', MT.N_ITY)]),
        (('adversity', MT.NN), [('adverse', MT.JJ), ('+ity', MT.N_ITY)]),
        (('jollity', MT.NN), [('jolly', MT.JJ), ('+ity', MT.N_ITY)]),
        (('frivolity', MT.NN), [('frivol', MT.VB), ('+ous', MT.J_OUS), ('+ity', MT.N_ITY)]),
        (('loyalty', MT.NN), [('loyal', MT.JJ), ('+ty', MT.N_ITY)]),

        # noun: 'man'
        (('chairman', MT.NN), [('chair', MT.VB), ('+man', MT.N_MAN)]),
        (('chairwoman', MT.NN), [('chair', MT.VB), ('+woman', MT.N_MAN)]),
        (('chairperson', MT.NN), [('chair', MT.VB), ('+person', MT.N_MAN)]),

        # noun: 'ment'
        (('development', MT.NN), [('develop', MT.VB), ('+ment', MT.N_MENT)]),
        (('abridgment', MT.NN), [('abridge', MT.VB), ('+ment', MT.N_MENT)]),

        # noun: 'ness'
        (('happiness', MT.NN), [('happy', MT.JJ), ('+iness', MT.N_NESS)]),
        (('kindness', MT.NN), [('kind', MT.JJ), ('+ness', MT.N_NESS)]),
        (('thinness', MT.NN), [('thin', MT.JJ), ('+ness', MT.N_NESS)]),

        # noun: 'ship'
        (('friendship', MT.NN), [('friend', MT.NN), ('+ship', MT.N_SHIP)]),

        # noun: 'sis'
        (('diagnosis', MT.NN), [('diagnose', MT.VB), ('+sis', MT.N_SIS)]),
        (('analysis', MT.NN), [('analyze', MT.VB), ('+sis', MT.N_SIS)]),

        # noun: 'tion'
        (('verification', MT.NN), [('verify', MT.VB), ('+ication', MT.N_TION)]),
        (('flirtation', MT.NN), [('flirt', MT.VB), ('+ation', MT.N_TION)]),
        (('admiration', MT.NN), [('admire', MT.VB), ('+ation', MT.N_TION)]),
        (('suspicion', MT.NN), [('suspect', MT.VB), ('+icion', MT.N_TION)]),
        (('addition', MT.NN), [('add', MT.VB), ('+ition', MT.N_TION)]),
        (('extension', MT.NN), [('extend', MT.VB), ('+sion', MT.N_TION)]),
        (('decision', MT.NN), [('decide', MT.VB), ('+sion', MT.N_TION)]),
        (('introduction', MT.NN), [('introduce', MT.VB), ('+tion', MT.N_TION)]),
        (('resurrection', MT.NN), [('resurrect', MT.VB), ('+ion', MT.N_TION)]),
        (('alienation', MT.NN), [('alien', MT.VB), ('+ation', MT.N_TION)]),

        # adjective: 'able'
        (('certifiable', MT.JJ), [('cert', MT.NN), ('+ify', MT.V_FY), ('+iable', MT.J_ABLE)]),
        (('readable', MT.JJ), [('read', MT.VB), ('+able', MT.J_ABLE)]),
        (('writable', MT.JJ), [('write', MT.VB), ('+able', MT.J_ABLE)]),
        (('irritable', MT.JJ), [('irritate', MT.VB), ('+able', MT.J_ABLE)]),
        (('flammable', MT.JJ), [('flam', MT.VB), ('+able', MT.J_ABLE)]),
        (('visible', MT.JJ), [('vision', MT.NN), ('+ible', MT.J_ABLE)]),

        # adjective: 'al'
        (('influential', MT.JJ), [('influence', MT.NN), ('+tial', MT.J_AL)]),
        (('colonial', MT.JJ), [('colony', MT.NN), ('+ial', MT.J_AL)]),
        (('accidental', MT.JJ), [('accident', MT.NN), ('+al', MT.J_AL)]),
        (('visceral', MT.JJ), [('viscera', MT.NN), ('+al', MT.J_AL)]),
        (('universal', MT.JJ), [('universe', MT.NN), ('+al', MT.J_AL)]),
        (('bacterial', MT.JJ), [('bacteria', MT.NN), ('+al', MT.J_AL)]),
        (('focal', MT.JJ), [('focus', MT.NN), ('+al', MT.J_AL)]),
        (('economical', MT.JJ), [('economy', MT.NN), ('+ic', MT.J_IC), ('+al', MT.J_AL)]),

        # adjective: 'ant'
        (('applicant', MT.JJ), [('apply', MT.VB), ('+icant', MT.J_ANT)]),
        (('relaxant', MT.JJ), [('relax', MT.VB), ('+ant', MT.J_ANT)]),
        (('propellant', MT.JJ), [('propel', MT.VB), ('+ant', MT.J_ANT)]),
        (('pleasant', MT.JJ), [('please', MT.VB), ('+ant', MT.J_ANT)]),
        (('dominant', MT.JJ), [('dominate', MT.VB), ('+ant', MT.J_ANT)]),
        (('absorbent', MT.JJ), [('absorb', MT.VB), ('+ent', MT.J_ANT)]),
        (('abhorrent', MT.JJ), [('abhor', MT.VB), ('+ent', MT.J_ANT)]),
        (('adherent', MT.JJ), [('adhere', MT.VB), ('+ent', MT.J_ANT)]),

        # adjective: 'ary'
        (('cautionary', MT.JJ), [('caution', MT.VB), ('+ary', MT.J_ARY)]),
        (('imaginary', MT.JJ), [('imagine', MT.VB), ('+ary', MT.J_ARY)]),
        (('pupillary', MT.JJ), [('pupil', MT.NN), ('+ary', MT.J_ARY)]),
        (('monetary', MT.JJ), [('money', MT.NN), ('+tary', MT.J_ARY)]),

        # adjective: 'ed'
        (('diffused', MT.JJ), [('diffuse', MT.VB), ('+d', MT.J_ED)]),
        (('shrunk', MT.JJ), [('shrink', MT.VB), ('+u+', MT.J_ED)]),

        # adjective: 'ful'
        (('beautiful', MT.JJ), [('beauty', MT.NN), ('+iful', MT.J_FUL)]),
        (('thoughtful', MT.JJ), [('thought', MT.NN), ('+ful', MT.J_FUL)]),
        (('helpful', MT.JJ), [('help', MT.VB), ('+ful', MT.J_FUL)]),

        # adjective: 'ic'
        (('realistic', MT.JJ), [('real', MT.NN), ('+ize', MT.V_IZE), ('+stic', MT.J_IC)]),
        (('fantastic', MT.JJ), [('fantasy', MT.NN), ('+tic', MT.J_IC)]),
        (('diagnostic', MT.JJ), [('diagnose', MT.VB), ('+sis', MT.N_SIS), ('+tic', MT.J_IC)]),
        (('analytic', MT.JJ), [('analyze', MT.VB), ('+sis', MT.N_SIS), ('+tic', MT.J_IC)]),
        (('poetic', MT.JJ), [('poet', MT.NN), ('+ic', MT.J_IC)]),
        (('metallic', MT.JJ), [('metal', MT.NN), ('+ic', MT.J_IC)]),
        (('sophomoric', MT.JJ), [('sophomore', MT.NN), ('+ic', MT.J_IC)]),

        # adjective: 'ing'
        (('dignifying', MT.JJ), [('dignity', MT.NN), ('+ify', MT.V_FY), ('+ing', MT.J_ING)]),
        (('abiding', MT.JJ), [('abide', MT.VB), ('+ing', MT.J_ING)]),

        # adjective: 'ish'
        (('bearish', MT.JJ), [('bear', MT.VB), ('+ish', MT.J_ISH)]),
        (('ticklish', MT.JJ), [('tickle', MT.VB), ('+ish', MT.J_ISH)]),
        (('reddish', MT.JJ), [('red', MT.VB), ('+ish', MT.J_ISH)]),
        (('boyish', MT.JJ), [('boy', MT.NN), ('+ish', MT.J_ISH)]),
        (('faddish', MT.JJ), [('fade', MT.VB), ('+ish', MT.J_ISH)]),
        (('mulish', MT.JJ), [('mule', MT.NN), ('+ish', MT.J_ISH)]),

        # adjective: 'ive'
        (('talkative', MT.JJ), [('talk', MT.VB), ('+ative', MT.J_IVE)]),
        (('adjudicative', MT.JJ), [('adjudicate', MT.VB), ('+ative', MT.J_IVE)]),
        (('destructive', MT.JJ), [('destruct', MT.VB), ('+ive', MT.J_IVE)]),
        (('defensive', MT.JJ), [('defense', MT.NN), ('+ive', MT.J_IVE)]),
        (('divisive', MT.JJ), [('divide', MT.VB), ('+sion', MT.N_TION), ('+ive', MT.J_IVE)]),

        # adjective: 'less'
        (('countless', MT.JJ), [('count', MT.VB), ('+less', MT.J_LESS)]),
        (('speechless', MT.JJ), [('speech', MT.NN), ('+less', MT.J_LESS)]),

        # adjective: 'like'
        (('childlike', MT.JJ), [('child', MT.NN), ('+like', MT.J_LIKE)]),

        # adjective: 'ly'
        (('daily', MT.JJ), [('day', MT.NN), ('+ily', MT.J_LY)]),
        (('weekly', MT.JJ), [('week', MT.NN), ('+ly', MT.J_LY)]),

        # adjective: 'most'
        (('innermost', MT.JJ), [('inner', MT.JJ), ('+most', MT.J_MOST)]),

        # adjective: 'ous'
        (('courteous', MT.JJ), [('court', MT.NN), ('+eous', MT.J_OUS)]),
        (('glorious', MT.JJ), [('glory', MT.VB), ('+ious', MT.J_OUS)]),
        (('wondrous', MT.JJ), [('wonder', MT.NN), ('+rous', MT.J_OUS)]),
        (('marvellous', MT.JJ), [('marvel', MT.VB), ('+ous', MT.J_OUS)]),
        (('covetous', MT.JJ), [('covet', MT.VB), ('+ous', MT.J_OUS)]),
        (('nervous', MT.JJ), [('nerve', MT.VB), ('+ous', MT.J_OUS)]),
        (('cancerous', MT.JJ), [('cancer', MT.NN), ('+ous', MT.J_OUS)]),
        (('analogous', MT.JJ), [('analogy', MT.NN), ('+ous', MT.J_OUS)]),
        (('religious', MT.JJ), [('religion', MT.NN), ('+ous', MT.J_OUS)]),

        # adjective: 'some'
        (('worrisome', MT.JJ), [('worry', MT.NN), ('+isome', MT.J_SOME)]),
        (('troublesome', MT.JJ), [('trouble', MT.NN), ('+some', MT.J_SOME)]),
        (('awesome', MT.JJ), [('awe', MT.NN), ('+some', MT.J_SOME)]),
        (('fulsome', MT.JJ), [('full', MT.JJ), ('+some', MT.J_SOME)]),

        # adjective: 'wise'
        (('clockwise', MT.JJ), [('clock', MT.NN), ('+wise', MT.J_WISE)]),
        (('likewise', MT.JJ), [('like', MT.JJ), ('+wise', MT.J_WISE)]),

        # adjective: 'y'
        (('clayey', MT.JJ), [('clay', MT.NN), ('+ey', MT.J_Y)]),
        (('grouchy', MT.JJ), [('grouch', MT.VB), ('+y', MT.J_Y)]),
        (('runny', MT.JJ), [('run', MT.VB), ('+y', MT.J_Y)]),
        (('rumbly', MT.JJ), [('rumble', MT.VB), ('+y', MT.J_Y)]),

        # adverb: 'ly'
        (('electronically', MT.RB), [('electron', MT.NN), ('+ic', MT.J_IC), ('+ally', MT.R_LY)]),
        (('easily', MT.RB), [('ease', MT.VB), ('+y', MT.J_Y), ('+ily', MT.R_LY)]),
        (('sadly', MT.RB), [('sad', MT.JJ), ('+ly', MT.R_LY)]),
        (('fully', MT.RB), [('full', MT.JJ), ('+ly', MT.R_LY)]),
        (('incredibly', MT.RB), [('incredible', MT.JJ), ('+ly', MT.R_LY)]),
    ])
]


@pytest.mark.parametrize('data', data_analyze_derivation_rules)
def test_analyze_derivation_rules(en_morph_analyzer, data):
    input, expected = zip(*data)
    actual = tuple(en_morph_analyzer._analyze_derivation_rules(token, pos) for token, pos in input)
    assert actual == expected


data_analyze = [
    ([
        (('ownerships', 'NNS'), [('own', 'VB'), ('+er', 'N_ER'), ('+ship', 'N_SHIP'), ('+s', 'I_PLR')]),
        (('offensiveness', 'NN'), [('offense', 'NN'), ('+ive', 'J_IVE'), ('+ness', 'N_NESS')]),
        (('overachievers', 'NNS'), [('achieve', 'VB'), ('over+', 'P'), ('+er', 'N_ER'), ('+s', 'I_PLR')]),
        (('chairmen', 'NNS'), [('chair', 'VB'), ('+man', 'N_MAN'), ('+men', 'I_PLR')]),
        (('girlisher', 'JJR'), [('girl', 'NN'), ('+ish', 'J_ISH'), ('+er', 'I_COM')]),
        (('environmentalists', 'NNS'), [('environ', 'VB'), ('+ment', 'N_MENT'), ('+al', 'J_AL'), ('+ist', 'N_IST'), ('+s', 'I_PLR')]),
        (('beautifulliest', 'RBS'), [('beauty', 'NN'), ('+iful', 'J_FUL'), ('+ly', 'R_LY'), ('+iest', 'I_SUP')]),
    ])
]


@pytest.mark.parametrize('data', data_analyze)
def test_analyze(en_morph_analyzer, data):
    input, expected = zip(*data)
    actual = tuple(en_morph_analyzer.analyze(token, pos, prefix=1) for token, pos in input)
    assert actual == expected
