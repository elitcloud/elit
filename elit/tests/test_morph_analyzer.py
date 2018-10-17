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
from elit.morph_analyzer import MorphTag as MT
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
        (('was', None), [[('be', EM.V), ('', MT.I_3PS), ('', MT.I_PAS)]]),
        (("'d", None), [[('have', EM.V), ('+d', MT.I_PAS)], [('would', EM.M)]]),
    ])
]


@pytest.mark.parametrize('data', data_irregular)
def test_irregular(en_morph_analyzer, data):
    input, expected = zip(*data)
    actual = tuple(en_morph_analyzer.analyze(token, pos) for token, pos in input)
    assert actual == expected


data_base = [
    ([
        (('study', 'VB'), [[('study', EM.V)]]),
        (('bound', 'JJ'), [[('bound', EM.J)]]),
        (('Jinho', 'NNP'), [[('jinho', EM.N)]]),
        (('study', None), [[('study', EM.V)], [('study', EM.N)]]),
        (('bound', None), [[('bound', EM.V)], [('bind', EM.V), ('+ou+', MT.I_PAS)], [('bound', EM.N)], [('bound', EM.J)]]),
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
         (('studies', 'VBZ'), [[('study', EM.V), ('+ies', MT.I_3PS)]]),
         (('pushes', 'VBZ'), [[('push', EM.V), ('+es', MT.I_3PS)]]),
         (('takes', 'VBZ'), [[('take', EM.V), ('+s', MT.I_3PS)]]),

         # verb: gerund
         (('lying', 'VBG'), [[('lie', EM.V), ('+ying', MT.I_GER)]]),
         (('feeling', 'VBG'), [[('feel', EM.V), ('+ing', MT.I_GER)]]),
         (('taking', 'VBG'), [[('take', EM.V), ('+ing', MT.I_GER)]]),
         (('running', 'VBG'), [[('run', EM.V), ('+ing', MT.I_GER)]]),

         # verb: past (participle)
         (('denied', 'VBD'), [[('deny', EM.V), ('+ied', MT.I_PAS)]]),
         (('entered', 'VBD'), [[('enter', EM.V), ('+ed', MT.I_PAS)]]),
         (('zipped', 'VBD'), [[('zip', EM.V), ('+ed', MT.I_PAS)]]),
         (('heard', 'VBD'), [[('hear', EM.V), ('+d', MT.I_PAS)]]),
         (('fallen', 'VBN'), [[('fall', EM.V), ('+en', MT.I_PAS)]]),
         (('written', 'VBN'), [[('write', EM.V), ('+en', MT.I_PAS)]]),
         (('drawn', 'VBN'), [[('draw', EM.V), ('+n', MT.I_PAS)]]),
         (('clung', 'VBN'), [[('cling', EM.V), ('+ung', MT.I_PAS)]]),

         # verb: irregular
         (('bit', 'VBD'), [[('bite', EM.V), ('-e', MT.I_PAS)]]),
         (('bites', 'VBZ'), [[('bite', EM.V), ('+s', MT.I_3PS)]]),
         (('biting', 'VBG'), [[('bite', EM.V), ('+ing', MT.I_GER)]]),
         (('bitted', 'VBD'), [[('bit', EM.V), ('+ed', MT.I_PAS)]]),
         (('bitten', 'VBN'), [[('bite', EM.V), ('+en', MT.I_PAS)]]),
         (('bitting', 'VBG'), [[('bit', EM.V), ('+ing', MT.I_GER)]]),
         (('chivvies', 'VBZ'), [[('chivy', EM.V), ('+ies', MT.I_3PS)]]),
         (('took', 'VBD'), [[('take', EM.V), ('+ook', MT.I_PAS)]]),
         (('slept', 'VBD'), [[('sleep', EM.V), ('+pt', MT.I_PAS)]]),
         (('spoken', 'VBN'), [[('speak', EM.V), ('+oken', MT.I_PAS)]]),
         (('woken', 'VBN'), [[('wake', EM.V), ('+oken', MT.I_PAS)]]),

         # noun: plural
         (('studies', 'NNS'), [[('study', EM.N), ('+ies', MT.I_PLU)]]),
         (('crosses', 'NNS'), [[('cross', EM.N), ('+es', MT.I_PLU)]]),
         (('areas', 'NNS'), [[('area', EM.N), ('+s', MT.I_PLU)]]),
         (('men', 'NNS'), [[('man', EM.N), ('+men', MT.I_PLU)]]),
         (('vertebrae', 'NNS'), [[('vertebra', EM.N), ('+ae', MT.I_PLU)]]),
         (('foci', 'NNS'), [[('focus', EM.N), ('+i', MT.I_PLU)]]),
         (('optima', 'NNS'), [[('optimum', EM.N), ('+a', MT.I_PLU)]]),

         # noun: irregular
         (('indices', 'NNS'), [[('index', EM.N), ('+ices', MT.I_PLU)]]),
         (('wolves', 'NNS'), [[('wolf', EM.N), ('+ves', MT.I_PLU)]]),
         (('knives', 'NNS'), [[('knife', EM.N), ('+ves', MT.I_PLU)]]),
         (('quizzes', 'NNS'), [[('quiz', EM.N), ('+es', MT.I_PLU)]]),

         # adjective: comparative
         (('easier', 'JJR'), [[('easy', EM.J), ('+ier', MT.I_COM)]]),
         (('larger', 'JJR'), [[('large', EM.J), ('+er', MT.I_COM)]]),
         (('smaller', 'JJR'), [[('small', EM.J), ('+er', MT.I_COM)]]),
         (('bigger', 'JJR'), [[('big', EM.J), ('+er', MT.I_COM)]]),

         # adjective: superlative
         (('easiest', 'JJS'), [[('easy', EM.J), ('+iest', MT.I_SUP)]]),
         (('largest', 'JJS'), [[('large', EM.J), ('+est', MT.I_SUP)]]),
         (('smallest', 'JJS'), [[('small', EM.J), ('+est', MT.I_SUP)]]),
         (('biggest', 'JJS'), [[('big', EM.J), ('+est', MT.I_SUP)]]),

         # adjective: irregular
         (('cagier', 'JJR'), [[('cagey', EM.J), ('+ier', MT.I_COM)]]),
         (('worse', 'JJR'), [[('bad', EM.J), ('', MT.I_COM)]]),

         # adverb: comparative
         (('earlier', 'RBR'), [[('early', EM.R), ('+ier', MT.I_COM)]]),
         (('sooner', 'RBR'), [[('soon', EM.R), ('+er', MT.I_COM)]]),
         (('larger', 'RBR'), [[('large', EM.R), ('+er', MT.I_COM)]]),

         # adverb: superlative
         (('earliest', 'RBS'), [[('early', EM.R), ('+iest', MT.I_SUP)]]),
         (('soonest', 'RBS'), [[('soon', EM.R), ('+est', MT.I_SUP)]]),
         (('largest', 'RBS'), [[('large', EM.R), ('+est', MT.I_SUP)]]),

         # adverb: irregular
         (('further', 'RBR'), [[('far', EM.R), ('+urthe+', MT.I_COM)]]),
         (('best', 'RBS'), [[('well', EM.R), ('', MT.I_SUP)]]),
         (('worst', 'RBS'), [[('bad', EM.R), ('', MT.I_SUP)]]),
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
        (([('fortify', EM.V)]), [('forty', EM.N), ('+ify', MT.V_FY)]),
        (([('glorify', EM.V)]), [('glory', EM.N), ('+ify', MT.V_FY)]),
        (([('terrify', EM.V)]), [('terror', EM.N), ('+ify', MT.V_FY)]),
        (([('qualify', EM.V)]), [('quality', EM.N), ('+ify', MT.V_FY)]),
        (([('simplify', EM.V)]), [('simple', EM.J), ('+ify', MT.V_FY)]),
        (([('beautify', EM.V)]), [('beauty', EM.N), ('+ify', MT.V_FY)]),
        (([('liquefy', EM.V)]), [('liquid', EM.N), ('+efy', MT.V_FY)]),

        # verb: 'ize'
        (([('hospitalize', EM.V)]), [('hospital', EM.N), ('+ize', MT.V_IZE)]),
        (([('oxidize', EM.V)]), [('oxide', EM.N), ('+ize', MT.V_IZE)]),
        (([('theorize', EM.V)]), [('theory', EM.N), ('+ize', MT.V_IZE)]),
        (([('sterilize', EM.V)]), [('sterile', EM.J), ('+ize', MT.V_IZE)]),
        (([('crystallize', EM.V)]), [('crystal', EM.N), ('+ize', MT.V_IZE)]),
        (([('dramatize', EM.V)]), [('dramatic', EM.J), ('+ize', MT.V_IZE)]),
        (([('barbarize', EM.V)]), [('barbary', EM.N), ('+ize', MT.V_IZE)]),

        # verb: 'en'
        (([('strengthen', EM.V)]), [('strength', EM.N), ('+en', MT.V_EN)]),
        (([('brighten', EM.V)]), [('bright', EM.J), ('+en', MT.V_EN)]),

        # noun: 'age'
        (([('marriage', EM.N)]), [('marry', EM.V), ('+iage', MT.N_AGE)]),
        (([('passage', EM.N)]), [('pass', EM.V), ('+age', MT.N_AGE)]),
        (([('mileage', EM.N)]), [('mile', EM.N), ('+age', MT.N_AGE)]),

        # noun: 'al'
        (([('denial', EM.N)]), [('deny', EM.V), ('+ial', MT.N_AL)]),
        (([('approval', EM.N)]), [('approve', EM.V), ('+al', MT.N_AL)]),

        # noun: 'ance'
        (([('defiance', EM.N)]), [('defy', EM.V), ('+iance', MT.N_ANCE)]),
        (([('annoyance', EM.N)]), [('annoy', EM.V), ('+ance', MT.N_ANCE)]),
        (([('insurance', EM.N)]), [('insure', EM.V), ('+ance', MT.N_ANCE)]),
        (([('admittance', EM.N)]), [('admit', EM.V), ('+ance', MT.N_ANCE)]),
        (([('relevance', EM.N)]), [('relevant', EM.J), ('+ance', MT.N_ANCE)]),
        (([('pregnancy', EM.N)]), [('pregnant', EM.J), ('+ancy', MT.N_ANCE)]),
        (([('difference', EM.N)]), [('differ', EM.V), ('+ent', MT.J_ANT), ('+ence', MT.N_ANCE)]),
        (([('fluency', EM.N)]), [('fluent', EM.J), ('+ency', MT.N_ANCE)]),
        (([('accuracy', EM.N)]), [('accurate', EM.J), ('+cy', MT.N_ANCE)]),

        # noun: 'ant'
        (([('applicant', EM.N)]), [('apply', EM.V), ('+icant', MT.N_ANT)]),
        (([('assistant', EM.N)]), [('assist', EM.V), ('+ant', MT.N_ANT)]),
        (([('propellant', EM.N)]), [('propel', EM.V), ('+ant', MT.N_ANT)]),
        (([('servant', EM.N)]), [('serve', EM.V), ('+ant', MT.N_ANT)]),
        (([('immigrant', EM.N)]), [('immigrate', EM.V), ('+ant', MT.N_ANT)]),
        (([('dependent', EM.N)]), [('depend', EM.V), ('+ent', MT.N_ANT)]),
        (([('resident', EM.N)]), [('reside', EM.V), ('+ent', MT.N_ANT)]),

        # noun: 'dom'
        (([('freedom', EM.N)]), [('free', EM.J), ('+dom', MT.N_DOM)]),
        (([('kingdom', EM.N)]), [('king', EM.N), ('+dom', MT.N_DOM)]),

        # noun: 'ee'
        (([('employee', EM.N)]), [('employ', EM.V), ('+ee', MT.N_EE)]),
        (([('escapee', EM.N)]), [('escape', EM.V), ('+ee', MT.N_EE)]),

        # noun: 'er'
        (([('carrier', EM.N)]), [('carry', EM.V), ('+ier', MT.N_ER)]),
        (([('cashier', EM.N)]), [('cash', EM.N), ('+ier', MT.N_ER)]),
        (([('financier', EM.N)]), [('finance', EM.N), ('+ier', MT.N_ER)]),
        (([('profiteer', EM.N)]), [('profit', EM.N), ('+eer', MT.N_ER)]),
        (([('bowyer', EM.N)]), [('bow', EM.V), ('+yer', MT.N_ER)]),
        (([('lawyer', EM.N)]), [('law', EM.N), ('+yer', MT.N_ER)]),
        (([('reader', EM.N)]), [('read', EM.V), ('+er', MT.N_ER)]),
        (([('runner', EM.N)]), [('run', EM.V), ('+er', MT.N_ER)]),
        (([('writer', EM.N)]), [('write', EM.V), ('+er', MT.N_ER)]),
        (([('engineer', EM.N)]), [('engine', EM.N), ('+er', MT.N_ER)]),
        (([('hatter', EM.N)]), [('hat', EM.V), ('+er', MT.N_ER)]),
        (([('tiler', EM.N)]), [('tile', EM.V), ('+er', MT.N_ER)]),
        (([('beggar', EM.N)]), [('beg', EM.V), ('+ar', MT.N_ER)]),
        (([('liar', EM.N)]), [('lie', EM.V), ('+ar', MT.N_ER)]),
        (([('actor', EM.N)]), [('act', EM.V), ('+or', MT.N_ER)]),
        (([('abator', EM.N)]), [('abate', EM.V), ('+or', MT.N_ER)]),

        # noun: 'hood'
        (([('likelihood', EM.N)]), [('like', EM.N), ('+ly', MT.J_LY), ('+ihood', MT.N_HOOD)]),
        (([('childhood', EM.N)]), [('child', EM.N), ('+hood', MT.N_HOOD)]),

        # adjective: 'ing'
        (([('building', EM.N)]), [('build', EM.V), ('+ing', MT.N_ING)]),

        # noun: 'ism'
        (([('witticism', EM.N)]), [('wit', EM.N), ('+y', MT.J_Y), ('+icism', MT.N_ISM)]),
        (([('baptism', EM.N)]), [('baptize', EM.V), ('+ism', MT.N_ISM)]),
        (([('capitalism', EM.N)]), [('capital', EM.N), ('+ize', MT.V_IZE), ('+ism', MT.N_ISM)]),
        (([('bimetallism', EM.N)]), [('bimetal', EM.N), ('+ism', MT.N_ISM)]),

        # noun: 'ist'
        (([('apologist', EM.N)]), [('apology', EM.N), ('+ist', MT.N_IST)]),
        (([('capitalist', EM.N)]), [('capital', EM.J), ('+ist', MT.N_IST)]),
        (([('machinist', EM.N)]), [('machine', EM.N), ('+ist', MT.N_IST)]),
        (([('panellist', EM.N)]), [('panel', EM.N), ('+ist', MT.N_IST)]),
        (([('environmentalist', EM.N)]), [('environ', EM.V), ('+ment', MT.N_MENT), ('+al', MT.J_AL), ('+ist', MT.N_IST)]),

        # noun: 'ity'
        (([('capability', EM.N)]), [('capable', EM.J), ('+ility', MT.N_ITY)]),
        (([('variety', EM.N)]), [('vary', EM.V), ('+ious', MT.J_OUS), ('+ety', MT.N_ITY)]),
        (([('normality', EM.N)]), [('norm', EM.N), ('+al', MT.J_AL), ('+ity', MT.N_ITY)]),
        (([('adversity', EM.N)]), [('adverse', EM.J), ('+ity', MT.N_ITY)]),
        (([('jollity', EM.N)]), [('jolly', EM.J), ('+ity', MT.N_ITY)]),
        (([('frivolity', EM.N)]), [('frivol', EM.V), ('+ous', MT.J_OUS), ('+ity', MT.N_ITY)]),
        (([('loyalty', EM.N)]), [('loyal', EM.J), ('+ty', MT.N_ITY)]),

        # noun: 'man'
        (([('chairman', EM.N)]), [('chair', EM.V), ('+man', MT.N_MAN)]),
        (([('chairwoman', EM.N)]), [('chair', EM.V), ('+woman', MT.N_MAN)]),
        (([('chairperson', EM.N)]), [('chair', EM.V), ('+person', MT.N_MAN)]),

        # noun: 'ment'
        (([('development', EM.N)]), [('develop', EM.V), ('+ment', MT.N_MENT)]),
        (([('abridgment', EM.N)]), [('abridge', EM.V), ('+ment', MT.N_MENT)]),

        # noun: 'ness'
        (([('happiness', EM.N)]), [('happy', EM.J), ('+iness', MT.N_NESS)]),
        (([('kindness', EM.N)]), [('kind', EM.J), ('+ness', MT.N_NESS)]),
        (([('thinness', EM.N)]), [('thin', EM.J), ('+ness', MT.N_NESS)]),

        # noun: 'ship'
        (([('friendship', EM.N)]), [('friend', EM.N), ('+ship', MT.N_SHIP)]),

        # noun: 'sis'
        (([('diagnosis', EM.N)]), [('diagnose', EM.V), ('+sis', MT.N_SIS)]),
        (([('analysis', EM.N)]), [('analyze', EM.V), ('+sis', MT.N_SIS)]),

        # noun: 'tion'
        (([('verification', EM.N)]), [('verify', EM.V), ('+ication', MT.N_TION)]),
        (([('flirtation', EM.N)]), [('flirt', EM.V), ('+ation', MT.N_TION)]),
        (([('admiration', EM.N)]), [('admire', EM.V), ('+ation', MT.N_TION)]),
        (([('suspicion', EM.N)]), [('suspect', EM.V), ('+icion', MT.N_TION)]),
        (([('addition', EM.N)]), [('add', EM.V), ('+ition', MT.N_TION)]),
        (([('extension', EM.N)]), [('extend', EM.V), ('+sion', MT.N_TION)]),
        (([('decision', EM.N)]), [('decide', EM.V), ('+sion', MT.N_TION)]),
        (([('introduction', EM.N)]), [('introduce', EM.V), ('+tion', MT.N_TION)]),
        (([('resurrection', EM.N)]), [('resurrect', EM.V), ('+ion', MT.N_TION)]),
        (([('alienation', EM.N)]), [('alien', EM.V), ('+ation', MT.N_TION)]),

        # adjective: 'able'
        (([('certifiable', EM.J)]), [('cert', EM.N), ('+ify', MT.V_FY), ('+iable', MT.J_ABLE)]),
        (([('readable', EM.J)]), [('read', EM.V), ('+able', MT.J_ABLE)]),
        (([('writable', EM.J)]), [('write', EM.V), ('+able', MT.J_ABLE)]),
        (([('irritable', EM.J)]), [('irritate', EM.V), ('+able', MT.J_ABLE)]),
        (([('flammable', EM.J)]), [('flam', EM.V), ('+able', MT.J_ABLE)]),
        (([('visible', EM.J)]), [('vision', EM.N), ('+ible', MT.J_ABLE)]),

        # adjective: 'al'
        (([('influential', EM.J)]), [('influence', EM.N), ('+tial', MT.J_AL)]),
        (([('colonial', EM.J)]), [('colony', EM.N), ('+ial', MT.J_AL)]),
        (([('accidental', EM.J)]), [('accident', EM.N), ('+al', MT.J_AL)]),
        (([('visceral', EM.J)]), [('viscera', EM.N), ('+al', MT.J_AL)]),
        (([('universal', EM.J)]), [('universe', EM.N), ('+al', MT.J_AL)]),
        (([('bacterial', EM.J)]), [('bacteria', EM.N), ('+al', MT.J_AL)]),
        (([('focal', EM.J)]), [('focus', EM.N), ('+al', MT.J_AL)]),
        (([('economical', EM.J)]), [('economy', EM.N), ('+ic', MT.J_IC), ('+al', MT.J_AL)]),

        # adjective: 'ant'
        (([('applicant', EM.J)]), [('apply', EM.V), ('+icant', MT.J_ANT)]),
        (([('relaxant', EM.J)]), [('relax', EM.V), ('+ant', MT.J_ANT)]),
        (([('propellant', EM.J)]), [('propel', EM.V), ('+ant', MT.J_ANT)]),
        (([('pleasant', EM.J)]), [('please', EM.V), ('+ant', MT.J_ANT)]),
        (([('dominant', EM.J)]), [('dominate', EM.V), ('+ant', MT.J_ANT)]),
        (([('absorbent', EM.J)]), [('absorb', EM.V), ('+ent', MT.J_ANT)]),
        (([('abhorrent', EM.J)]), [('abhor', EM.V), ('+ent', MT.J_ANT)]),
        (([('adherent', EM.J)]), [('adhere', EM.V), ('+ent', MT.J_ANT)]),

        # adjective: 'ary'
        (([('cautionary', EM.J)]), [('caution', EM.V), ('+ary', MT.J_ARY)]),
        (([('imaginary', EM.J)]), [('imagine', EM.V), ('+ary', MT.J_ARY)]),
        (([('pupillary', EM.J)]), [('pupil', EM.N), ('+ary', MT.J_ARY)]),
        (([('monetary', EM.J)]), [('money', EM.N), ('+tary', MT.J_ARY)]),

        # adjective: 'ed'
        (([('diffused', EM.J)]), [('diffuse', EM.V), ('+d', MT.J_ED)]),
        (([('shrunk', EM.J)]), [('shrink', EM.V), ('+u+', MT.J_ED)]),

        # adjective: 'ful'
        (([('beautiful', EM.J)]), [('beauty', EM.N), ('+iful', MT.J_FUL)]),
        (([('thoughtful', EM.J)]), [('thought', EM.N), ('+ful', MT.J_FUL)]),
        (([('helpful', EM.J)]), [('help', EM.V), ('+ful', MT.J_FUL)]),

        # adjective: 'ic'
        (([('realistic', EM.J)]), [('real', EM.N), ('+ize', MT.V_IZE), ('+stic', MT.J_IC)]),
        (([('fantastic', EM.J)]), [('fantasy', EM.N), ('+tic', MT.J_IC)]),
        (([('diagnostic', EM.J)]), [('diagnose', EM.V), ('+sis', MT.N_SIS), ('+tic', MT.J_IC)]),
        (([('analytic', EM.J)]), [('analyze', EM.V), ('+sis', MT.N_SIS), ('+tic', MT.J_IC)]),
        (([('poetic', EM.J)]), [('poet', EM.N), ('+ic', MT.J_IC)]),
        (([('metallic', EM.J)]), [('metal', EM.N), ('+ic', MT.J_IC)]),
        (([('sophomoric', EM.J)]), [('sophomore', EM.N), ('+ic', MT.J_IC)]),

        # adjective: 'ing'
        (([('dignifying', EM.J)]), [('dignity', EM.N), ('+ify', MT.V_FY), ('+ing', MT.J_ING)]),
        (([('abiding', EM.J)]), [('abide', EM.V), ('+ing', MT.J_ING)]),

        # adjective: 'ish'
        (([('bearish', EM.J)]), [('bear', EM.V), ('+ish', MT.J_ISH)]),
        (([('ticklish', EM.J)]), [('tickle', EM.V), ('+ish', MT.J_ISH)]),
        (([('reddish', EM.J)]), [('red', EM.V), ('+ish', MT.J_ISH)]),
        (([('boyish', EM.J)]), [('boy', EM.N), ('+ish', MT.J_ISH)]),
        (([('faddish', EM.J)]), [('fade', EM.V), ('+ish', MT.J_ISH)]),
        (([('mulish', EM.J)]), [('mule', EM.N), ('+ish', MT.J_ISH)]),

        # adjective: 'ive'
        (([('talkative', EM.J)]), [('talk', EM.V), ('+ative', MT.J_IVE)]),
        (([('adjudicative', EM.J)]), [('adjudicate', EM.V), ('+ative', MT.J_IVE)]),
        (([('destructive', EM.J)]), [('destruct', EM.V), ('+ive', MT.J_IVE)]),
        (([('defensive', EM.J)]), [('defense', EM.N), ('+ive', MT.J_IVE)]),
        (([('divisive', EM.J)]), [('divide', EM.V), ('+sion', MT.N_TION), ('+ive', MT.J_IVE)]),

        # adjective: 'less'
        (([('countless', EM.J)]), [('count', EM.V), ('+less', MT.J_LESS)]),
        (([('speechless', EM.J)]), [('speech', EM.N), ('+less', MT.J_LESS)]),

        # adjective: 'like'
        (([('childlike', EM.J)]), [('child', EM.N), ('+like', MT.J_LIKE)]),

        # adjective: 'ly'
        (([('daily', EM.J)]), [('day', EM.N), ('+ily', MT.J_LY)]),
        (([('weekly', EM.J)]), [('week', EM.N), ('+ly', MT.J_LY)]),

        # adjective: 'most'
        (([('innermost', EM.J)]), [('inner', EM.J), ('+most', MT.J_MOST)]),

        # adjective: 'ous'
        (([('courteous', EM.J)]), [('court', EM.N), ('+eous', MT.J_OUS)]),
        (([('glorious', EM.J)]), [('glory', EM.V), ('+ious', MT.J_OUS)]),
        (([('wondrous', EM.J)]), [('wonder', EM.N), ('+rous', MT.J_OUS)]),
        (([('marvellous', EM.J)]), [('marvel', EM.V), ('+ous', MT.J_OUS)]),
        (([('covetous', EM.J)]), [('covet', EM.V), ('+ous', MT.J_OUS)]),
        (([('nervous', EM.J)]), [('nerve', EM.V), ('+ous', MT.J_OUS)]),
        (([('cancerous', EM.J)]), [('cancer', EM.N), ('+ous', MT.J_OUS)]),
        (([('analogous', EM.J)]), [('analogy', EM.N), ('+ous', MT.J_OUS)]),
        (([('religious', EM.J)]), [('religion', EM.N), ('+ous', MT.J_OUS)]),

        # adjective: 'some'
        (([('worrisome', EM.J)]), [('worry', EM.N), ('+isome', MT.J_SOME)]),
        (([('troublesome', EM.J)]), [('trouble', EM.N), ('+some', MT.J_SOME)]),
        (([('awesome', EM.J)]), [('awe', EM.N), ('+some', MT.J_SOME)]),
        (([('fulsome', EM.J)]), [('full', EM.J), ('+some', MT.J_SOME)]),

        # adjective: 'wise'
        (([('clockwise', EM.J)]), [('clock', EM.N), ('+wise', MT.J_WISE)]),
        (([('likewise', EM.J)]), [('like', EM.J), ('+wise', MT.J_WISE)]),

        # adjective: 'y'
        (([('clayey', EM.J)]), [('clay', EM.N), ('+ey', MT.J_Y)]),
        (([('grouchy', EM.J)]), [('grouch', EM.V), ('+y', MT.J_Y)]),
        (([('runny', EM.J)]), [('run', EM.V), ('+y', MT.J_Y)]),
        (([('rumbly', EM.J)]), [('rumble', EM.V), ('+y', MT.J_Y)]),

        # adverb: 'ly'
        (([('electronically', EM.R)]), [('electron', EM.N), ('+ic', MT.J_IC), ('+ally', MT.R_LY)]),
        (([('easily', EM.R)]), [('ease', EM.V), ('+y', MT.J_Y), ('+ily', MT.R_LY)]),
        (([('sadly', EM.R)]), [('sad', EM.J), ('+ly', MT.R_LY)]),
        (([('fully', EM.R)]), [('full', EM.J), ('+ly', MT.R_LY)]),
        (([('incredibly', EM.R)]), [('incredible', EM.J), ('+ly', MT.R_LY)]),
     ])
]


@pytest.mark.parametrize('data', data_analyze_derivation)
def test_analyze_derivation(en_morph_analyzer, data):
    input, expected = zip(*data)
    actual = tuple(en_morph_analyzer._analyze_derivation(tp) for tp in input)
    assert actual == expected


data_inflection_derivation = [
    ([
        (('ownerships', None), [[('own', EM.V), ('+er', MT.N_ER), ('+ship', MT.N_SHIP), ('+s', MT.I_PLU)]]),
        (('offensiveness', None), [[('offense', EM.N), ('+ive', MT.J_IVE), ('+ness', MT.N_NESS)]]),
        (('chairmen', None), [[('chair', EM.V), ('+man', MT.N_MAN), ('+men', MT.I_PLU)]]),
        (('girlisher', None), [[('girl', EM.N), ('+ish', MT.J_ISH), ('+er', MT.I_COM)]]),
        (('environmentalist', None), [[('environ', EM.V), ('+ment', MT.N_MENT), ('+al', MT.J_AL), ('+ist', MT.N_IST)]]),
        (('economically', None), [[('economy', EM.N), ('+ic', MT.J_IC), ('+ally', MT.R_LY)]]),
        (('beautifulliest', None), [[('beauty', EM.N), ('+iful', MT.J_FUL), ('+ly', MT.R_LY), ('+iest', MT.I_SUP)]]),
     ])
]


@pytest.mark.parametrize('data', data_inflection_derivation)
def test_data_inflection_derivation(en_morph_analyzer, data):
    input, expected = zip(*data)
    actual = tuple(en_morph_analyzer.analyze(token, pos, True) for token, pos in input)
    assert actual == expected