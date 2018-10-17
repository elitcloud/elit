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
from collections import OrderedDict
from types import SimpleNamespace
from typing import Sequence, Set, List, Optional, Tuple, Dict

import os

import marisa_trie
from pkg_resources import resource_filename

from elit.util import io
from elit.util.io import read_word_set

__author__ = 'Jinho D. Choi'


class MorphTag:
    # inflection: verb
    I_3PS = 'I_3PS'  # 3rd-person singular present
    I_GER = 'I_GER'  # gerund
    I_PAS = 'I_PAS'  # past

    # inflection: noun
    I_PLU = 'I_PLU'  # plural

    # inflection: adjective/adverb
    I_COM = 'I_COM'  # comparative
    I_SUP = 'I_SUP'  # superlative

    # derivation: verb
    V_EN = 'V_EN'
    V_FY = 'V_FY'
    V_IZE = 'V_IZE'

    # derivation: noun
    N_AGE = 'N_AGE'
    N_AL = 'N_AL'
    N_ANCE = 'N_ANCE'
    N_ANT = 'N_ANT'
    N_DOM = 'N_DOM'
    N_EE = 'N_EE'
    N_ER = 'N_ER'
    N_HOOD = 'N_HOOD'
    N_ING = 'N_ING'
    N_ISM = 'N_ISM'
    N_IST = 'N_IST'
    N_ITY = 'N_ITY'
    N_MAN = 'N_MAN'
    N_MENT = 'N_MENT'
    N_NESS = 'N_NESS'
    N_SHIP = 'N_SHIP'
    N_SIS = 'N_SIS'
    N_TION = 'N_TION'
    N_WARE = 'N_WARE'

    # derivation: adjective
    J_ABLE = 'J_ABLE'
    J_AL = 'J_AL'
    J_ANT = 'J_ANT'
    J_ARY = 'J_ARY'
    J_ED = 'J_ED'
    J_FUL = 'J_FUL'
    J_IC = 'J_IC'
    J_ING = 'J_ING'
    J_ISH = 'J_ISH'
    J_IVE = 'J_IVE'
    J_LESS = 'J_LESS'
    J_LIKE = 'J_LIKE'
    J_LY = 'J_LY'
    J_MOST = 'J_MOST'
    J_OUS = 'J_OUS'
    J_SOME = 'J_SOME'
    J_WISE = 'J_WISE'
    J_Y = 'J_Y'

    # derivation: adverb
    R_LY = 'R_LY'


class AffixRule:
    CK = {'c', 'k'}

    def __init__(self, affix_tag: str, affix_form: str, stem_affixes: Sequence[str], token_tagset: Set[str] = None, stem_tag: str = None, double_consonants: bool = False):
        """
        :param affix_tag: the morphology tag of the affix (see :class:`MorphTag`).
        :param affix_form: the form of the affix (e.g., 'ies', 'es', 's').
        :param stem_affixes: the affixes appended to the stem to recover the lemma (e.g., 'y').
        :param token_tagset: the set of possible part-of-speech tags (Penn Treebank style) for the input token.
        :param stem_tag: the part-of-speech tag (Penn Treebank style) of the stem.
        :param double_consonants: if True, the affix forms with double consonants are considered (e.g., run+n+ing).
        """
        self.affix_tag = affix_tag
        self.affix_form = affix_form
        self.token_tagset = token_tagset
        self.stem_tag = stem_tag
        self.stem_affixes = stem_affixes
        self.double_consonants = double_consonants

    @classmethod
    def is_double_consonants(cls, s: str) -> bool:
        return len(s) >= 4 and (
            s[-1] == s[-2] or
            (s[-1] in cls.CK and s[-2] in cls.CK))


def suffix_matcher(rule: AffixRule, base_set: Set[str], token: str, pos: str = None) -> Optional[str]:
    """
    :param rule: the affix rule.
    :param base_set: the set including base forms (lemmas).
    :param token: the input token.
    :param pos: the part-of-speech tag of the input token.
    :return: the lemma of the input token if it matches to the affix rule; otherwise, None.
    """

    def match(s: str):
        for suffix in rule.stem_affixes:
            base = s + suffix
            if base in base_set: return base
        return None

    if len(rule.affix_form) > len(token) or not token.endswith(rule.affix_form): return None
    if pos is not None and pos not in rule.token_tagset: return None
    stem = token[:-len(rule.affix_form)]

    lemma = match(stem)
    if lemma is not None: return lemma

    if rule.double_consonants and AffixRule.is_double_consonants(stem):
        lemma = match(stem[:-1])
        if lemma is not None: return lemma

    return None


def extract_suffix(token: str, lemma: str) -> str:
    i = len(os.path.commonprefix((token, lemma)))
    if i == 0: return ''
    tsuf = token[i:]
    lsuf = lemma[i:]
    if i < len(token) and AffixRule.is_double_consonants(token[:i+1]): tsuf = token[i+1:]

    j = len(os.path.commonprefix((tsuf[::-1], lsuf[::-1])))
    if j == 0: return '-'+lsuf if not tsuf else '+'+tsuf

    tsuf = tsuf[:-j]
    lsuf = lsuf[:-j]
    return '-'+lsuf+'-' if not tsuf else '+'+tsuf+'+'


class EnglishMorphAnalyzer:
    V = 'V'  # verb
    N = 'N'  # noun
    J = 'J'  # adjective
    R = 'R'  # adverb
    M = 'M'  # modal
    T = 'T'  # TO

    def __init__(self):
        # initialize _lexicons
        resource_path = 'elit.resources.lemmatizer.english'
        self._lexicons = self._init_lexicons(resource_path)
        self._derivation_exc = read_word_set(resource_filename(resource_path, 'derivation.exc'))

        # initialize rules
        self._prefixes = self._init_prefixes()
        self._irregular_rules = self._init_irregular_rules()
        self._inflection_rules = self._init_inflection_rules()
        self._derivation_rules = self._init_derivation_rules()

    def _init_lexicons(self, resource_path: str) -> Dict[str, SimpleNamespace]:
        return {
            self.V: SimpleNamespace(base_set=io.read_word_set(resource_filename(resource_path, 'verb.base')),
                                    exc_dict=io.read_word_dict(resource_filename(resource_path, 'verb.exc')),
                                    stem_tag=self.V,
                                    stem_tagset={'VB', 'VBP'},
                                    infl_tagset={'VBD', 'VBG', 'VBN', 'VBZ'},
                                    affix_tag=lambda x: MorphTag.I_GER if x.endswith('ing') else MorphTag.I_3PS if x.endswith('s') else MorphTag.I_PAS),
            self.N: SimpleNamespace(base_set=io.read_word_set(resource_filename(resource_path, 'noun.base')),
                                    exc_dict=io.read_word_dict(resource_filename(resource_path, 'noun.exc')),
                                    stem_tag=self.N,
                                    stem_tagset={'NN', 'NNP'},
                                    infl_tagset={'NNS', 'NNPS'},
                                    affix_tag=lambda x: MorphTag.I_PLU),
            self.J: SimpleNamespace(base_set=io.read_word_set(resource_filename(resource_path, 'adjective.base')),
                                    exc_dict=io.read_word_dict(resource_filename(resource_path, 'adjective.exc')),
                                    stem_tag=self.J,
                                    stem_tagset={'JJ'},
                                    infl_tagset={'JJR', 'JJS'},
                                    affix_tag=lambda x: MorphTag.I_SUP if x.endswith('st') else MorphTag.I_COM),
            self.R: SimpleNamespace(base_set=io.read_word_set(resource_filename(resource_path, 'adverb.base')),
                                    exc_dict=io.read_word_dict(resource_filename(resource_path, 'adverb.exc')),
                                    stem_tag=self.R,
                                    stem_tagset={'RB'},
                                    infl_tagset={'RBR', 'RBS'},
                                    affix_tag=lambda x: MorphTag.I_SUP if x.endswith('st') else MorphTag.I_COM)
        }

    def _init_irregular_rules(self) -> Dict[Tuple[str, str], List[List[Tuple[str, str]]]]:
        return {
            ("n't", 'RB'): [[('not', self.R)]],
            ("n't", None): [[('not', self.R)]],
            ("'nt", 'RB'): [[('not', self.R)]],
            ("'nt", None): [[('not', self.R)]],
            ("'d", 'MD'): [[('would', self.M)]],
            ("'ll", 'MD'): [[('will', self.M)]],
            ("'ll", None): [[('will', self.M)]],
            ('ca', 'MD'): [[('can', self.M)]],
            ('na', 'TO'): [[('to', self.T)]],
            ("'d", 'VBD'): [[('have', self.V), ('+d', MorphTag.I_PAS)]],
            ("'d", None): [[('have', self.V), ('+d', MorphTag.I_PAS)], [('would', self.M)]],
            ("'ve", 'VBP'): [[('have', self.V)]],
            ("'ve", None): [[('have', self.V)]],
            ("'m", 'VBP'): [[('be', self.V)]],
            ("'m", None): [[('be', self.V)]],
            ("'re", 'VBP'): [[('be', self.V)]],
            ("'re", None): [[('be', self.V)]],
            ('ai', 'VBP'): [[('be', self.V)]],
            ('am', 'VBP'): [[('be', self.V)]],
            ('am', None): [[('be', self.V)], [('am', self.N)]],
            ('are', 'VBP'): [[('be', self.V)]],
            ('are', None): [[('be', self.V)]],
            ('is', 'VBZ'): [[('be', self.V), ('', MorphTag.I_3PS)]],
            ('is', None): [[('be', self.V), ('', MorphTag.I_3PS)]],
            ('was', 'VBD'): [[('be', self.V), ('', MorphTag.I_3PS), ('', MorphTag.I_PAS)]],
            ('was', None): [[('be', self.V), ('', MorphTag.I_3PS), ('', MorphTag.I_PAS)]],
            ('were', 'VBD'): [[('be', self.V), ('', MorphTag.I_PAS)]],
            ('were', None): [[('be', self.V), ('', MorphTag.I_PAS)]],
            ('gon', 'VBG'): [[('go', self.V), ('+ing', MorphTag.I_GER)]],
        }

    def _init_inflection_rules(self) -> Dict[str, List[AffixRule]]:
        TPS = {'VBZ'}
        GER = {'VBG'}
        PAS = {'VBD', 'VBN'}
        PLU = {'NNS', 'NNPS'}
        JCO = {'JJR'}
        JSU = {'JJS'}
        RCO = {'RBR'}
        RSU = {'RBS'}
        t = None

        return {
            self.V: [
                AffixRule(MorphTag.I_3PS, 'ies', ['y'], TPS, t),  # studies
                AffixRule(MorphTag.I_3PS, 'es', [''], TPS, t),  # pushes
                AffixRule(MorphTag.I_3PS, 's', [''], TPS, t),  # takes

                AffixRule(MorphTag.I_GER, 'ying', ['ie'], GER, t),  # lying
                AffixRule(MorphTag.I_GER, 'ing', ['', 'e'], GER, t, True),  # feeling, running, taking

                AffixRule(MorphTag.I_PAS, 'ied', ['y'], PAS, t),  # denied
                AffixRule(MorphTag.I_PAS, 'ed', [''], PAS, t, True),  # entered, zipped
                AffixRule(MorphTag.I_PAS, 'd', [''], PAS, t),  # heard
                AffixRule(MorphTag.I_PAS, 'en', ['', 'e'], PAS, t, True),  # fallen, written
                AffixRule(MorphTag.I_PAS, 'n', [''], PAS, t),  # drawn
                AffixRule(MorphTag.I_PAS, 'ung', ['ing'], PAS, t),  # clung
            ],
            self.N: [
                AffixRule(MorphTag.I_PLU, 'ies', ['y'], PLU, t),  # studies
                AffixRule(MorphTag.I_PLU, 'es', [''], PLU, t),  # crosses
                AffixRule(MorphTag.I_PLU, 's', [''], PLU, t),  # areas
                AffixRule(MorphTag.I_PLU, 'men', ['man'], PLU, t),  # women
                AffixRule(MorphTag.I_PLU, 'ae', ['a'], PLU, t),  # vertebrae
                AffixRule(MorphTag.I_PLU, 'i', ['us'], PLU, t),  # foci
                AffixRule(MorphTag.I_PLU, 'a', ['um'], PLU, t),  # optima
            ],
            self.J: [
                AffixRule(MorphTag.I_COM, 'ier', ['y'], JCO, t),  # easier
                AffixRule(MorphTag.I_COM, 'er', [''], JCO, t, True),  # smaller, bigger
                AffixRule(MorphTag.I_COM, 'er', ['e'], JCO, t),  # larger

                AffixRule(MorphTag.I_SUP, 'iest', ['y'], JSU, t),  # easiest
                AffixRule(MorphTag.I_SUP, 'est', ['e'], JSU, t),  # smallest, biggest
                AffixRule(MorphTag.I_SUP, 'est', [''], JSU, t, True),  # largest
            ],
            self.R: [
                AffixRule(MorphTag.I_COM, 'ier', ['y'], RCO, t),  # earlier
                AffixRule(MorphTag.I_COM, 'er', ['', 'e'], RCO, t),  # sooner, larger

                AffixRule(MorphTag.I_SUP, 'iest', ['y'], RSU, t),  # earliest
                AffixRule(MorphTag.I_SUP, 'est', ['', 'e'], RSU, t),  # soonest, largest
            ]
        }

    def _init_derivation_rules(self) -> Dict[str, List[AffixRule]]:
        t = None
        return {
            self.V: [
                AffixRule(MorphTag.V_EN, 'en', [''], t, self.N),  # strengthen
                AffixRule(MorphTag.V_EN, 'en', [''], t, self.J),  # brighten

                AffixRule(MorphTag.V_FY, 'ify', ['or', 'y', '', 'ity'], t, self.N),  # fortify, glorify, terrify, qualify
                AffixRule(MorphTag.V_FY, 'ify', ['e', 'y'], t, self.J),  # simplify, beautify
                AffixRule(MorphTag.V_FY, 'efy', ['id'], t, self.N),  # liquefy

                AffixRule(MorphTag.V_IZE, 'ize', ['', 'e', 'y'], t, self.N, True),  # hospitalize, oxidize, theorize
                AffixRule(MorphTag.V_IZE, 'ize', ['', 'e', 'ic', 'ous'], t, self.J, True),  # sterilize, crystallize, dramatize, barbarize
                AffixRule(MorphTag.V_IZE, 'ise', ['', 'e', 'y'], t, self.N, True),  # hospitalise, oxidise, theorise
                AffixRule(MorphTag.V_IZE, 'ise', ['', 'e', 'ic', 'ous'], t, self.J, True),  # sterilise, crystallise, dramatise, barbarise
            ],
            self.N: [
                AffixRule(MorphTag.N_AGE, 'iage', ['y'], t, self.V),  # marriage
                AffixRule(MorphTag.N_AGE, 'age', [''], t, self.V),  # passage
                AffixRule(MorphTag.N_AGE, 'age', [''], t, self.N),  # mileage

                AffixRule(MorphTag.N_AL, 'ial', ['y'], t, self.V),  # denial
                AffixRule(MorphTag.N_AL, 'al', ['e'], t, self.V),  # approval

                AffixRule(MorphTag.N_ANCE, 'iance', ['y'], t, self.V),  # defiance
                AffixRule(MorphTag.N_ANCE, 'ance', ['', 'e'], t, self.V, True),  # annoyance, insurance, admittance
                AffixRule(MorphTag.N_ANCE, 'ance', ['ant'], t, self.J),  # relevance
                AffixRule(MorphTag.N_ANCE, 'ancy', ['ant'], t, self.J),  # pregnancy
                AffixRule(MorphTag.N_ANCE, 'ence', ['ent'], t, self.J),  # difference
                AffixRule(MorphTag.N_ANCE, 'ency', ['ent'], t, self.J),  # fluency
                AffixRule(MorphTag.N_ANCE, 'cy', ['te'], t, self.J),  # accuracy

                AffixRule(MorphTag.N_ANT, 'icant', ['y'], t, self.V, True),  # applicant
                AffixRule(MorphTag.N_ANT, 'ant', ['', 'e', 'ate'], t, self.V, True),  # assistant, propellant, servant, immigrant
                AffixRule(MorphTag.N_ANT, 'ent', ['', 'e'], t, self.V),  # dependent, resident

                AffixRule(MorphTag.N_DOM, 'dom', [''], t, self.J),  # freedom
                AffixRule(MorphTag.N_DOM, 'dom', [''], t, self.N),  # kingdom

                AffixRule(MorphTag.N_EE, 'ee', ['', 'e'], t, self.V, True),  # employee, escapee,

                AffixRule(MorphTag.N_ER, 'ier', ['y'], t, self.V),  # carrier
                AffixRule(MorphTag.N_ER, 'ier', ['', 'e'], t, self.N),  # cashier, financier
                AffixRule(MorphTag.N_ER, 'yer', [''], t, self.V),  # bowyer
                AffixRule(MorphTag.N_ER, 'yer', [''], t, self.N),  # lawyer
                AffixRule(MorphTag.N_ER, 'eer', [''], t, self.N),  # profiteer
                AffixRule(MorphTag.N_ER, 'er', ['', 'e'], t, self.V, True),  # reader, runner, writer
                AffixRule(MorphTag.N_ER, 'er', ['', 'e'], t, self.N, True),  # engineer, hatter, tiler
                AffixRule(MorphTag.N_ER, 'ar', ['', 'e'], t, self.V, True),  # beggar, liar
                AffixRule(MorphTag.N_ER, 'or', ['', 'e'], t, self.V, True),  # actor, abator

                AffixRule(MorphTag.N_HOOD, 'ihood', ['y'], t, self.J),  # likelihood
                AffixRule(MorphTag.N_HOOD, 'ihood', ['y'], t, self.N),  # likelihood
                AffixRule(MorphTag.N_HOOD, 'hood', [''], t, self.J),  # childhood
                AffixRule(MorphTag.N_HOOD, 'hood', [''], t, self.N),  # childhood

                AffixRule(MorphTag.N_ING, 'ying', ['ie'], t, self.V),  # lying
                AffixRule(MorphTag.N_ING, 'ing', ['', 'e'], t, self.V, True),  # building

                AffixRule(MorphTag.N_ISM, 'icism', ['y'], t, self.J),  # witticism
                AffixRule(MorphTag.N_ISM, 'ism', ['ize'], t, self.V),  # baptism
                AffixRule(MorphTag.N_ISM, 'ism', ['', 'e'], t, self.N, True),  # capitalism, bimetallism

                AffixRule(MorphTag.N_IST, 'ist', ['', 'e', 'y'], t, self.J, True),  # environmentalist
                AffixRule(MorphTag.N_IST, 'ist', ['', 'e', 'y'], t, self.N, True),  # apologist, capitalist, machinist, panellist

                AffixRule(MorphTag.N_ITY, 'ility', ['le'], t, self.J),  # capability
                AffixRule(MorphTag.N_ITY, 'ety', ['ous'], t, self.J),  # variety
                AffixRule(MorphTag.N_ITY, 'ity', ['', 'e', 'y', 'ous'], t, self.J),  # normality, adversity, jollity, frivolity
                AffixRule(MorphTag.N_ITY, 'ty', [''], t, self.J),  # loyalty

                AffixRule(MorphTag.N_MAN, 'man', [''], t, self.V),  # repairman
                AffixRule(MorphTag.N_MAN, 'man', [''], t, self.N),  # chairman
                AffixRule(MorphTag.N_MAN, 'woman', [''], t, self.V),  # repairwoman
                AffixRule(MorphTag.N_MAN, 'woman', [''], t, self.N),  # chairwoman
                AffixRule(MorphTag.N_MAN, 'person', [''], t, self.V),  # repairperson
                AffixRule(MorphTag.N_MAN, 'person', [''], t, self.N),  # chairperson

                AffixRule(MorphTag.N_MENT, 'ment', ['', 'e'], t, self.V),  # development, abridgment

                AffixRule(MorphTag.N_NESS, 'iness', ['y'], t, self.J),  # happiness
                AffixRule(MorphTag.N_NESS, 'ness', [''], t, self.J, True),  # kindness, thinness

                AffixRule(MorphTag.N_SHIP, 'ship', [''], t, self.N),  # friendship

                AffixRule(MorphTag.N_SIS, 'sis', ['ze', 'se'], t, self.V),  # diagnosis, analysis

                AffixRule(MorphTag.N_TION, 'ication', ['y'], t, self.V),  # verification
                AffixRule(MorphTag.N_TION, 'ation', ['', 'e'], t, self.V),  # flirtation, admiration
                AffixRule(MorphTag.N_TION, 'icion', ['ect'], t, self.V),  # suspicion
                AffixRule(MorphTag.N_TION, 'ition', [''], t, self.V),  # addition
                AffixRule(MorphTag.N_TION, 'sion', ['d', 'de'], t, self.V),  # extension, decision (illusion, division)
                AffixRule(MorphTag.N_TION, 'tion', ['', 'e'], t, self.V),  # introduction
                AffixRule(MorphTag.N_TION, 'ion', ['', 'e'], t, self.V),  # resurrection, alienation
            ],
            self.J: [
                AffixRule(MorphTag.J_ABLE, 'iable', ['y'], t, self.V),  # certifiable
                AffixRule(MorphTag.J_ABLE, 'able', ['', 'e', 'ate'], t, self.V, True),  # readable, writable, irritable
                AffixRule(MorphTag.J_ABLE, 'able', ['', 'e'], t, self.N, True),  # flammable, fashionable
                AffixRule(MorphTag.J_ABLE, 'ible', ['ion'], t, self.N),  # visible

                AffixRule(MorphTag.J_AL, 'tial', ['ce'], t, self.N),  # influential
                AffixRule(MorphTag.J_AL, 'ial', ['y'], t, self.N),  # colonial
                AffixRule(MorphTag.J_AL, 'al', ['y'], t, self.N),  # colonial
                AffixRule(MorphTag.J_AL, 'al', ['', 'a', 'e', 'um', 'us'], t, self.N),  # accidental, visceral, universal, bacterial, focal
                AffixRule(MorphTag.J_AL, 'al', [''], t, self.J),  # economical

                AffixRule(MorphTag.J_ANT, 'icant', ['y'], t, self.V),  # applicant
                AffixRule(MorphTag.J_ANT, 'ant', ['', 'e', 'ate'], t, self.V, True),  # relaxant, propellant, pleasant, dominant
                AffixRule(MorphTag.J_ANT, 'ent', ['', 'e'], t, self.V, True),  # absorbent, abhorrent, adherent

                AffixRule(MorphTag.J_ARY, 'tary', ['y'], t, self.N),  # monetary
                AffixRule(MorphTag.J_ARY, 'ary', ['', 'e'], t, self.V, True),  # cautionary
                AffixRule(MorphTag.J_ARY, 'ary', ['', 'e'], t, self.N, True),  # imaginary, pupillary

                AffixRule(MorphTag.J_FUL, 'iful', ['y'], t, self.V),  # beautiful
                AffixRule(MorphTag.J_FUL, 'iful', ['y'], t, self.N),  # beautiful
                AffixRule(MorphTag.J_FUL, 'ful', [''], t, self.V),  # harmful
                AffixRule(MorphTag.J_FUL, 'ful', [''], t, self.N),  # thoughtful

                AffixRule(MorphTag.J_IC, 'stic', ['ze'], t, self.V),  # realistic
                AffixRule(MorphTag.J_IC, 'tic', ['y', 'is', 'sis'], t, self.N),  # fantastic, diagnostic, analytic
                AffixRule(MorphTag.J_IC, 'ic', ['', 'e', 'y'], t, self.N, True),  # poetic, metallic, sophomoric

                AffixRule(MorphTag.J_ING, 'ying', ['ie'], t, self.V),  # lying
                AffixRule(MorphTag.J_ING, 'ing', ['', 'e'], t, self.V, True),  # differing

                AffixRule(MorphTag.J_ISH, 'ish', ['', 'e'], t, self.V, True),  # bearish, ticklish
                AffixRule(MorphTag.J_ISH, 'ish', ['', 'e'], t, self.J, True),  # foolish, reddish, bluish
                AffixRule(MorphTag.J_ISH, 'ish', ['', 'e'], t, self.N, True),  # boyish, faddish, mulish

                AffixRule(MorphTag.J_IVE, 'ative', ['', 'ate'], t, self.V),  # talkative, adjudicative
                AffixRule(MorphTag.J_IVE, 'ive', ['', 'e'], t, self.V),  # destructive
                AffixRule(MorphTag.J_IVE, 'ive', ['', 'e'], t, self.J),  # corrective
                AffixRule(MorphTag.J_IVE, 'ive', ['', 'e', 'ion'], t, self.N),  # massive, defensive, divisive

                AffixRule(MorphTag.J_LESS, 'less', [''], t, self.V),  # countless
                AffixRule(MorphTag.J_LESS, 'less', [''], t, self.N),  # speechless

                AffixRule(MorphTag.J_LIKE, 'like', [''], t, self.N),  # childlike

                AffixRule(MorphTag.J_LY, 'ily', ['y'], t, self.N),  # daily
                AffixRule(MorphTag.J_LY, 'ly', [''], t, self.N),  # weekly

                AffixRule(MorphTag.J_MOST, 'most', [''], t, self.J),  # innermost

                AffixRule(MorphTag.J_OUS, 'eous', [''], t, self.N),  # courteous
                AffixRule(MorphTag.J_OUS, 'ious', ['y'], t, self.V),  # various
                AffixRule(MorphTag.J_OUS, 'ious', ['y'], t, self.N),  # glorious
                AffixRule(MorphTag.J_OUS, 'rous', ['er'], t, self.N),  # wondrous (disastrous)
                AffixRule(MorphTag.J_OUS, 'ous', ['', 'e'], t, self.V, True),  # covetous, marvellous, nervous
                AffixRule(MorphTag.J_OUS, 'ous', ['y', '', 'e', 'on'], t, self.N, True),  # cancerous, analogous, religious

                AffixRule(MorphTag.J_SOME, 'isome', ['y'], t, self.J),  # worrisome
                AffixRule(MorphTag.J_SOME, 'isome', ['y'], t, self.N),  # worrisome
                AffixRule(MorphTag.J_SOME, 'some', ['', 'l'], t, self.J),  # awesome, fulsome
                AffixRule(MorphTag.J_SOME, 'some', [''], t, self.N),  # troublesome

                AffixRule(MorphTag.J_WISE, 'wise', [''], t, self.J),  # likewise
                AffixRule(MorphTag.J_WISE, 'wise', [''], t, self.N),  # clockwise

                AffixRule(MorphTag.J_Y, 'ey', [''], t, self.N),  # clayey
                AffixRule(MorphTag.J_Y, 'y', ['', 'e'], t, self.V, True),  # grouchy, runny, rumbly
                AffixRule(MorphTag.J_Y, 'y', ['', 'e'], t, self.N, True),  # dreamy, skinny, juicy
            ],
            self.R: [
                AffixRule(MorphTag.R_LY, 'ally', [''], t, self.J),  # electronically
                AffixRule(MorphTag.R_LY, 'ily', ['y'], t, self.J),  # easily
                AffixRule(MorphTag.R_LY, 'ly', ['', 'l', 'le'], t, self.J),  # sadly, fully, incredibly
            ]
        }

    def _init_prefixes(self) -> Dict[str, marisa_trie.Trie]:
        v = ['be', 'co', 'de', 'dis', 'fore', 'inter', 'mis', 'out', 'over', 'pre', 're', 'sub', 'trans', 'un', 'under']
        n = ['anti', 'auto', 'bi', 'co', 'counter', 'dis', 'ex', 'hyper', 'in', 'inter', 'kilo', 'mal', 'mega', 'mis', 'mini', 'mono', 'neo', 'out', 'poly', 'pseudo', 're', 'semi', 'sub', 'super', 'sur', 'tele', 'tri', 'ultra', 'under', 'vice']
        j = ['dis', 'il', 'im', 'in', 'ir', 'non', 'un']

        return {
            self.V: marisa_trie.Trie(v + [e + '-' for e in v]),
            self.N: marisa_trie.Trie(n + [e + '-' for e in n]),
            self.J: marisa_trie.Trie(v + [e + '-' for e in j]),
        }

    def _analyze_base(self, lex: SimpleNamespace, token: str, pos: str = None) -> Optional[List[Tuple[str, str]]]:
        """
        :param lex: a lexicon item from :attr:`EnglishMorphAnalyzer._lexicons`.
        :param token: the input token in lower cases.
        :param pos: the part-of-speech tag of the input token if available.
        :return: if the input token is in a base form, the lemma and its part-of-speech tag; otherwise, None.
        """
        if pos is None:
            return [(token, lex.stem_tag)] if token in lex.base_set else None
        if pos in lex.stem_tagset:
            return [(token, lex.stem_tag)]

    def _analyze_inflection(self, lex: SimpleNamespace, token: str, pos: str = None) -> Optional[List[Tuple[str, str]]]:
        """
        :param rules: a dictionary of inflection rules (see :attr:`EnglishMorphAnalyzer._inflection_rules`).
        :param lex: a lexicon item from :attr:`EnglishMorphAnalyzer._lexicons`.
        :param token: the input token in lower cases.
        :param pos: the part-of-speech tag of the input token if available.
        :return: if the input token matches an inflection rule, the lemma, inflection suffixes and their pos tags; otherwise, None.
        """
        if pos is not None and pos not in lex.infl_tagset: return None
        lemma = lex.exc_dict.get(token, None)
        if lemma is not None: return [(lemma, lex.stem_tag), (extract_suffix(token, lemma), lex.affix_tag(token))]

        for rule in self._inflection_rules[lex.stem_tag]:
            lemma = suffix_matcher(rule, lex.base_set, token, pos)
            if lemma is not None: return [(lemma, lex.stem_tag), ('+' + rule.affix_form, rule.affix_tag)]

        return None

    def _analyze_derivation(self, tp: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """
        :param tp:
        """
        token, pos = tp[0]
        if token in self._derivation_exc: return tp

        for rule in self._derivation_rules[pos]:
            lemma = suffix_matcher(rule, self._lexicons[rule.stem_tag].base_set, token)
            if lemma is not None:
                return self._analyze_derivation([(lemma, rule.stem_tag), ('+' + rule.affix_form, rule.affix_tag)] + tp[1:])

        if len(tp) == 1:
            t = self._analyze_inflection(self._lexicons[self.V], token, 'VBN')
            if t is not None: tp = [t[0], (t[1][0], MorphTag.J_ED)]

        return tp

    def _analyze_prefix(self, tp: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        token, pos = tp[0]

        if pos in self._prefixes:
            t = self._prefixes[pos].prefixes(token)
            if t:
                prefix = max(t, key=len)
                stem = token[len(prefix):]
                if stem in self._lexicons[pos].base_set:
                    prefix = prefix[:-1] if prefix.endswith('-') else prefix
                    tag = 'P_' + prefix.upper()
                    prefix += '+'
                    return self._analyze_prefix([(stem, pos)] + tp[1:] + [(prefix, tag)])

        return tp

    def analyze(self, token: str, pos: str = None, derivation=False) -> List[List[Tuple[str, str]]]:
        token = token.lower()
        t = self._irregular_rules.get((token, pos), None)
        if t is not None: return t

        morphs = []
        for lex in self._lexicons.values():
            t = self._analyze_base(lex, token, pos)
            if t is not None: morphs.append(t)

            t = self._analyze_inflection(lex, token, pos)
            if t is not None: morphs.append(t)

        if derivation:
            for i, morph in enumerate(morphs):
                morphs[i] = self._analyze_derivation(morph[:1]) + morph[1:]

        # if len(morphs) == 1 and len(morphs[0]) == 1 and morphs[0][0] == token: del morphs[:]
        return morphs
