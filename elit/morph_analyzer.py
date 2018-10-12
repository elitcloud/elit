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
from types import SimpleNamespace
from typing import Sequence, Set, List, Optional, Tuple, Dict

from pkg_resources import resource_filename

from elit.util import io

__author__ = 'Jinho D. Choi'


class MorphTag:
    # inflection: verb
    TPS = '3PS'  # 3rd-person singular present
    GER = 'GER'  # gerund
    PAS = 'PAS'  # past

    # inflection: noun
    PLU = 'PLU'  # plural

    # inflection: adjective/adverb
    COM = 'COM'  # comparative
    SUP = 'SUP'  # superlative

    # derivation: verb
    # DSV = 'DSV'  # to verb
    # DSN = 'DSN'  # to noun
    # DSN = 'DSN'  # to noun
    # DSN = 'DSN'  # to noun


class AffixRule:
    def __init__(self, affix_tag: str, affix_form: str, token_tagset: Set[str], stem_tag: str, stem_affixes: Sequence[str], double_consonants: bool = False):
        """
        :param affix_tag: the morphology tag of the affix (see :class:`MorphTag`).
        :param affix_form: the form of the affix (e.g., 'ies', 'es', 's').
        :param token_tagset: the set of possible part-of-speech tags (Penn Treebank style) for the input token.
        :param stem_tag: the part-of-speech tag (Penn Treebank style) of the stem.
        :param stem_affixes: the affixes appended to the stem to recover the lemma (e.g., 'y').
        :param double_consonants: if True, the affix forms with double consonants are considered (e.g., run+n+ing).
        """
        self.affix_tag = affix_tag
        self.affix_form = affix_form
        self.token_tagset = token_tagset
        self.stem_tag = stem_tag
        self.stem_affixes = stem_affixes
        self.double_consonants = double_consonants


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

    def double_consonants(s: str):
        return len(s) >= 4 and s[-1] == s[-2]

    if len(rule.affix_form) > len(token) or not token.endswith(rule.affix_form): return None
    if pos is not None and pos not in rule.token_tagset: return None
    stem = token[:-len(rule.affix_form)]

    lemma = match(stem)
    if lemma is not None: return lemma

    if rule.double_consonants and double_consonants(stem):
        lemma = match(stem[:-1])
        if lemma is not None: return lemma

    return None


def extract_suffix(token: str, lemma: str) -> str:
    i = 0
    for c in token:
        if i >= len(lemma) or c != lemma[i]: break
        i += 1

    if i == 0 or i >= len(token): return ''
    if i > 2 and token[i] == token[i-1]: i += 1
    return token[i:]
    # return 'en' if suffix.endswith('en') else 'er' if suffix.endswith('er') else 'est' if suffix.endswith('est') else ''


class EnglishMorphAnalyzer:
    def __init__(self):
        # initialize _lexicons
        resource_path = 'elit.resources.lemmatizer.english'
        self._lexicons = self._init_lexicons(resource_path)

        # initialize rules
        self._irregular_rules = self._init_irregular_rules()
        self._inflection_rules = self._init_inflection_rules()
        self._derivation_rules = self._init_derivation_rules()
        self._base_tagset = {'VB', 'VBP', 'NN', 'JJ', 'RB'}

    @classmethod
    def _init_lexicons(cls, resource_path: str):
        return [
            SimpleNamespace(base_set=io.read_word_set(resource_filename(resource_path, 'verb.base')),
                            exc_dict=io.read_word_dict(resource_filename(resource_path, 'verb.exc')),
                            token_tagset={'VBD', 'VBG', 'VBN', 'VBZ'},
                            stem_tag='VB',
                            affix_tag=lambda x: MorphTag.GER if x.endswith('ing') else MorphTag.TPS if x.endswith('s') else MorphTag.PAS),
            SimpleNamespace(base_set=io.read_word_set(resource_filename(resource_path, 'noun.base')),
                            exc_dict=io.read_word_dict(resource_filename(resource_path, 'noun.exc')),
                            token_tagset={'NNS'},
                            stem_tag='NN',
                            affix_tag=lambda x: MorphTag.PLU),
            SimpleNamespace(base_set=io.read_word_set(resource_filename(resource_path, 'adjective.base')),
                            exc_dict=io.read_word_dict(resource_filename(resource_path, 'adjective.exc')),
                            token_tagset={'JJR', 'JJS'},
                            stem_tag='JJ',
                            affix_tag=lambda x: MorphTag.SUP if x.endswith('st') else MorphTag.COM),
            SimpleNamespace(base_set=io.read_word_set(resource_filename(resource_path, 'adverb.base')),
                            exc_dict=io.read_word_dict(resource_filename(resource_path, 'adverb.exc')),
                            token_tagset={'RBR', 'RBS'},
                            stem_tag='RB',
                            affix_tag=lambda x: MorphTag.SUP if x.endswith('st') else MorphTag.COM)]

    @classmethod
    def _init_irregular_rules(cls) -> Dict[Tuple[str, str], List[List[Tuple[str, str]]]]:
        return {
            ("n't", 'RB'): [[('not', 'RB')]],
            ("n't", None): [[('not', 'RB')]],
            ("'nt", 'RB'): [[('not', 'RB')]],
            ("'nt", None): [[('not', 'RB')]],
            ("'d", 'MD'): [[('would', 'MD')]],
            ("'ll", 'MD'): [[('will', 'MD')]],
            ("'ll", None): [[('will', 'MD')]],
            ('ca', 'MD'): [[('can', 'MD')]],
            ('na', 'TO'): [[('to', 'TO')]],
            ("'d", 'VBD'): [[('have', 'VB'), ('d', MorphTag.PAS)]],
            ("'d", None): [[('have', 'VB'), ('d', MorphTag.PAS)], [('would', 'MD')]],
            ("'ve", 'VBP'): [[('have', 'VB')]],
            ("'ve", None): [[('have', 'VB')]],
            ("'m", 'VBP'): [[('be', 'VB')]],
            ("'m", None): [[('be', 'VB')]],
            ("'mmm", 'VBP'): [[('be', 'VB')]],
            ("'re", 'VBP'): [[('be', 'VB')]],
            ("'re", None): [[('be', 'VB')]],
            ('ai', 'VBP'): [[('be', 'VB')]],
            ('am', 'VBP'): [[('be', 'VB')]],
            ('am', None): [[('be', 'VB')], [('am', 'NN')]],
            ('are', 'VBP'): [[('be', 'VB')]],
            ('are', None): [[('be', 'VB')]],
            ('is', 'VBZ'): [[('be', 'VB'), ('', MorphTag.TPS)]],
            ('is', None): [[('be', 'VB'), ('', MorphTag.TPS)]],
            ('was', 'VBD'): [[('be', 'VB'), ('', MorphTag.TPS), ('', MorphTag.PAS)]],
            ('was', None): [[('be', 'VB'), ('', MorphTag.TPS), ('', MorphTag.PAS)]],
            ('were', 'VBD'): [[('be', 'VB'), ('', MorphTag.PAS)]],
            ('were', None): [[('be', 'VB'), ('', MorphTag.PAS)]],
            ('gon', 'VBG'): [[('go', 'VB'), ('ing', MorphTag.GER)]],
        }

    @classmethod
    def _init_inflection_rules(cls) -> Dict[str, List[AffixRule]]:
        TPS = {'VBZ'}
        GER = {'VBG'}
        PAS = {'VBD', 'VBN'}
        PLU = {'NNS', 'NNPS'}
        JCO = {'JJR'}
        JSU = {'JJS'}
        RCO = {'RBR'}
        RSU = {'RBS'}

        VB = 'VB'
        NN = 'NN'
        JJ = 'JJ'
        RB = 'RB'

        return {
            'VB': [
                AffixRule(MorphTag.TPS, 'ies', TPS, VB, ['y']),
                AffixRule(MorphTag.TPS, 'es', TPS, VB, ['']),
                AffixRule(MorphTag.TPS, 's', TPS, VB, ['']),
                AffixRule(MorphTag.GER, 'ying', GER, VB, ['ie']),
                AffixRule(MorphTag.GER, 'ing', GER, VB, ['', 'e'], True),
                AffixRule(MorphTag.PAS, 'ied', PAS, VB, ['y']),
                AffixRule(MorphTag.PAS, 'ed', PAS, VB, [''], True),
                AffixRule(MorphTag.PAS, 'd', PAS, VB, ['']),
                AffixRule(MorphTag.PAS, 'en', PAS, VB, ['', 'e'], True),
                AffixRule(MorphTag.PAS, 'n', PAS, VB, ['']),
                AffixRule(MorphTag.PAS, 'ung', PAS, VB, ['ing'])],
            'NN': [
                AffixRule(MorphTag.PLU, 'ies', PLU, NN, ['y']),
                AffixRule(MorphTag.PLU, 'es', PLU, NN, ['']),
                AffixRule(MorphTag.PLU, 's', PLU, NN, ['']),
                AffixRule(MorphTag.PLU, 'men', PLU, NN, ['man']),
                AffixRule(MorphTag.PLU, 'ae', PLU, NN, ['a']),
                AffixRule(MorphTag.PLU, 'i', PLU, NN, ['us']),
                AffixRule(MorphTag.PLU, 'a', PLU, NN, ['um'])],
            'JJ': [
                AffixRule(MorphTag.COM, 'ier', JCO, JJ, ['y']),
                AffixRule(MorphTag.COM, 'er', JCO, JJ, ['e']),
                AffixRule(MorphTag.COM, 'er', JCO, JJ, [''], True),
                AffixRule(MorphTag.SUP, 'iest', JSU, JJ, ['y']),
                AffixRule(MorphTag.SUP, 'est', JSU, JJ, ['e']),
                AffixRule(MorphTag.SUP, 'est', JSU, JJ, [''], True)],
            'RB': [
                AffixRule(MorphTag.COM, 'ier', RCO, RB, ['y']),
                AffixRule(MorphTag.COM, 'er', RCO, RB, ['', 'e']),
                AffixRule(MorphTag.SUP, 'iest', RSU, RB, ['y']),
                AffixRule(MorphTag.SUP, 'est', RSU, RB, ['', 'e'])]
        }

    @classmethod
    def _init_derivation_rules(cls) -> List[AffixRule]:
        return [
            # AffixRule(AffixType.SUFFIX, AffixTag.DSV, 'ify',   MorphPOS.V, MorphPOS.N, ['', 'y', 'or', 'ity']),  # fortify -> fort, glorify -> glory, terrify -> terror, qualify -> quality
            # AffixRule(AffixType.SUFFIX, AffixTag.DSV, 'ize',   MorphPOS.V, MorphPOS.N, ['', 'y']),               # terrorize -> terror, tyrannize -> tyranny
            # AffixRule(AffixType.SUFFIX, AffixTag.DSV, 'ise',   MorphPOS.V, MorphPOS.N, ['', 'y']),               # terrorise -> terror, tyrannise -> tyranny
            #
            # AffixRule(AffixType.SUFFIX, AffixTag.DSV, 'ify',   MorphPOS.V, MorphPOS.J, ['e', 'y']),              # simplify -> simple, fancify -> fancy
            # AffixRule(AffixType.SUFFIX, AffixTag.DSV, 'ize',   MorphPOS.V, MorphPOS.J, [''], True),              # normalize -> normal, crystallize -> crystal
            # AffixRule(AffixType.SUFFIX, AffixTag.DSV, 'ise',   MorphPOS.V, MorphPOS.J, [''], True),              # normalise -> normal, crystallise -> crystal
            #
            # AffixRule(AffixType.SUFFIX, AffixTag.DSN, 'al',    MorphPOS.N, MorphPOS.V, ['e']),                   # arrival -> arrive
            # AffixRule(AffixType.SUFFIX, AffixTag.DSN, 'iance', MorphPOS.N, MorphPOS.V, ['y']),                   # defiance -> defy
            # AffixRule(AffixType.SUFFIX, AffixTag.DSN, 'ance',  MorphPOS.N, MorphPOS.V, ['', 'e', 'ate']),        # annoyance -> annoy, insurance -> insure, deviance -> deviate
            # AffixRule(AffixType.SUFFIX, AffixTag.DSN, 'ant',   MorphPOS.N, MorphPOS.V, ['', 'e', 'ate']),        # assistant -> assist, servant -> serve, immigrant -> immigrate
            # AffixRule(AffixType.SUFFIX, AffixTag.DSN, 'ent',   MorphPOS.N, MorphPOS.V, ['', 'e', 'y']),          # dependent -> depend, resident -> reside, student -> study
            # AffixRule(AffixType.SUFFIX, AffixTag.DSN, 'ying',  MorphPOS.N, MorphPOS.V, ['ie']),                  # lying -> lie
            # AffixRule(AffixType.SUFFIX, AffixTag.DSN, 'ing',   MorphPOS.N, MorphPOS.V, ['', 'e'], True),         # building -> build, striking -> strike, running -> run
            #
            # AffixRule(AffixType.SUFFIX, AffixTag.DSN, 'ier',   MorphPOS.N, MorphPOS.V, ['y']),                   # carrier -> carry
            #
            #
            # AffixRule(AffixType.SUFFIX, AffixTag.DSN, 'ance',  MorphPOS.N, MorphPOS.J, ['ant']),                 # relevance -> relevant
            # AffixRule(AffixType.SUFFIX, AffixTag.DSN, 'ancy',  MorphPOS.N, MorphPOS.J, ['ant']),                 # pregnancy -> pregnant
            # AffixRule(AffixType.SUFFIX, AffixTag.DSN, 'ence',  MorphPOS.N, MorphPOS.J, ['ent']),                 # difference -> different
            # AffixRule(AffixType.SUFFIX, AffixTag.DSN, 'ency',  MorphPOS.N, MorphPOS.J, ['ent']),                 # fluency -> fluent
            # AffixRule(AffixType.SUFFIX, AffixTag.DSN, 'acy',   MorphPOS.N, MorphPOS.J, ['ate']),                 # accuracy -> accurate
        ]

    def _base(self, lex: SimpleNamespace, token: str, pos: str = None) -> Optional[List[Tuple[str, str]]]:
        if (pos is None or pos == lex.stem_tag) and token in lex.base_set: return [(token, lex.stem_tag)]
        if pos in self._base_tagset: return [(token, pos)]

    def _inflection(self, lex: SimpleNamespace, token: str, pos: str = None) -> Optional[List[Tuple[str, str]]]:
        if pos is not None and pos not in lex.token_tagset: return None
        lemma = lex.exc_dict.get(token, None)
        if lemma is not None: return [(lemma, lex.stem_tag), (extract_suffix(token, lemma), lex.affix_tag(token))]

        for rule in self._inflection_rules[lex.stem_tag]:
            lemma = suffix_matcher(rule, lex.base_set, token, pos)
            if lemma is not None: return [(lemma, rule.stem_tag), (rule.affix_form, rule.affix_tag)]

        return None

    def analyze(self, token: str, pos: str = None) -> List[List[Tuple[str, str]]]:
        token = token.lower()
        t = self._irregular_rules.get((token, pos), None)
        if t is not None: return t

        morphs = []
        for lex in self._lexicons:
            t = self._base(lex, token, pos)
            if t is not None: morphs.append(t)

            t = self._inflection(lex, token, pos)
            if t is not None: morphs.append(t)

        return morphs
