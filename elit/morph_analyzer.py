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
import json
import os
import re
from marisa_trie import BytesTrie
from types import SimpleNamespace
from typing import Sequence, Set, List, Optional, Tuple, Dict, Any

from pkg_resources import resource_filename

from elit.util import io
from elit.util.io import read_word_set

__author__ = 'Jinho D. Choi'


class AffixTag:
    # lemma
    V = 'V'  # verb
    N = 'N'  # noun
    J = 'J'  # adjective
    R = 'R'  # adverb
    M = 'M'  # modal
    T = 'T'  # TO

    # inflection: verb
    I_3PS = 'I_3PS'  # 3rd-person singular present
    I_GRD = 'I_GRD'  # gerund
    I_PST = 'I_PST'  # past

    # inflection: noun
    I_PLR = 'I_PLR'  # plural

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


class NoIndent(object):
    def __init__(self, value):
        self.value = value


class NoIndentEncoder(json.JSONEncoder):
    REGEX = re.compile(r'@@@(\d+)@@@')

    def __init__(self, *args, **kwargs):
        super(NoIndentEncoder, self).__init__(*args, **kwargs)
        self.kwargs = dict(kwargs)
        del self.kwargs['indent']
        self._replacements = {}

    def default(self, o):
        if isinstance(o, NoIndent):
            key = len(self._replacements)
            self._replacements[key] = json.dumps(o.value, **self.kwargs)
            return "@@@%d@@@" % key
        else:
            return super(NoIndentEncoder, self).default(o)

    def encode(self, o):
        result = super(NoIndentEncoder, self).encode(o)
        out = []
        m = self.REGEX.search(result)
        while m:
            key = int(m.group(1))
            out.append(result[:m.start(0) - 1])
            out.append(self._replacements[key])
            result = result[m.end(0) + 1:]
            m = self.REGEX.search(result)
        return ''.join(out)


class AffixRule:
    CK = {'c', 'k'}

    def __init__(self, affix_tag: str, affix: str, token_affixes: Sequence[str], token_tags: Set[str] = None, lemma_tag: str = None, double_consonants: bool = False):
        """
        :param affix: the affix (e.g., `ies` as in `stud+ies`).
        :param affix_tag: the tag of the affix (:class:`AffixTag`).
        :param token_affixes: the affixes appended to the stem to recover the lemma (e.g., `y` as in `stud+y`).
        :param token_tags: the set of possible part-of-speech tags for the input token.
        :param lemma_tag: the part-of-speech tag of the lemma.
        :param double_consonants: if ``True``, double consonants are considered to find the stem (e.g., `run+n+ing`).
        """
        self.affix = affix
        self.affix_tag = affix_tag
        self.token_affixes = token_affixes
        self.token_tags = token_tags
        self.lemma_tag = lemma_tag
        self.double_consonants = double_consonants

    @classmethod
    def is_double_consonants(cls, s: str) -> bool:
        """
        :param s: an input string.
        :return: ``True`` if the input string ends with double consonants (e.g., run+n)
        """
        return len(s) >= 4 and (s[-1] == s[-2] or (s[-1] in cls.CK and s[-2] in cls.CK))

    @classmethod
    def factory(cls, d: Dict[str, Any]):
        """
        :param d: the return value of :meth:`AffixRule.to_json_obj`.
        :return: an affix rule (object of this class) containing all information from the json object.
        """
        return AffixRule(d['affix_tag'], d['affix'], d['token_affixes'], set(d['token_tags']), d['lemma_tag'], d['double_consonants'])

    def to_json_obj(self) -> Dict[str, Any]:
        """
        :return: the json-compatible object containing all information about this rule.
        """
        return {
            'affix_tag': self.affix_tag,
            'affix': self.affix,
            'token_affixes': self.token_affixes,
            'token_tags': list(self.token_tags),
            'lemma_tag': self.lemma_tag,
            'double_consonants': self.double_consonants
        }


def extract_suffix(token: str, lemma: str) -> str:
    """
    :param token: the input token.
    :param lemma: the lemma of the token.
    :return: the suffix in the input token varied from the lemma (e.g., `ies` in `stud+ies` and `study`)
    """
    i = len(os.path.commonprefix((token, lemma)))
    if i == 0: return ''
    tsuf = token[i:]
    lsuf = lemma[i:]
    if i < len(token) and AffixRule.is_double_consonants(token[:i + 1]): tsuf = token[i + 1:]

    j = len(os.path.commonprefix((tsuf[::-1], lsuf[::-1])))
    if j == 0: return '-' + lsuf if not tsuf else '+' + tsuf

    tsuf = tsuf[:-j]
    lsuf = lsuf[:-j]
    return '-' + lsuf + '-' if not tsuf else '+' + tsuf + '+'


def suffix_matcher(rule: AffixRule, base_set: Set[str], token: str, pos: str = None) -> Optional[str]:
    """
    :param rule: the affix rule.
    :param base_set: the set including base forms (lemmas).
    :param token: the input token.
    :param pos: the part-of-speech tag of the input token.
    :return: the lemma of the input token if it matches to the affix rule; otherwise, ``None``.
    """

    def match(s: str):
        for suffix in rule.token_affixes:
            base = s + suffix
            if base in base_set: return base
        return None

    if len(rule.affix) > len(token) or not token.endswith(rule.affix): return None
    if pos is not None and pos not in rule.token_tags: return None
    stem = token[:-len(rule.affix)]

    lemma = match(stem)
    if lemma is not None: return lemma

    if rule.double_consonants and AffixRule.is_double_consonants(stem):
        lemma = match(stem[:-1])
        if lemma is not None: return lemma

    return None


class EnglishMorphAnalyzer:
    _RES_INFLECTION_LOOKUP = 'inflection_lookup.json'
    _RES_INFLECTION_RULES = 'inflection_rules.json'

    _RES_PREFIX = 'prefix.txt'
    _RES_PREFIX_EX = 'prefix_lookup.json'
    _RES_DERIVATION_RULES = 'derivation_rules.json'

    PREFIX_NONE = 0
    PREFIX_LONGEST = 1
    PREFIX_SHORTEST = 2

    def __init__(self):
        self._base_lookup = {'VB': AffixTag.V, 'VBP': AffixTag.V, 'NN': AffixTag.N, 'NNP': AffixTag.N, 'JJ': AffixTag.J, 'RB': AffixTag.R}

        # initialize resources
        resource_path = 'elit.resources.lemmatizer.english'
        self._inflection_lookup = self._load_inflection_lookup(resource_filename(resource_path, self._RES_INFLECTION_LOOKUP))
        self._inflection_rules = self._load_affix_rules(resource_filename(resource_path, self._RES_INFLECTION_RULES))




        self._lexicons = self._init_lexicons(resource_path)
        self._derivation_exc = read_word_set(resource_filename(resource_path, 'derivation.exc'))

        # initialize rules
        self._prefix = self._load_prefix(resource_filename(resource_path, self._RES_PREFIX))
        self._prefix_ex = self._load_prefix_ex(resource_filename(resource_path, self._RES_PREFIX_EX))


        self._derivation = self._load_affix_rules(resource_filename(resource_path, self._RES_DERIVATION_RULES))

    def load(self, resource_path):
        pass

    @classmethod
    def _load_simple_rules(cls, filename: str) -> Dict[str, List[Tuple[str, str]]]:
        with open(filename) as fin: d = json.load(fin)
        return {k: list(map(tuple, v)) for k, v in d.items()}

    @classmethod
    def _load_affix_rules(self, filename: str) -> Dict[str, List[AffixRule]]:
        with open(filename) as fin: d = json.load(fin)
        return {pos: [AffixRule.factory(rule) for rule in rules] for pos, rules in d.items()}

    @classmethod
    def _load_inflection_lookup(cls, filename) -> Dict[str, List[Tuple[str, str]]]:
        return cls._load_simple_rules(filename)

    @classmethod
    def _load_prefix(cls, filename: str) -> BytesTrie:
        prefixes = read_word_set(filename)
        prefixes = [e.split() for e in prefixes]
        return BytesTrie([(p[0], p[1].encode('utf-8')) for p in prefixes] + [(p[0] + '-', p[1].encode('utf-8')) for p in prefixes])

    @classmethod
    def _load_prefix_ex(cls, filename: str) -> Dict[str, List[Tuple[str, str]]]:
        return cls._load_simple_rules(filename)







    @classmethod
    def _save_simple_rules(cls, filename: str, d: Dict[str, List[Tuple[str, str]]]):
        with open(filename, 'w') as fout:
            d = {k: NoIndent(v) for k, v in d.items()}
            d = json.dumps(d, cls=NoIndentEncoder, indent=2) + '\n}'
            fout.write(d)

    @classmethod
    def _save_affix_rules(cls, filename, rule_dict: Dict[str, List[AffixRule]]):
        with open(filename, 'w') as fout:
            d = {pos: [NoIndent(rule.to_json_obj()) for rule in rules] for pos, rules in rule_dict.items()}
            d = json.dumps(d, cls=NoIndentEncoder, indent=2) + '\n  ]\n}'
            fout.write(d)

    def _save_inflection_lookup(self, filename):
        self._save_simple_rules(filename, self._inflection_lookup)

    def _save_prefix(self, filename: str):
        with open(filename, 'w') as fout:
            for prefix, tag in sorted(self._prefix.items()):
                if not prefix.endswith('-'): fout.write(prefix + ' ' + tag.decode('utf-8') + '\n')

    def _save_prefix_ex(self, filename: str):
        self._save_simple_rules(filename, self._prefix_ex)





    def _init_lexicons(self, resource_path: str) -> Dict[str, SimpleNamespace]:
        return {
            AffixTag.V: SimpleNamespace(
                base_set=io.read_word_set(resource_filename(resource_path, 'verb.base')),
                exc_dict=io.read_word_dict(resource_filename(resource_path, 'verb.exc')),
                stem_tag=AffixTag.V,
                infl_tagset={'VBD', 'VBG', 'VBN', 'VBZ'},
                affix_tag=lambda x: AffixTag.I_GRD if x.endswith('ing') else AffixTag.I_3PS if x.endswith('s') else AffixTag.I_PST),
            AffixTag.N: SimpleNamespace(
                base_set=io.read_word_set(resource_filename(resource_path, 'noun.base')),
                exc_dict=io.read_word_dict(resource_filename(resource_path, 'noun.exc')),
                stem_tag=AffixTag.N,
                infl_tagset={'NNS', 'NNPS'},
                affix_tag=lambda x: AffixTag.I_PLR),
            AffixTag.J: SimpleNamespace(
                base_set=io.read_word_set(resource_filename(resource_path, 'adjective.base')),
                exc_dict=io.read_word_dict(resource_filename(resource_path, 'adjective.exc')),
                stem_tag=AffixTag.J,
                infl_tagset={'JJR', 'JJS'},
                affix_tag=lambda x: AffixTag.I_SUP if x.endswith('st') else AffixTag.I_COM),
            AffixTag.R: SimpleNamespace(
                base_set=io.read_word_set(resource_filename(resource_path, 'adverb.base')),
                exc_dict=io.read_word_dict(resource_filename(resource_path, 'adverb.exc')),
                stem_tag=AffixTag.R,
                infl_tagset={'RBR', 'RBS'},
                affix_tag=lambda x: AffixTag.I_SUP if x.endswith('st') else AffixTag.I_COM)
        }

    def analyze(self, token: str, pos: str = None, derivation=False) -> List[Tuple[str, str]]:
        token = token.lower()

        # inflection lookup
        morphs = self._analyze_inflection_lookup(token, pos)

        # base lookup
        if morphs is None:
            morphs = self._analyze_base_lookup(token, pos)

        # inflection
        if morphs is None:
            morphs = self._analyze_inflection_rules(token, pos)

        morphs = []
        for lex in self._lexicons.values():

            t = self._analyze_inflection_rules(lex, token, pos)
            if t is not None: morphs.append(t)

        if derivation:
            for i, morph in enumerate(morphs):
                morphs[i] = self._analyze_derivation(morph[:1]) + morph[1:]

        # if len(morphs) == 1 and len(morphs[0]) == 1 and morphs[0][0] == token: del morphs[:]
        return morphs

    def _analyze_base_lookup(self, token: str, pos: str) -> Optional[List[Tuple[str, str]]]:
        pos = self._base_lookup.get(pos, None)
        return [(token, pos)] if pos is not None else None

    def _analyze_inflection_lookup(self, token: str, pos: str) -> Optional[List[Tuple[str, str]]]:
        return self._inflection_lookup.get(token + ' ' + pos, None)

    def _analyze_inflection_rules(self, token: str, pos: str) -> Optional[List[Tuple[str, str]]]:
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
            if lemma is not None: return [(lemma, lex.stem_tag), ('+' + rule.affix, rule.affix_tag)]

        return None

    def _analyze_derivation(self, tp: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """
        :param tp:
        """
        token, pos = tp[0]
        if token in self._derivation_exc: return tp

        for rule in self._derivation[pos]:
            lemma = suffix_matcher(rule, self._lexicons[rule.lemma_tag].base_set, token)
            if lemma is not None:
                return self._analyze_derivation([(lemma, rule.lemma_tag), ('+' + rule.affix, rule.affix_tag)] + tp[1:])

        if len(tp) == 1:
            t = self._analyze_inflection_rules(self._lexicons[AffixTag.V], token, 'VBN')
            if t is not None: tp = [t[0], (t[1][0], AffixTag.J_ED)]

        return tp

    def _analyze_prefix(self, token: str, pos: str, option=PREFIX_SHORTEST) -> List[Tuple[str, str]]:
        def prefix_stem_pos(prefix: str) -> Optional[Tuple[str, str, str]]:
            stem = token[len(prefix):]
            if prefix == 'be':  # be+little
                for p in [AffixTag.V, AffixTag.J, AffixTag.N]:
                    if stem in self._lexicons[p].base_set: return prefix, stem, p

            return (prefix, stem, pos) if stem in self._lexicons[pos].base_set else None

        def get_key(psp):
            return len(psp[0])

        # none option
        if option == self.PREFIX_NONE: return [(token, pos)]

        # exception
        t = self._prefix_ex.get(token + ' ' + pos, None)
        if t is not None: return t

        # prefix matching
        prefixes = self._prefix.prefixes(token)
        if not prefixes: return [(token, pos)]

        # stem matching
        t = [prefix_stem_pos(prefix) for prefix in prefixes]
        t = [psp for psp in t if psp is not None]
        if not t: return [(token, pos)]

        prefix, lemma, pos = max(t, key=get_key) if option == self.PREFIX_LONGEST else min(t, key=get_key)
        tag = '|'.join([x.decode('utf-8') for x in self._prefix[prefix]])
        prefix = prefix[:-1] if prefix.endswith('-') else prefix

        t = self._analyze_prefix(lemma, pos, option)
        return [t[0], (prefix + '+', tag)] + t[1:]
