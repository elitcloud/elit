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
import json
import os
from marisa_trie import BytesTrie
from types import SimpleNamespace
from typing import Sequence, Set, List, Optional, Tuple, Dict, Any

from pkg_resources import resource_filename

from elit.component import NLPComponent
from elit.structure import Document, MORPH
from elit.util import io
from elit.util.io import read_word_set, NoIndent, NoIndentEncoder

__author__ = 'Jinho D. Choi'


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


def suffix_matcher(rule: AffixRule, lemma_set: Set[str], token: str, pos: str = None) -> Optional[str]:
    """
    :param rule: the affix rule.
    :param lemma_set: the set including base forms (lemmas).
    :param token: the input token.
    :param pos: the part-of-speech tag of the input token.
    :return: the lemma of the input token if it matches to the affix rule; otherwise, ``None``.
    """

    def match(s: str):
        for suffix in rule.token_affixes:
            base = s + suffix
            if base in lemma_set: return base
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


class EnglishMorphAnalyzer(NLPComponent):
    PREFIX_NONE = 0
    PREFIX_LONGEST = 1
    PREFIX_SHORTEST = 2

    def __init__(self):
        self._base_lookup = self._init_base_lookup()
        self._inflection_lexicons = None
        self._inflection_lookup = None
        self._inflection_rules = None
        self._prefix = None
        self._prefix_lookup = None
        self._derivation_lookup = None
        self._derivation_rules = None
        self.init()

    def init(self):
        self.load('elit.resources.morph_analyzer.english')

    def load(self, resource_path: str, **kwargs):
        self._inflection_lexicons = self._load_inflection_lexicons(resource_path)
        self._inflection_lookup = self._load_inflection_lookup(resource_path)
        self._inflection_rules = self._load_inflection_rules(resource_path)
        self._prefix = self._load_prefix(resource_path)
        self._prefix_lookup = self._load_prefix_lookup(resource_path)
        self._derivation_lookup = self._load_derivation_lookup(resource_path)
        self._derivation_rules = self._load_affix_rules(resource_filename(resource_path, 'derivation_rules.json'))

    def decode(self, docs: Sequence[Document], derivation=True, prefix=PREFIX_NONE):
        for doc in docs:
            for sen in doc:
                sen[MORPH] = [self.analyze(token, pos, derivation, prefix) for token, pos in zip(sen.tokens, sen.part_of_speech_tags)]

    def save(self, **kwargs):
        pass

    def train(self, trn_docs: Sequence[Document], dev_docs: Sequence[Document], model_path: str, **kwargs) -> float:
        pass

    def evaluate(self, docs: Sequence[Document], **kwargs):
        pass

    @classmethod
    def _init_base_lookup(cls) -> Dict[str, str]:
        return {'VB': EnglishMorphTag.VB, 'VBP': EnglishMorphTag.VB, 'NN': EnglishMorphTag.NN, 'NNP': EnglishMorphTag.NN, 'JJ': EnglishMorphTag.JJ, 'RB': EnglishMorphTag.RB}

    @classmethod
    def _load_simple_rules(cls, filename: str) -> Dict[str, List[Tuple[str, str]]]:
        with open(filename) as fin: d = json.load(fin)
        return {k: list(map(tuple, v)) for k, v in d.items()}

    @classmethod
    def _load_affix_rules(cls, filename: str) -> Dict[str, List[AffixRule]]:
        with open(filename) as fin: d = json.load(fin)
        return {pos: [AffixRule.factory(rule) for rule in rules] for pos, rules in d.items()}

    def _load_inflection_lexicons(self, resource_path: str) -> Dict[str, SimpleNamespace]:
        # vv_tag = lambda x: EnglishMorphTag.I_GRD if x.endswith('ing') else EnglishMorphTag.I_3PS if x.endswith('s') else EnglishMorphTag.I_PST
        # nn_tag = lambda x: EnglishMorphTag.I_PLR),
        # jj_tag = lambda x: EnglishMorphTag.I_SUP if x.endswith('st') else EnglishMorphTag.I_COM
        # rb_tag = lambda x: EnglishMorphTag.I_SUP if x.endswith('st') else EnglishMorphTag.I_COM

        return {
            EnglishMorphTag.VB: SimpleNamespace(
                lemma_set=io.read_word_set(resource_filename(resource_path, 'verb.txt')),
                exc_dict=self._load_simple_rules(resource_filename(resource_path, 'inflection_lookup_verb.json')),
                token_tagset={'VBD', 'VBG', 'VBN', 'VBZ'}),
            EnglishMorphTag.NN: SimpleNamespace(
                lemma_set=io.read_word_set(resource_filename(resource_path, 'noun.txt')),
                exc_dict=self._load_simple_rules(resource_filename(resource_path, 'inflection_lookup_noun.json')),
                token_tagset={'NNS', 'NNPS'}),
            EnglishMorphTag.JJ: SimpleNamespace(
                lemma_set=io.read_word_set(resource_filename(resource_path, 'adjective.txt')),
                exc_dict=self._load_simple_rules(resource_filename(resource_path, 'inflection_lookup_adjective.json')),
                token_tagset={'JJR', 'JJS'}),
            EnglishMorphTag.RB: SimpleNamespace(
                lemma_set=io.read_word_set(resource_filename(resource_path, 'adverb.txt')),
                exc_dict=self._load_simple_rules(resource_filename(resource_path, 'inflection_lookup_adverb.json')),
                token_tagset={'RBR', 'RBS'}),
        }

    def _load_inflection_lookup(self, resource_path: str) -> Dict[str, List[Tuple[str, str]]]:
        return self._load_simple_rules(resource_filename(resource_path, 'inflection_lookup.json'))

    def _load_inflection_rules(self, resource_path: str) -> Dict[str, List[AffixRule]]:
        return self._load_affix_rules(resource_filename(resource_path, 'inflection_rules.json'))

    @classmethod
    def _load_prefix(cls, resource_path: str) -> BytesTrie:
        prefixes = read_word_set(resource_filename(resource_path, 'prefix.txt'))
        prefixes = [e.split() for e in prefixes]
        return BytesTrie([(p[0], p[1].encode('utf-8')) for p in prefixes] + [(p[0] + '-', p[1].encode('utf-8')) for p in prefixes])

    def _load_prefix_lookup(self, resource_path: str) -> Dict[str, List[Tuple[str, str]]]:
        return self._load_simple_rules(resource_filename(resource_path, 'prefix_lookup.json'))

    def _load_derivation_lookup(self, resource_path: str) -> Dict[str, List[Tuple[str, str]]]:
        return self._load_simple_rules(resource_filename(resource_path, 'derivation_lookup.json'))

    def _load_derivation_rules(self, resource_path: str) -> Dict[str, List[AffixRule]]:
        return self._load_affix_rules(resource_filename(resource_path, 'derivation_rules.json'))

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

    def _save_prefix(self, filename: str):
        with open(filename, 'w') as fout:
            for prefix, tag in sorted(self._prefix.items()):
                if not prefix.endswith('-'): fout.write(prefix + ' ' + tag.decode('utf-8') + '\n')

    def analyze(self, token: str, pos: str, derivation=True, prefix=PREFIX_NONE) -> List[Tuple[str, str]]:
        token = token.lower()

        # inflection lookup
        morphs = self._analyze_inflection_lookup(token, pos)

        # base lookup
        if morphs is None:
            morphs = self._analyze_base_lookup(token, pos)

            # inflection rules
            if morphs is None:
                morphs = self._analyze_inflection_rules(token, pos)

                # default
                if morphs is None:
                    return [(token, EnglishMorphTag.to_lemma_tag(pos))]

        if derivation:
            morphs = self._analyze_derivation_rules(*morphs[0]) + morphs[1:]

        if prefix != self.PREFIX_NONE:
            morphs = self._analyze_prefix(*morphs[0]) + morphs[1:]

        return morphs

    def _analyze_base_lookup(self, token: str, pos: str) -> Optional[List[Tuple[str, str]]]:
        pos = self._base_lookup.get(pos, None)
        return [(token, pos)] if pos is not None else None

    def _analyze_inflection_lookup(self, token: str, pos: str) -> Optional[List[Tuple[str, str]]]:
        return self._inflection_lookup.get(token + ' ' + pos, None)

    def _analyze_inflection_rules(self, token: str, pos: str) -> Optional[List[Tuple[str, str]]]:
        lemma_tag = pos[:2]
        lex = self._inflection_lexicons.get(lemma_tag, None)
        if lex is None or pos not in lex.token_tagset: return None
        morphs = lex.exc_dict.get(token, None)
        if morphs is not None: return morphs

        for rule in self._inflection_rules[lemma_tag]:
            lemma = suffix_matcher(rule, lex.lemma_set, token, pos)
            if lemma is not None: return [(lemma, lemma_tag), ('+' + rule.affix, rule.affix_tag)]

        return None

    def _analyze_prefix(self, token: str, pos: str, option=PREFIX_SHORTEST) -> List[Tuple[str, str]]:
        def prefix_stem_pos(prefix: str) -> Optional[Tuple[str, str, str]]:
            stem = token[len(prefix):]
            if prefix == 'be':  # be+little
                for p in [EnglishMorphTag.VB, EnglishMorphTag.JJ, EnglishMorphTag.NN]:
                    if stem in self._inflection_lexicons[p].lemma_set: return prefix, stem, p

            return (prefix, stem, pos) if stem in self._inflection_lexicons[pos].lemma_set else None

        def get_key(psp):
            return len(psp[0])

        # exception
        t = self._prefix_lookup.get(token + ' ' + pos, None)
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

    def _analyze_derivation_rules(self, token, pos) -> List[Tuple[str, str]]:
        # no derivation
        t = self._derivation_lookup.get(token, None)
        if t is not None: return [(token, pos)]

        # exception
        t = self._derivation_lookup.get(token + ' ' + pos, None)
        if t is not None: return t

        for rule in self._derivation_rules[pos]:
            lemma = suffix_matcher(rule, self._inflection_lexicons[rule.lemma_tag].lemma_set, token)
            if lemma is not None:
                t = self._analyze_derivation_rules(lemma, rule.lemma_tag)
                return t + [('+' + rule.affix, rule.affix_tag)]

        if pos == EnglishMorphTag.JJ:
            t = self._analyze_inflection_rules(token, 'VBN')
            if t is not None: return t[:1] + [(t[1][0], EnglishMorphTag.J_ED)]

        return [(token, pos)]


class EnglishMorphTag:
    # lemma
    CC = 'CC'  # conjunction: CC
    CD = 'CD'  # number: CD
    DT = 'DT'  # determiner: DT, PDT
    EX = 'EX'  # existential: EX
    FW = 'FW'  # foreign words: FW
    GW = 'GW'  # goes with: AFX, GW
    IN = 'IN'  # case: IN, POS, RP, TO
    JJ = 'JJ'  # adjective: JJ, JJR, JJS
    LS = 'LS'  # list marker: LS
    MD = 'MD'  # modal: MD
    NN = 'NN'  # noun: NN, NNS, NNP, NNPS
    PR = 'PR'  # pronoun: PRP, PRP$, WP
    RB = 'RB'  # adverb: RB, RBR, RBS, WRB
    UH = 'UH'  # interjection: UH
    VB = 'VB'  # verb: VB, VBD, VBG, VBN, VBP, VBZ
    PU = 'PU'  # punctuation
    XX = 'XX'  # unknown: ADD, XX

    _PENN2 = {
        'AFX': GW,
        'CC': CC,
        'CD': CD,
        'DT': DT,
        'EX': EX,
        'FW': FW,
        'GW': GW,
        'IN': IN,
        'JJ': JJ,
        'JJR': JJ,
        'JJS': JJ,
        'LS': LS,
        'MD': MD,
        'NN': NN,
        'NNS': NN,
        'NNP': NN,
        'NNPS': NN,
        'PDT': DT,
        'POS': IN,
        'PRP': PR,
        'PRP$': PR,
        'RB': RB,
        'RBR': RB,
        'RBS': RB,
        'RP': IN,
        'TO': IN,
        'UH': UH,
        'VB': VB,
        'VBD': VB,
        'VBG': VB,
        'VBN': VB,
        'VBP': VB,
        'VBZ': VB,
        'WDT': DT,
        'WP': PR,
        'WP$': PR,
        'WRB': RB,
        '$': PU,
        ':': PU,
        ',': PU,
        '.': PU,
        '``': PU,
        "''": PU,
        '-LRB-': PU,
        '-RRB-': PU,
        'HYPH': PU,
        'NFP': PU,
        'SYM': PU,
        'PUNC': PU
    }

    @classmethod
    def to_lemma_tag(cls, pos):
        """
        :param pos: the Penn Treebank style part-of-speech tag.
        :return: the lemma tag converted from the original pos tag.
        """
        return cls._PENN2.get(pos, cls.XX)

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

    # prefix
    P = 'P'
