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
import abc
import codecs
from typing import Sequence
from xml.etree import ElementTree

from elit.lemmatization.english.suffix_group import SuffixGroup
from elit.lemmatization.english.suffix_rule import SuffixRule
from pkg_resources import resource_filename

from elit.component import NLPComponent
from elit.nlp.lemmatization.english.inflection import Inflection

__author__ = "Liyan Xu"


class Lemmatizer(NLPComponent):
    def __init__(self):
        super(Lemmatizer, self).__init__()
        pass

    @abc.abstractmethod
    def decode(self, docs: Sequence[Sequence[str]], **kwargs):
        """
        :param docs: the list of pairs in the form of (form, pos). e.g. [("He", "PRP"), ("is", "VBZ"), ("tall", "JJ")]
        :param kwargs:
        :return:
        """
        pass

    def init(self):
        pass

    def load(self, model_path: str, **kwargs):
        """
        :param model_path:
        :param kwargs:
        :return:
        """
        pass

    def save(self, model_path: str, **kwargs):
        """
        :param model_path:
        :param kwargs:
        :return:
        """
        pass

    def train(self, trn_docs: Sequence[Sequence[str]], dev_docs: Sequence[Sequence[str]], model_path: str, **kwargs):
        """
        :param trn_docs:
        :param dev_docs:
        :param model_path:
        :param kwargs:
        :return:
        """
        pass

    def evaluate(self, docs: Sequence[Sequence[str]], **kwargs):
        """
        :param docs:
        :param kwargs:
        :return:
        """
        pass


class EnglishLemmatizer(Lemmatizer):

    CONST_CARDINAL = "#crd#"
    CONST_ORDINAL = "#ord#"

    PATH_ROOT = "elit.resources.lemmatizer.english"
    FILENAME_ABBREVIATION = "abbreviation.rule"
    FILENAME_CARDINAL = "cardinal.txt"
    FILENAME_ORDINAL = "ordinal.txt"
    FILENAME_INFLECTION = "inflection_suffix.xml"

    BASE_POS_VERB = "VB"
    BASE_POS_NOUN = "NN"
    BASE_POS_ADJECTIVE = "JJ"
    BASE_POS_ADVERB = "RB"
    VERB = "verb"
    NOUN = "noun"
    ADJECTIVE = "adjective"
    ADVERB = "adverb"

    def __init__(self):
        super(EnglishLemmatizer, self).__init__()

        self.rule_abbreviation = None
        self.base_cardinal = None
        self.base_ordinal = None
        self.inf_by_base_pos = None

        self.init()

    def init(self):
        self.rule_abbreviation = self.__load_abbreviation_rule__(
            resource_filename(self.PATH_ROOT, self.FILENAME_ABBREVIATION))
        self.base_cardinal = self.read_word_set(
            resource_filename(self.PATH_ROOT, self.FILENAME_CARDINAL))
        self.base_ordinal = self.read_word_set(
            resource_filename(self.PATH_ROOT, self.FILENAME_ORDINAL))
        self.inf_by_base_pos = self.__load_inflections_from_xml__(
            resource_filename(self.PATH_ROOT, self.FILENAME_INFLECTION))

    def decode(self, docs: Sequence[Sequence[str]], **kwargs):
        return [self.get_lemma(doc[0], doc[1]) for doc in docs]

    def get_lemma(self, form: str, pos: str) -> str:
        """
        :param form:
        :param pos:
        :return:
        """
        form = self.simplify_form(form)

        # Get base form from abbreviation and inflection
        lemma = self.get_abbreviation(form, pos)
        lemma = lemma if lemma is not None else self.get_base_form_from_inflection(form, pos)
        lemma = lemma if lemma is not None else form

        # Mark cardinal or ordinal if applicable
        if self.is_cardinal(lemma):
            return self.CONST_CARDINAL
        if self.is_ordinal(lemma):
            return self.CONST_ORDINAL

        return lemma

    @classmethod
    def simplify_form(cls, form: str) -> str:
        """
        TODO: figure out what transformation is needed
        :param form:
        :return:
        """
        return form.lower()

    def get_abbreviation(self, lower: str, pos: str) -> str:
        """
        Get the base form from abbreviation.
        :param lower:
        :param pos:
        :return: abbreviation form or None
        """
        return self.rule_abbreviation.get(self.__generate_abbreviation_key__(lower, pos), None)

    def get_base_form_from_inflection(self, lower: str, pos: str) -> str:
        """
        Get the base form from corresponding inflection.
        :param lower:
        :param pos:
        :return: base form or None
        """
        inflection = self.inf_by_base_pos.get(pos[:2], None)
        return None if inflection is None else inflection.get_base_form(lower, pos)

    def is_cardinal(self, lower: str) -> bool:
        """
        :param lower:
        :return:
        """
        return lower in self.base_cardinal

    def is_ordinal(self, lower: str) -> bool:
        """
        :param lower:
        :return:
        """
        return lower in self.base_ordinal

    @classmethod
    def __load_abbreviation_rule__(cls, file_path: str) -> dict:
        """
        :param file_path:
        :return:
        """
        def key_value(line: str):
            tokens = line.strip().split()
            return cls.__generate_abbreviation_key__(tokens[0], tokens[1]), tokens[2]

        fin = codecs.open(file_path, mode='r', encoding='utf-8')
        d = dict(key_value(line) for line in fin)
        print('Init: %s(keys=%d)' % (file_path, len(d)))
        return d

    @classmethod
    def __generate_abbreviation_key__(cls, form: str, pos: str) -> str:
        """
        :param form:
        :param pos:
        :return:
        """
        return form + "-" + pos

    @classmethod
    def read_word_set(cls, file_path: str) -> set:
        """
        TODO: may refactor in standalone util
        :param file_path:
        :return:
        """
        fin = codecs.open(file_path, mode='r', encoding='utf-8')
        s = set(line.strip() for line in fin)
        print('Init: %s(keys=%d)' % (file_path, len(s)))
        return s

    @classmethod
    def read_word_dict(cls, file_path: str) -> dict:
        """
        TODO: may refactor in standalone util
        :param file_path:
        :return:
        """
        fin = codecs.open(file_path, mode='r', encoding='utf-8')
        d = dict(line.strip().split() for line in fin)
        print('Init: %s(keys=%d)' % (file_path, len(d)))
        return d

    @classmethod
    def __load_inflections_from_xml__(cls, file_path: str) -> dict:
        """
        :param file_path:
        :return:
        """
        d = dict()
        root = ElementTree.parse(file_path).getroot()
        d[cls.BASE_POS_VERB] = cls.__build_inflection_by_pos__(cls.BASE_POS_VERB, cls.VERB, root)
        d[cls.BASE_POS_NOUN] = cls.__build_inflection_by_pos__(cls.BASE_POS_NOUN, cls.NOUN, root)
        d[cls.BASE_POS_ADJECTIVE] = cls.__build_inflection_by_pos__(cls.BASE_POS_ADJECTIVE, cls.ADJECTIVE, root)
        d[cls.BASE_POS_ADVERB] = cls.__build_inflection_by_pos__(cls.BASE_POS_ADVERB, cls.ADVERB, root)
        return d

    @classmethod
    def __build_inflection_by_pos__(cls, base_pos: str, type: str, root) -> Inflection:
        """
        Initialize inflection for a certain word type/basePOS.
        :param base_pos:
        :param type:
        :param root:
        :return:
        """
        set_base = cls.read_word_set(resource_filename(cls.PATH_ROOT, type + ".base"))
        dict_exc = cls.read_word_dict(resource_filename(cls.PATH_ROOT, type + ".exc"))

        inflection = Inflection(base_pos, set_base, dict_exc, list())

        affixes = root.find(type).findall("affix")
        for affix in affixes:
            suffix_group = SuffixGroup(affix.get("form"), affix.get("org_pos"), list())
            inflection.suffix_groups.append(suffix_group)
            rules = affix.findall("rule")
            for rule in rules:
                suffix_rule = SuffixRule(
                    rule.get("affix"),
                    [r.strip() for r in rule.get("token_affixes").split(",")],
                    cls.str_to_bool(rule.get("doubleConsonants")),
                    set_base
                )
                suffix_group.rules.append(suffix_rule)

        return inflection

    @classmethod
    def str_to_bool(cls, s: str) -> bool:
        """
        TODO: may refactor in standalone util
        :param s:
        :return:
        """
        return s is not None and s.lower() == "true"


'''if __name__ == "__main__":
    lemmatizer = EnglishLemmatizer()

    docs = [("He", "PRP"), ("is", "VBZ"), ("tall", "JJ")]
    print(lemmatizer.decode(docs))

    docs = [("He", "PRP"), ("is", "VBZ"), ("n't", "RB"), ("tall", "JJ")]
    print(lemmatizer.decode(docs))

    docs = [("He", "PRP"), ("has", "VBZ"), ("one", "CD"), ("paper", "NN")]
    print(lemmatizer.decode(docs))

    docs = [("He", "PRP"), ("is", "VBZ"), ("the", "DT"), ("first", "JJ"), ("winner", "NN")]
    print(lemmatizer.decode(docs))

    docs = [("He", "PRP"), ("is", "VBZ"), ("lying", "VBG")]
    print(lemmatizer.decode(docs))

    docs = [("He", "PRP"), ("is", "VBZ"), ("running", "VBG")]
    print(lemmatizer.decode(docs))

    docs = [("He", "PRP"), ("is", "VBZ"), ("feeling", "VBG"), ("cold", "JJ")]
    print(lemmatizer.decode(docs))

    docs = [("They", "PRP"), ("are", "VBP"), ("gentlemen", "NNS")]
    print(lemmatizer.decode(docs))

    docs = [("He", "PRP"), ("bought", "VBD"), ("a", "DT"), ("car", "NN")]
    print(lemmatizer.decode(docs))'''
