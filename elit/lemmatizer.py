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
import abc
import codecs
from elit.component import NLPComponent
from pkg_resources import resource_filename
from typing import Sequence

__author__ = "Liyan Xu"


class Lemmatizer(NLPComponent):
    def __init__(self):
        super(Lemmatizer, self).__init__()
        pass

    @abc.abstractmethod
    def decode(self, docs: Sequence[Sequence[str]], **kwargs):
        """
        :param docs: the list of pairs in the form of (form, pos)
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
    PATH_ABBREVIATION = "abbreviation.rule"

    def __init__(self):
        super(EnglishLemmatizer, self).__init__()

        self.abbreviation_rule = None

        self.init()

    def init(self):
        self.abbreviation_rule = self.__load_abbreviation_rule__(
            resource_filename(self.PATH_ROOT, self.PATH_ABBREVIATION))

    def decode(self, docs: Sequence[Sequence[str]], **kwargs):
        return [self.get_lemma(doc[0], doc[1]) for doc in docs]

    def get_lemma(self, form: str, pos: str) -> str:
        """
        :param form:
        :param pos:
        :return:
        """
        form = self.simplify_form(form)

        lemma = self.get_abbreviation(form, pos)
        lemma = lemma if lemma is not None else self.get_base_form_from_inflection(form, pos)
        lemma = lemma if lemma is not None else form

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
        :param lower:
        :param pos:
        :return: abbreviation form or None
        """
        return self.abbreviation_rule.get(self.__generate_abbreviation_key__(lower, pos), None)

    def get_base_form_from_inflection(self, lower: str, pos: str) -> str:
        """
        :param lower:
        :param pos:
        :return: base form or None
        """
        return None

    def is_cardinal(self, lower: str) -> bool:
        """
        :param lower:
        :return:
        """
        return False

    def is_ordinal(self, lower: str) -> bool:
        """
        :param lower:
        :return:
        """
        return False

    @classmethod
    def __load_abbreviation_rule__(cls, path: str) -> dict:
        """
        :param path:
        :return:
        """
        def key_value(line: str):
            tokens = line.strip().split()
            return cls.__generate_abbreviation_key__(tokens[0], tokens[1]), tokens[2]

        fin = codecs.open(path, mode='r', encoding='utf-8')
        return dict(key_value(line) for line in fin)

    @classmethod
    def __generate_abbreviation_key__(cls, form: str, pos: str) -> str:
        """
        :param form:
        :param pos:
        :return:
        """
        return form + "-" + pos


if __name__ == "__main__":
    lemmatizer = EnglishLemmatizer()

    docs = [("He", "PRP"), ("is", "VBZ"), ("tall", "JJ")]
    print(lemmatizer.decode(docs))

    docs = [("He", "PRP"), ("is", "VBZ"), ("n't", "RB"), ("tall", "JJ")]
    print(lemmatizer.decode(docs))