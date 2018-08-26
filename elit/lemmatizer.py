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
from elit.component import NLPComponent
from elit.util.structure import Document
from typing import Sequence

__author__ = "Liyan Xu"


class Lemmatizer(NLPComponent):
    def __init__(self):
        super(Lemmatizer, self).__init__()
        self.init()

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

    def __init__(self):
        super(EnglishLemmatizer, self).__init__()
        pass

    def init(self):
        pass

    def decode(self, docs: Sequence[Sequence[str]], **kwargs):
        return [self.get_lemma(doc[0], doc[1]) for doc in docs]

    @classmethod
    def get_lemma(cls, form: str, pos: str) -> str:
        """
        :param form:
        :param pos:
        :return:
        """
        form = cls.simplify_form(form)

        lemma = cls.get_abbreviation(form, pos)
        lemma = lemma if lemma is not None else cls.get_base_form_from_inflection(form, pos)
        lemma = lemma if lemma is not None else form

        if cls.is_cardinal(lemma):
            return cls.CONST_CARDINAL

        if cls.is_ordinal(lemma):
            return cls.CONST_ORDINAL

        return lemma

    @classmethod
    def simplify_form(cls, form: str) -> str:
        """
        TODO: figure out what transformation is needed
        :param form:
        :return:
        """
        return form.lower()

    @classmethod
    def get_abbreviation(cls, lower: str, pos: str) -> str:
        """
        :param lower:
        :param pos:
        :return: abbreviation or None
        """
        return None

    @classmethod
    def get_base_form_from_inflection(cls, lower: str, pos: str) -> str:
        """
        :param lower:
        :param pos:
        :return: base form or None
        """
        return None

    @classmethod
    def is_cardinal(cls, lower: str) -> bool:
        """
        :param lower:
        :return:
        """
        return False

    @classmethod
    def is_ordinal(cls, lower: str) -> bool:
        """
        :param lower:
        :return:
        """
        return False


if __name__ == "__main__":
    lemmatizer = EnglishLemmatizer()
    docs = [("He", "PRP"), ("is", "VBZ"), ("tall", "JJ")]
    print(lemmatizer.decode(docs))
