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
__author__ = 'Jinho D. Choi'

DOC_ID = 'doc_id'   # document ID
SEN = 'sen'       # sentences
SEN_ID = 'sen_id'   # sentence ID
TOK = 'tok'
OFF = 'off'
LEM = 'lem'
POS = 'pos'
NER = 'ner'
DEP = 'dep'
COREF = 'coref'
SENTI = 'senti'


class Document(dict):
    def __init__(self, d=None, **kwargs):
        """
        :param d: a dictionary, if not None, all of whose fields are added to this document.
        :type d: dict
        :param kwargs: additional fields to be added; if keys already exist; the values are overwritten with these.
        """
        super().__init__()
        self._iter = -1

        if d is not None: self.update(d)
        self.update(kwargs)
        self._sentences = self.setdefault(SEN, [])

    def __len__(self):
        """
        :return: the number of sentences in the document.
        :rtype: int
        """
        return len(self.sentences)

    def __iter__(self):
        self._iter = -1
        return self

    def __next__(self):
        self._iter += 1
        if self._iter >= len(self.sentences):
            raise StopIteration
        return self._sentences[self._iter]

    @property
    def sentences(self):
        """
        :return: the list of sentences in the document.
        :rtype: list of Sentence
        """
        return self._sentences

    def add_sentence(self, sentence):
        """
        :param sentence: a sentence to be added.
        :type sentence: Sentence
        """
        self.sentences.append(sentence)

    def get_sentence(self, index):
        """
        :return: the index'th sentence.
        :rtype: Sentence
        """
        return self.sentences[index]


class Sentence(dict):
    def __init__(self, d=None, **kwargs):
        """
        :param d: a dictionary, if not None, all of whose fields are added to this sentence.
        :type d: dict
        :param kwargs: additional fields to be added; if keys already exist; the values are overwritten with these.
        """
        super().__init__()
        self._iter = -1

        if d is not None: self.update(d)
        self.update(kwargs)
        self._tokens = self.setdefault(TOK, [])

    def __len__(self):
        """
        :return: the number of tokens in the sentence.
        """
        return len(self.tokens)

    def __iter__(self):
        self._iter = -1
        return self

    def __next__(self):
        self._iter += 1
        if self._iter >= len(self.tokens):
            raise StopIteration
        return self._tokens[self._iter]

    @property
    def tokens(self):
        """
        :return: the list of tokens in the sentence.
        :rtype: list of str
        """
        return self._tokens

    @property
    def part_of_speech_tags(self):
        """
        :return: the list of part-of-speech tags corresponding to the tokens in this sentence.
        :rtype: list of str
        """
        return self[POS]