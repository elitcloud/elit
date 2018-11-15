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
import glob
import inspect
import logging
import marisa_trie
import pickle
from typing import List, Optional, Sequence


import abc

import fastText
import numpy as np
from gensim.models import KeyedVectors
from types import SimpleNamespace

from elit.structure import TOK, Document

__author__ = 'Jinho D. Choi'


# ======================================== Vector Space Models ========================================

class VectorSpaceModel(abc.ABC):
    """
    :class:`VectorSpaceModel` is an abstract class to implement vector space models.

    Abstract methods to be implemented:
      - :meth:`VectorSpaceModel.emb`
    """

    def __init__(self, dim: int):
        """
        :param dim: the dimension of each embedding.
        """
        self.dim = dim

        # for zero padding
        self.pad = np.zeros(dim).astype('float32')
        self.vs_map = {}

    @abc.abstractmethod
    def emb(self, value: str) -> np.ndarray:
        """
        :param value: the value (e.g., a word) to retrieve the embedding for.
        :return: the embedding of the value.

        This is an auxiliary function for :meth:`VectorSpaceModel.embedding`.
        """
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))

    def lookup(self, value: Optional[str] = None):
        self.vs_map[value] = self.vs_map.get(value, self.emb(value))
        return self.vs_map[value]

    def embedding(self, value: Optional[str] = None) -> np.ndarray:
        """
        :param value: the value (e.g., a word) to retrieve the embedding for.
        :return: the embedding of the value; if the value is ``None``, the zero embedding.
        """
        return self.lookup(value) if value is not None else self.pad

    def embedding_list(self, values: Sequence[str]) -> List[np.ndarray]:
        """
        :param values: the sequence of values (e.g., words).
        :return: the list of embeddings for the corresponding values.
        """
        return [self.embedding(value) for value in values]

    def sentence_embedding_list(self, document: Document, key: str = TOK) -> List[List[np.ndarray]]:
        """
        :param document: the input document.
        :param key: the key to the values in each sentence to retrieve the embeddings for.
        :return: the list of embedding lists, where each embedding list contains the embeddings for
        the values in the corresponding sentence.
        """
        return [self.embedding_list(s[key]) for s in document]

    def embedding_matrix(self, values: Sequence[str], maxlen: int) -> List[np.ndarray]:
        """
        :param values: the sequence of values (e.g., words).
        :param maxlen: the maximum length of the output list;
                       if ``> len(values)``, the bottom part of the matrix is padded with zero
                       embeddings;
                       if ``< len(values)``, embeddings of the exceeding values are discarded from
                       the resulting matrix.
        :return: the matrix where each row is the embedding of the corresponding value.
        """
        return [self.embedding(values[i]) if i < len(values) else self.pad for i in range(maxlen)]


class FastText(VectorSpaceModel):
    """
    :class:`FastText` is a vector space model whose embeddings are trained by `FastText <https://github.com/facebookresearch/fastText>`_.
    """

    def __init__(self, filepath: str):
        """
        :param filepath: the path to the binary file containing a word embedding model trained by FastText.
        :type filepath: str
        """
        logging.info('FastText')
        logging.info('- model: {}'.format(filepath))
        self.model = fastText.load_model(filepath)
        dim = self.model.get_dimension()
        super().__init__(dim)
        logging.info('- vocab = %d, dim = %d' % (len(self.model.get_words()), dim))

    def emb(self, value: str) -> np.ndarray:
        return self.model.get_word_vector(value)


class Word2Vec(VectorSpaceModel):
    """
    :class:`Word2Vec` is a vector space model whose embeddings are trained by `Word2Vec <https://github.com/tmikolov/word2vec>`_.
    """

    def __init__(self, filepath: str):
        """
        :param filepath: the path to the binary file containing a word embedding model trained by Word2Ved.
        :type filepath: str
        """
        logging.info('Word2Vec')
        logging.info('- model: {}'.format(filepath))
        self.model = KeyedVectors.load(filepath) if filepath.lower().endswith(
            '.gnsm') else KeyedVectors.load_word2vec_format(filepath, binary=True)
        dim = self.model.syn0.shape[1]
        super().__init__(dim)
        logging.info('- vocab = %d, dim = %d' % (len(self.model.vocab), dim))

    def emb(self, value: str) -> np.ndarray:
        vocab = self.model.vocab.get(value, None)
        return self.model.syn0[vocab.index] if vocab else self.pad


class Gaze(VectorSpaceModel):

    def __init__(self, filepath: str):
        with open(filepath, 'rb') as fin:
            self.model = pickle.load(fin)
            self.dim = pickle.load(fin)
            self.num_labels = pickle.load(fin)
            self.option = pickle.load(fin)
        super().__init__(dim=self.dim)
        logging.info('GazeVec')
        logging.info('- model: {}'.format(filepath))

    def emb(self, value: str) -> np.ndarray:
        pass

    def embedding_list(self, values: Sequence[str]):

        entity_vectors = [np.zeros(self.dim).astype('float32') for _ in values]

        def normalize(v):
            norm = sum(v)
            if norm < 0.0001:
                return v
            return v / norm

        for i, word in enumerate(values):
            entity = ''
            components = []
            for j in range(i, len(values)):
                if j > i:
                    entity += ' '
                entity += values[j]
                components.append(j)
                if entity not in self.model:
                    if not self.model.keys(entity):  # subtrie is empty
                        break
                    else:
                        continue
                for label in self.model[entity]:
                    for ind in components:
                        if self.option == 1:
                            entity_vectors[ind][label] += (j + 1 - i) * (j + 1 - i)
                        else:
                            if ind == i:  # first
                                entity_vectors[ind][self.option * label[0]] += (j + 1 - i) * (j + 1 - i)
                            elif ind == j:  # last
                                entity_vectors[ind][self.option * label[0] + 1] += (j + 1 - i) * (j + 1 - i)
                            if self.option > 2 and i == j:  # unique
                                entity_vectors[ind][self.option * label[0] + 2] += 1
                            if self.option > 3 and ind != i and ind != j:  # middle
                                entity_vectors[ind][self.option * label[0] + 3] += (j + 1 - i) * (j + 1 - i)
                            if self.option > 4:
                                entity_vectors[ind][self.option * label[0] + 4] += (j + 1 - i) * (j + 1 - i)
        entity_vectors = [normalize(vector) for vector in entity_vectors]
        return entity_vectors

    @staticmethod
    def create(source_dir: str, model_path: str, option=1):
        files = glob.glob('{}/*.txt'.format(source_dir))
        num_labels = len(files)
        dim = num_labels * option
        keys, values = [], []
        for i, file in enumerate(sorted(files)):
            with open(file, 'r', encoding='latin1') as f:
                l = ['' + u' '.join(line.split()) for line in f]
            logging.info('Init: {} - {}'.format(file, len(l)))
            keys.extend(l)
            values.extend([[i]] * len(l))
        model = marisa_trie.RecordTrie("h", zip(keys, values))
        logging.info('Init: {} (NE-vocab = {}, NE-labels = {}, option = {})'
                     .format(source_dir, len(model.keys()), dim, option))
        logging.info('Saving {}'.format(model_path))
        with open(model_path, 'wb') as fout:
            pickle.dump(model, fout)
            pickle.dump(dim, fout)
            pickle.dump(num_labels, fout)
            pickle.dump(option, fout)


# class GloVe(VectorSpaceModel):
#     def __init__(self, filepath: str):
#         """
#         :param filepath: the path to the binary file containing word embeddings.
#         :type filepath: str
#         """
#         super().__init__(0)
#         raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))
#
#     def emb(self, value: str) -> np.ndarray:
#         return self.pad


class Position2Vec(VectorSpaceModel):
    """
    :class:`Position2Vec` is a vector space model that generates position embeddings
    indicating the positions of tokens in a sentence (e.g., first token, last token).

    Supported method:
      - :meth:`Position2Vec.sentence_embedding_list`
    """
    _FST = np.array([1, 0]).astype('float32')  # the first word
    _LST = np.array([0, 1]).astype('float32')  # the last word
    _ANY = np.array([0, 0]).astype('float32')  # any other word

    def __init__(self):
        super().__init__(2)

    # override
    def emb(self, value: str) -> np.ndarray:
        """
        Unsupported (any token-based embeddings are not supported in this vector space model).
        """
        raise AttributeError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))

    # override
    def sentence_embedding_list(self, document: Document, key: str = None) -> List[List[np.ndarray]]:
        """
        :param document: the input document.
        :param key: ``None``.
        :return: the list of embedding lists, where each embedding list contains the position embeddings of the corresponding sentence.
        """

        def aux(sentence):
            size = len(sentence)
            return [self._FST if i == 0 else self._LST if i + 1 == size else self._ANY for i in range(size)]

        return [aux(s) for s in document]


# ======================================== Embeddings ========================================


def init_vsm(l: list) -> SimpleNamespace:
    model, path = l
    if model.lower() == 'word2vec':
        model = Word2Vec
    elif model.lower() == 'fasttext':
        model = FastText
    elif model.lower() == 'gaze':
        model = Gaze
    else:
        raise TypeError('model {} is not supported'.format(model))
    return SimpleNamespace(model=model(path))
