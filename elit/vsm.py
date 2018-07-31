# ========================================================================
# Copyright 2017 Emory University
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
from typing import List, Tuple, Any, Optional

import fastText
import numpy as np
from gensim.models import KeyedVectors

from elit.structure import TOK, Document

__author__ = 'Jinho D. Choi'


# ======================================== Labels ========================================

class LabelMap:
    """
    LabelMap provides the mapping between class labels and their unique IDs.
    """

    def __init__(self):
        self.index_map = {}
        self.labels = []

    def __len__(self):
        return len(self.labels)

    def __str__(self):
        return str(self.index_map)

    def index(self, label: str) -> int:
        """
        :param label: a string label.
        :return: the class ID of the label if exists; otherwise, -1.
        """
        return self.index_map.get(label, -1)

    def get(self, index: int) -> str:
        """
        :param index: the class ID of the label.
        :return: the index'th class label.
        """
        return self.labels[index]

    def add(self, label: str) -> int:
        """
        Adds the class label to this map if not already exist.
        :param label: the class label.
        :return: the class ID of the label.
        """
        idx = self.index(label)
        if idx < 0:
            idx = len(self.labels)
            self.index_map[label] = idx
            self.labels.append(label)
        return idx

    def argmax(self, scores: np.ndarray) -> str:
        """
        :param scores: scores of all labels.
        :return: the label with the maximum score.
        """
        if self.__len__() < len(scores): scores = scores[:self.__len__()]
        return self.get(np.asscalar(np.argmax(scores)))


# ======================================== Vector Space Models ========================================

class VectorSpaceModel(abc.ABC):
    """
    VectorSpaceModel provides an abstract class to wrap token-based vector space models.
    """

    def __init__(self, model: Any, dim: int):
        """
        :param model: a (pre-trained) vector space model (e.g., Word2Vec, FastText, GloVe).
        :param dim: the dimension of each embedding.
        """
        self.model = model
        self.dim = dim

        # for zero padding
        self.pad = np.zeros(dim).astype('float32')

    @abc.abstractmethod
    def _emb(self, key: str) -> np.ndarray:
        """
        Auxiliary function for #embedding().
        :param key: a key (e.g., word, part-of-speech tag).
        :return: the embedding of the key.
        """
        pass

    def embedding(self, key: Optional[str] = None) -> np.ndarray:
        """
        :param key: a key (e.g., word, part-of-speech tag).
        :return: the embedding of the key if not None; otherwise, the zero embedding (self.pad).
        """
        return self.pad if key is None else self._emb(key)

    def embedding_list(self, keys: List[str]) -> List[np.ndarray]:
        """
        :param keys: a list of keys.
        :return: the list of embeddings for the corresponding keys.
        """
        return [self.embedding(key) for key in keys]

    def embedding_matrix(self, keys: List[str], maxlen: int) -> List[np.ndarray]:
        """
        :param keys: a list of keys.
        :param maxlen: the maximum length of the output list;
                       if this is greater than the number of input keys, the bottom part of the matrix is padded with zero embeddings.
                       if this is smaller than the number of input keys, the exceeding keys are not included in the embedding matrix.
        :return: a matrix where each row is the embedding of the corresponding key.
        """
        size = len(keys)
        return [self.embedding(keys[i]) if i < size else self.pad for i in range(maxlen)]

    def sentence_embedding_list(self, document: Document, key: str = TOK) -> Tuple[List[List[np.ndarray]], np.ndarray]:
        """
        :param document: a document.
        :param key: the key to the fields to be retrieved in each sentence.
        :return: a tuple of (the list of embeddings, zero-pad embedding).
        """
        return [self.embedding_list(s[key]) for s in document], self.pad


class FastText(VectorSpaceModel):
    def __init__(self, filepath: str):
        """
        :param filepath: the path to the binary file containing word embeddings.
        :type filepath: str
        """
        model = fastText.load_model(filepath)
        dim = model.get_dimension()
        super().__init__(model, dim)
        print('FastText: %s (vocab = %d, dim = %d)' % (filepath, len(model.get_words()), dim))

    def _emb(self, key: str) -> np.ndarray:
        return self.model.get_word_vector(key)


class Word2Vec(VectorSpaceModel):
    def __init__(self, filepath: str):
        """
        :param filepath: the path to the binary file containing word embeddings.
        :type filepath: str
        """
        model = KeyedVectors.load(filepath) if filepath.lower().endswith('.gnsm') else KeyedVectors.load_word2vec_format(filepath, binary=True)
        dim = model.syn0.shape[1]
        super().__init__(model, dim)
        print('Word2Vec: %s (vocab = %d, dim = %d)' % (filepath, len(model.vocab), dim))

    def _emb(self, key: str) -> np.ndarray:
        vocab = self.model.vocab.get(key, None)
        return self.model.syn0[vocab.index] if vocab else self.pad


class GloVe(VectorSpaceModel):
    def __init__(self, filepath: str):
        """
        :param filepath: the path to the binary file containing word embeddings.
        :type filepath: str
        """
        # TODO: to be completed
        model = None
        dim = 0
        super().__init__(model, dim)
        print('GloVe: %s (vocab = %d, dim = %d)' % (filepath, len(model.vocab), dim))

    def _emb(self, key: str) -> np.ndarray:
        # TODO: to be completed
        return self.pad


# ======================================== Embeddings ========================================

X_FST = np.array([1, 0]).astype('float32')  # the first word
X_LST = np.array([0, 1]).astype('float32')  # the last word
X_ANY = np.array([0, 0]).astype('float32')  # any other word


def get_loc_embeddings(document: Document) -> Tuple[List[List[np.ndarray]], np.ndarray]:
    """
    :param document: a document.
    :return: a tuple of (the position embeddings of all tokens in the document, default embedding).
    """

    def aux(sentence):
        size = len(sentence)
        return [X_FST if i == 0 else X_LST if i + 1 == size else X_ANY for i in range(size)]

    return [aux(s) for s in document], X_ANY


def x_extract(emb_id: int, offset: int, emb: List[np.ndarray], pad: np.ndarray) -> np.ndarray:
    """
    :param emb_id: the embedding ID.
    :param offset: an offset.
    :param emb: a list of embeddings.
    :param pad: the zero-pad embedding.
    :return: the (emb_id + offset)'th embedding if available; otherwise, the zero-pad embedding.
    """
    i = emb_id + offset
    return emb[i] if 0 <= i < len(emb) else pad
