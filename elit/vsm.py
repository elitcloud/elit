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
import inspect
from typing import List, Tuple, Any, Optional, Union, Sequence

import fastText
import numpy as np
from gensim.models import KeyedVectors
from mxnet import nd
from mxnet.ndarray import NDArray

from elit.structure import TOK, Document

__author__ = 'Jinho D. Choi'


class LabelMap:
    """
    :class:`LabelMap` provides the mapping between string labels and their unique integer IDs.
    """

    def __init__(self):
        self.index_map = {}
        self.labels = []

    def __len__(self):
        return len(self.labels)

    def __str__(self):
        return str(self.index_map)

    def add(self, label: str) -> int:
        """
        :param label: the label.
        :return: the class ID of the label.

        Adds the label to this map and assigns its class ID if not already exists.
        """
        idx = self.cid(label)
        if idx < 0:
            idx = len(self.labels)
            self.index_map[label] = idx
            self.labels.append(label)
        return idx

    def get(self, cid: int) -> str:
        """
        :param cid: the class ID.
        :return: the label corresponding to the class ID.
        """
        return self.labels[cid]

    def cid(self, label: str) -> int:
        """
        :param label: the label.
        :return: the class ID of the label if exists; otherwise, -1.
        """
        return self.index_map.get(label, -1)

    def argmax(self, scores: Union[np.ndarray, NDArray]) -> str:
        """
        :param scores: the prediction scores of all labels.
        :return: the label with the maximum score.
        """
        if self.__len__() < len(scores): scores = scores[:self.__len__()]
        n = nd.argmax(scores, axis=0).asscalar() if isinstance(scores, NDArray) else np.argmax(scores)
        return self.get(int(n))


# ======================================== Vector Space Models ========================================

class VectorSpaceModel(abc.ABC):
    """
    :class:`VectorSpaceModel` is an abstract class to implement token-based vector space models.

    Abstract methods to be implemented:
      - :meth:`VectorSpaceModel.emb`
    """

    def __init__(self, model: Any, dim: int):
        """
        :param model: the vector space model (e.g., Word2Vec, FastText, GloVe).
        :param dim: the dimension of each embedding.
        """
        self.model = model
        self.dim = dim

        # for zero padding
        self.pad = np.zeros(dim).astype('float32')

    @abc.abstractmethod
    def emb(self, value: str) -> np.ndarray:
        """
        :param value: the value (e.g., a word) to retrieve the embedding for.
        :return: the embedding of the value.

        This is an auxiliary function for :meth:`VectorSpaceModel.embedding`.
        """
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))

    def embedding(self, value: Optional[str] = None) -> np.ndarray:
        """
        :param value: the value (e.g., a word) to retrieve the embedding for.
        :return: the embedding of the value; if the value is ``None``, the zero embedding.
        """
        return self.emb(value) if value is not None else self.pad

    def embedding_list(self, values: Sequence[str]) -> List[np.ndarray]:
        """
        :param values: the sequence of values (e.g., words).
        :return: the list of embeddings for the corresponding values.
        """
        return [self.embedding(value) for value in values]

    def embedding_list_of_sentences(self, document: Document, key: str = TOK) -> List[List[np.ndarray]]:
        """
        :param document: the input document.
        :param key: the key to the values in each sentence to retrieve the embeddings for.
        :return: the list of embedding lists, where each embedding list contains embeddings for the corresponding sentence.
        """
        return [self.embedding_list(s[key]) for s in document]

    def embedding_matrix(self, values: List[str], maxlen: int) -> List[np.ndarray]:
        """
        :param values: the sequence of values (e.g., words).
        :param maxlen: the maximum length of the output list;
                       if ``> len(values)``, the bottom part of the matrix is padded with zero embeddings;
                       if ``< len(values)``, embeddings of the exceeding values are discarded from the resulting matrix.
        :return: the matrix where each row is the embedding of the corresponding value.
        """
        return [self.embedding(values[i]) if i < len(values) else self.pad for i in range(maxlen)]


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

    def emb(self, value: str) -> np.ndarray:
        return self.model.get_word_vector(value)


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

    def emb(self, value: str) -> np.ndarray:
        vocab = self.model.vocab.get(value, None)
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

    def emb(self, value: str) -> np.ndarray:
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
