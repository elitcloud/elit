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

import fastText
import numpy as np
from gensim.models import KeyedVectors

from elit.util import TOK

__author__ = 'Jinho D. Choi'


# ======================================== Labels ========================================

class LabelMap:
    """
    LabelMap gives the mapping between class labels and their unique IDs.
    """
    def __init__(self):
        self.index_map = {}
        self.labels = []

    def __len__(self):
        return len(self.labels)

    def __str__(self):
        return str(self.index_map)

    def index(self, label):
        """
        :param label: a string label.
        :type label: str
        :return: the class ID of the label if exists; otherwise, -1.
        :rtype: int
        """
        return self.index_map.get(label, -1)

    def get(self, index):
        """
        :param index: the class ID of the label.
        :type index: int
        :return: the index'th class label.
        :rtype: str
        """
        return self.labels[index]

    def add(self, label):
        """
        Adds the class label to this map if not already exist.
        :param label: the class label.
        :type label: str
        :return: the class ID of the label.
        :rtype int
        """
        idx = self.index(label)
        if idx < 0:
            idx = len(self.labels)
            self.index_map[label] = idx
            self.labels.append(label)
        return idx


# ======================================== Vector Space Models ========================================

class VectorSpaceModel(abc.ABC):
    def __init__(self, model, dim):
        """
        :param model: a (pre-trained) vector space model (e.g., Word2Vec, FastText, GloVe).
        :type model: Union[KeyedVectors]
        :param dim: the dimension of each embedding.
        :type dim: int
        """
        self.model = model
        self.dim = dim

        # for zero padding
        self.zero = np.zeros(dim).astype('float32')

    @abc.abstractmethod
    def _get(self, key):
        """
        Auxiliary function for #get().
        :param key: a key.
        :type key: str
        :return: the embedding of the key.
        :rtype: numpy.array
        """
        return

    def get(self, key=None):
        """
        :param key: a key.
        :type key: Union[str, None]
        :return: the embedding of the key if not None; otherwise, the zero embedding.
        :rtype: numpy.array
        """
        return self.zero if key is None else self._get(key)

    def get_list(self, keys):
        """
        :param keys: a list of keys.
        :type keys: list of str
        :return: the list of embeddings for the corresponding keys.
        :rtype: list of numpy.array
        """
        return [self.get(key) for key in keys]


class FastText(VectorSpaceModel):
    def __init__(self, filepath):
        """
        :param filepath: the path to the binary file containing word embeddings.
        :type filepath: str
        """
        model = fastText.load_model(filepath)
        dim = model.get_dimension()
        super(FastText, self).__init__(model, dim)
        print('Init: %s (vocab = %d, dim = %d)' % (filepath, len(model.get_words()), dim))

    def _get(self, key):
        return self.model.get_word_vector(key)


class Word2Vec(VectorSpaceModel):
    def __init__(self, filepath):
        """
        :param filepath: the path to the binary file containing word embeddings.
        :type filepath: str
        """
        model = KeyedVectors.load(filepath) if filepath.lower().endswith('.gnsm') else KeyedVectors.load_word2vec_format(filepath, binary=True)
        dim = model.syn0.shape[1]
        super(Word2Vec, self).__init__(model, dim)
        print('Init: %s (vocab = %d, dim = %d)' % (filepath, len(model.vocab), dim))

    def _get(self, key):
        vocab = self.model.vocab.get(key, None)
        return self.model.syn0[vocab.index] if vocab else self.zero


class GloVe(VectorSpaceModel):
    def __init__(self, filepath):
        """
        :param filepath: the path to the binary file containing word embeddings.
        :type filepath: str
        """
        # TODO: to be completed
        super(GloVe, self).__init__(None, 0)
        pass

    def _get(self, key):
        # TODO: to be completed
        return


# class Word2VecTmp:
#     def __init__(self, filepath):
#         """
#         :param filepath: the path to the file containing word embeddings.
#         """
#         self.model = None
#
#         if filepath.endswith('.gnsm'):
#             self.model = KeyedVectors.load(filepath)
#         elif filepath.endswith('.bin'):
#             self.model = KeyedVectors.load_word2vec_format(filepath, binary=True)
#         else:
#             raise ValueError('Unknown type: ' + filepath)
#
#         self.dim = self.model.syn0.shape[1]
#         self.pad = np.zeros((self.dim,)).astype('float32')
#         print('Init: %s (vocab = %d, dim = %d)' % (filepath, len(self.model.vocab), self.dim))
#
#     def doc_to_emb(self, document, maxlen):
#         """
#         :param document: the document comprising tokens to retrieve word embeddings for.
#         :param maxlen: the maximum length of the document (# of tokens).
#         :return: the list of word embeddings corresponding to the tokens in the document.
#         """
#         def emb(token_index):
#             if token_index >= len(document): return self.pad
#             vocab = self.model.vocab.get(document[token_index], None)
#             return self.model.syn0[vocab.index] if vocab else self.pad
#
#         return np.array([emb(i) for i in range(maxlen)])
#         # TODO: the following 3 lines should be replaced by the above return statement
#         # l = [self.model.syn0[0] for _ in range(maxlen-len(document))]
#         # l.extend([emb(i) for i in range(min(maxlen, len(document)))])
#         # return np.array(l)
#
#     def docs_to_emb(self, documents, maxlen):
#         """
#         :param documents: a list of documents.
#         :param maxlen:
#         :return:
#         """
#         return np.array([self.doc_to_emb(document, maxlen) for document in documents])


# ======================================== Embeddings ========================================

X_FST = np.array([1, 0]).astype('float32')  # the first word
X_LST = np.array([0, 1]).astype('float32')  # the last word
X_ANY = np.array([0, 0]).astype('float32')  # any other word


def get_loc_embeddings(document):
    """
    :param document: a document.
    :type document: elit.util.Document
    :return: (the position embeddings of all tokens in the document, default embedding).
    :rtype: tuple of (list of numpy.array, numpy.array)
    """
    def aux(sentence):
        size = len(sentence)
        return [X_FST if i == 0 else X_LST if i + 1 == size else X_ANY for i in range(size)]

    return [aux(s) for s in document.sentences], X_ANY


def get_vsm_embeddings(vsm, document, key=TOK):
    """
    :param vsm: a vector space model.
    :type vsm: VectorSpaceModel
    :param document: a document.
    :type document: elit.util.Document
    :param key: the key to the fields to be retrieved in each sentence.
    :type key: str
    :return: (the list of embeddings, zero embedding).
    :rtype: tuple of (list of numpy.array, numpy.array)
    """
    return [vsm.get_list(s[key]) for s in document.sentences], vsm.zero


def x_extract(tok_id, window, size, emb, zero):
    """
    :param tok_id:
    :param window:
    :param size:
    :param emb:
    :param zero:
    :return:
    """
    i = tok_id + window
    return emb[i] if 0 <= i < size else zero
