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
from abc import ABCMeta, abstractmethod

import numpy as np
from gensim.models.keyedvectors import KeyedVectors

from elit.structure import NLPGraph

__author__ = 'Jinho D. Choi'


class NLPLexicon(metaclass=ABCMeta):
    def __init__(self, word_embeddings: KeyedVectors):
        """
        :param word_embeddings: KeyedVectors.load_word2vec_format('word_vectors.bin').
        """
        self.word_embeddings: KeyedVectors = self._init_vectors(word_embeddings)

    @classmethod
    def _init_vectors(cls, vectors: KeyedVectors, root_lower: float = -.25, root_upper: float = .25) -> KeyedVectors:
        """
        :param vectors: the original vectors.
        :param root_lower: the lower-bound for the root value to be randomly generated.
        :param root_upper: the upper-bound for the root value to be randomly generated.
        :return: the original vectors appended by the root vector and the default vector.
        """
        vector_size = vectors.syn0.shape[1]

        # root vector
        np.random.seed(9)
        root_vector = np.random.uniform(root_lower, root_upper, (1, vector_size)).astype('float32')
        np.random.seed()

        # default vector
        default_vector = np.zeros((1, vector_size)).astype('float32')
        vectors.syn0 = np.concatenate((vectors.syn0, root_vector, default_vector), axis=0)
        return vectors

    # ============================== Initialization ==============================

    @abstractmethod
    def init(self, graph: NLPGraph):
        """
        :param graph: the input graph
          Initialize each node in the graph with lexicons.
        """

    @classmethod
    def _init(cls, graph: NLPGraph, vectors: KeyedVectors, source_field: str, target_field: str):
        root = graph.nodes[0]

        if vectors and not hasattr(root, target_field):
            setattr(root, target_field, cls._get_root_vector(vectors))

            for node in graph:
                f = getattr(node, source_field)
                setattr(node, target_field, cls._get_vector(vectors, f))

    # ============================== Helpers ==============================

    @classmethod
    def get_default_vector(cls, vectors: KeyedVectors):
        return vectors.syn0[-1] if vectors else None

    @classmethod
    def _get_vector(cls, vectors: KeyedVectors, key: str) -> np.array:
        """
        :return: the vector corresponding to the key if exists; otherwise, the default vector.
        """
        vocab = vectors.vocab.get(key, None)
        return vectors.syn0[vocab.index] if vocab else cls.get_default_vector(vectors)

    @classmethod
    def _get_root_vector(cls, vectors: KeyedVectors):
        return vectors.syn0[-2]
