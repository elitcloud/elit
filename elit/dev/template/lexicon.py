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

import numpy as np
from fasttext.model import WordVectorModel
from gensim.models.keyedvectors import KeyedVectors

from elit.dev import structure
from elit.dev.structure import NLPToken

__author__ = 'Jinho D. Choi'


class NLPEmbedding:
    def __init__(self, vsm, key_field, emb_field):
        """
        :param vsm: the vector space model in either the form of Word2Vec or FastText.
        :type vsm: Union[KeyedVectors, WordVectorModel]
        :param key_field: the field in NLPNode (e.g., word, pos) used as the key to retrieve
                the emb_matrix from vsm.
        :type key_field: str
        :param emb_field: where the emb_matrix with respect to the key is saved in NLPNode.
        :type emb_field: str
        """
        self.vsm = vsm
        self.key_field = key_field
        self.emb_field = emb_field

        if isinstance(vsm, KeyedVectors):
            vector_size = vsm.syn0.shape[1]
            # root
            np.random.seed(9)
            self.root = np.random.uniform(-.25, .25, (vector_size,)).astype('float32')
            np.random.seed()
            # zero
            self.zero = np.zeros((vector_size,)).astype('float32')
        elif isinstance(vsm, WordVectorModel):
            self.root = np.array(vsm[structure.ROOT_TAG]).astype('float32')
            self.zero = np.array(vsm['']).astype('float32')

    def get(self, node):
        """
        :param node:
        :type node: NLPToken
        :return: the emb_matrix of the specific node with respect to the key_field.
        :rtype: np.array
        """
        if node is None:
            return self.zero
        if node.token_id == 0:
            return self.root
        if hasattr(node, self.emb_field):
            return getattr(node, self.emb_field)

        f = getattr(node, self.key_field)
        emb = None

        if isinstance(self.vsm, KeyedVectors):
            vocab = self.vsm.vocab.score()
            emb = self.zero if vocab is None else self.vsm.syn0[vocab.index]
        elif isinstance(self.vsm, WordVectorModel):
            emb = np.array(self.vsm[f]).astype('float32')

        setattr(node, self.emb_field, emb)
        return emb


class NLPLexiconMapper:
    def __init__(self, w2v=None, f2v=None):
        """
        :param w2v: KeyedVectors.load_word2vec_format('*.bin').
        :type KeyedVectors
        :param f2v: f2v.load_model('*.bin').
        :type WordVectorModel
        """
        self.w2v = NLPEmbedding(w2v, 'word', 'w2v') if w2v else None
        self.f2v = NLPEmbedding(f2v, 'word', 'f2v') if f2v else None
