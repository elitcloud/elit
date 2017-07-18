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
from typing import Union

import numpy as np
from fasttext.model import WordVectorModel
from gensim.models.keyedvectors import KeyedVectors

from elit import structure
from elit.structure import NLPNode

__author__ = 'Jinho D. Choi'


class NLPEmbedding:
    def __init__(self, vsm: Union[KeyedVectors, WordVectorModel], key_field: str, emb_field: str):
        """
        :param vsm: the vector space model in either the form of Word2Vec or FastText.
        :param key_field: the field in NLPNode (e.g., word, pos) used as the key to retrieve the embedding from vsm.
        :param emb_field: where the embedding with respect to the key is saved in NLPNode.
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

    def get(self, node: NLPNode) -> np.array:
        """
        :return: the embedding of the specific node with respect to the key_field.
        """
        if node is None: return self.zero
        if node.node_id == 0: return self.root
        if hasattr(node, self.emb_field): return getattr(node, self.emb_field)

        f = getattr(node, self.key_field)
        emb = None

        if isinstance(self.vsm, KeyedVectors):
            vocab = self.vsm.vocab.get(f, None)
            emb = self.zero if vocab is None else self.vsm.syn0[vocab.index]
        elif isinstance(self.vsm, WordVectorModel):
            emb = np.array(self.vsm[f]).astype('float32')

        setattr(node, self.emb_field, emb)
        return emb


class NLPLexiconMapper:
    def __init__(self, w2v: KeyedVectors=None, f2v: WordVectorModel=None):
        """
        :param w2v: KeyedVectors.load_word2vec_format('*.bin').
        :param f2v: f2v.load_model('*.bin').
        """
        self.w2v: NLPEmbedding = NLPEmbedding(w2v, 'word', 'w2v') if w2v else None
        self.f2v: NLPEmbedding = NLPEmbedding(f2v, 'word', 'f2v') if f2v else None
