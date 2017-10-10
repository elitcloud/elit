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
from gensim.models import KeyedVectors

__author__ = 'Jinho D. Choi'


class Word2Vec:
    def __init__(self, filepath):
        """
        :param filepath: the path to the file containing word embeddings.
        """
        self.model = None

        if filepath.endswith('.gnsm'):
            self.model = KeyedVectors.load(filepath, mmap='r')
        elif filepath.endswith('.bin'):
            self.model = KeyedVectors.load_word2vec_format(filepath, binary=True)
        else:
            raise ValueError('Unknown type: ' + filepath)

        self.dim = self.model.syn0.shape[1]
        self.pad = np.zeros((self.dim,)).astype('float32')
        print('Init: %s (vocab = %d, dim = %d)' % (filepath, len(self.model.vocab), self.dim))

    def doc_to_emb(self, document, maxlen):
        """
        :param document: the document comprising tokens to retrieve word embeddings for.
        :param maxlen: the maximum length of the document (# of tokens).
        :return: the list of word embeddings corresponding to the tokens in the document.
        """
        def emb(token_index):
            if token_index >= len(document): return self.pad
            v = self.model.vocab.get(document[token_index], None)
            return self.model.syn0[v.index] if v is not None else self.pad

        return np.array([emb(i) for i in range(maxlen)])

    def docs_to_emb(self, documents, maxlen):
        """
        :param documents: a list of documents.
        :param maxlen:
        :return:
        """
        return np.array([self.doc_to_emb(document, maxlen) for document in documents])





# import time
# import numpy as np
# filepath = '/Users/jdchoi/Downloads/tmp/w2v/w2v-400-amazon-review.gnsm'
# start = time.time()
# emb_model = Word2Vec(filepath)
# end = time.time()
# print(end-start)
#
# documents = [['hello', '', 'world'], ['hello', '', 'world']]
# print(emb_model.docs_to_emb(documents, 5))

# start = time.time()
# word_index = emb_model.vocab
# emd_dim = emb_model.syn0.shape[1]
# emb_matrix = np.zeros((len(word_index), emd_dim), dtype=np.float32)
# emb_matrix[:len(emb_model.syn0)] = emb_model.syn0
# # for word, i in word_index.items():
# #     emb = emb_model[word]
# #     if emb is not None:   # words not found in w2v_model are set to zeros
# #         # if not np.array_equal(emb, emb_model.syn0[i.index]):
# #             # print('NOOOO')
# #         emb_matrix[i.index] = emb
#
# end = time.time()
# print(end-start)
