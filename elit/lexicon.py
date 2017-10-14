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
            self.model = KeyedVectors.load(filepath)
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

        # return np.array([emb(i) for i in range(maxlen)])
        # TODO: the following 3 lines should be replaced by the above return statement
        l = [self.model.syn0[0] for _ in range(maxlen-len(document))]
        l.extend([emb(i) for i in range(min(maxlen, len(document)))])
        return np.array(l)

    def docs_to_emb(self, documents, maxlen):
        """
        :param documents: a list of documents.
        :param maxlen:
        :return:
        """
        return np.array([self.doc_to_emb(document, maxlen) for document in documents])