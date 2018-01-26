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
import codecs
import logging
import marisa_trie

import numpy as np
from fasttext import fasttext
from gensim.models import KeyedVectors

__author__ = 'Jinho D. Choi'


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
        :param label: the class label.
        :type label: str
        :return: the ID of the class label if exists; otherwise, -1.
        :rtype: int
        """
        return self.index_map[label] if label in self.index_map else -1

    def get(self, index):
        """
        :param index: the ID of the class label.
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
        :return: the ID of the class label.
        :rtype int
        """
        idx = self.index(label)
        if idx < 0:
            idx = len(self.labels)
            self.index_map[label] = idx
            self.labels.append(label)
        return idx


class VectorSpaceModel(object):
    def __init__(self, model, dim):
        """
        :param model: the word embedding model.
        :type model: Union[KeyedVectors]
        :param dim: the dimension of each word embedding.
        :type dim: int
        """
        self.model = model
        self.dim = dim

        # for zero padding
        self.zero = np.zeros(dim).astype('float32')

    @abc.abstractmethod
    def _get(self, word):
        """
        This is an auxiliary function for #get().
        :param word: a word form (must not be None).
        :type word: str
        :return: the embedding of the word.
        :rtype: numpy.array
        """
        return

    def get(self, word=None):
        """
        :param word: a word form.
        :type word: Union[str, None]
        :return: the embedding of the word; if word is None, the zero embedding.
        :rtype: numpy.array
        """
        return self.zero if word is None else self._get(word)

    def get_list(self, words):
        """
        :param words: a list of words.
        :type words: list of str
        :return: the list of embeddings for the corresponding words.
        :rtype: list of mxnet.nd.array
        """
        return [self.get(word) for word in words]


class FastText(VectorSpaceModel):
    def __init__(self, filepath):
        """
        :param filepath: the path to the file containing word embeddings.
        :type filepath: str
        """
        model = fasttext.load_model(filepath)
        dim = model.dim
        super(FastText, self).__init__(model, dim)
        logging.info('Init: %s (vocab = %d, dim = %d)' % (filepath, len(model.words), dim))

    def _get(self, word):
        return np.array(self.model[word]).astype('float32')


class Word2Vec(VectorSpaceModel):
    def __init__(self, filepath):
        """
        :param filepath: the path to the file containing word embeddings.
        :type filepath: str
        """
        model = KeyedVectors.load(filepath) if filepath.endswith('.gnsm') else KeyedVectors.load_word2vec_format(filepath, binary=True)
        dim = model.syn0.shape[1]
        super(Word2Vec, self).__init__(model, dim)
        logging.info('Init: %s (vocab = %d, dim = %d)' % (filepath, len(model.vocab), dim))

    def _get(self, word):
        vocab = self.model.vocab.get(word, None)
        return self.model.syn0[vocab.index] if vocab else self.zero


class NamedEntityTree:
    def __init__(self, filenames):
        """
        :param filepath: the path to the resource files (e.g., resources/nament/english).
        :type filepath: str
        """
        keys, values = [], []
        for i, filename in enumerate(sorted(filenames)):
            fin = codecs.open(filename, mode='r', encoding='utf-8')
            l = [''+u' '.join(line.split()) for line in fin]
            logging.info('Init: %s (%d)' % (filename, len(l)))
            keys.extend(l)
            values.extend([[i]]*len(l))

        self.trie = marisa_trie.RecordTrie("h", zip(keys, values))

    # def get_list(self, words):
    #     for i, word in enumerate(words):
    #         for j in range(i+1, len(words)):
    #             w = u' '.join(words[i:j])
    #             if w not in


class Word2VecTmp:
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
            vocab = self.model.vocab.get(document[token_index], None)
            return self.model.syn0[vocab.index] if vocab else self.pad

        return np.array([emb(i) for i in range(maxlen)])
        # TODO: the following 3 lines should be replaced by the above return statement
        # l = [self.model.syn0[0] for _ in range(maxlen-len(document))]
        # l.extend([emb(i) for i in range(min(maxlen, len(document)))])
        # return np.array(l)

    def docs_to_emb(self, documents, maxlen):
        """
        :param documents: a list of documents.
        :param maxlen:
        :return:
        """
        return np.array([self.doc_to_emb(document, maxlen) for document in documents])