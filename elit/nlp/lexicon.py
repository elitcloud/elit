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
import os
import mxnet as mx
import pickle

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
    def _get(self, word, index=0, sentence=None):
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

    def _get(self, word, index=0, sentence=None):
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

    def _get(self, word, index=0, sentence=None):
        vocab = self.model.vocab.get(word, None)
        return self.model.syn0[vocab.index] if vocab else self.zero

class Pos2Vec(VectorSpaceModel):
    def __init__(self, dim):
        self.label_map = {'WRB':0, 'WP':1, 'EX':2, 'VBP':3, 'WDT':4, 'PRP$':5, 'AFX':6, 'CC':7, 'PRP':8, 'NNPS':9,
                          'CD':10, 'RB':11, '\'\'':12, 'TO':13, 'VBG':14, 'VBN':15, 'POS':16, '$':17, 'IN':18, 'UH':19,
                          'JJR':20, 'FW':21, 'VBD':22, 'NNS':23, 'RBR':24, 'PDT':25, '.':26, '-RRB-':27, 'VBZ':28,
                          'RP':29, 'NN':30, 'XX':31, 'SYM':32, '-LRB-':33, 'WP$':34, 'VB':35, 'JJ':36, 'NNP':37,
                          'GW':38, 'JJS':39, 'LS':40, 'RBS':41, 'ADD':42, ':':43, 'DT':44, 'HYPH':45, 'MD':46,
                          'NFP':47, '``':48, ',':49}
        model = [np.random.rand(dim) for _ in self.label_map]
        super(Pos2Vec, self).__init__(model, dim)
        logging.info('Init: (Pos2Vec, dim = %d)' % dim)

    def _get(self, word, index=0, sentence=None):
        return self.model[self.label_map[word]]

class NamedEntityTree(VectorSpaceModel):
    def __init__(self, filepath):
        """
        :param filepath: the path to the resource files (e.g., resources/nament/english).
        :type filepath: str
        """
        savepath = '/home/jcoves/data'
        savename = 'gaze.p'
        self.num_labels = len(os.listdir(filepath))
        dim = self.num_labels  # * 3
        if savename not in os.listdir(savepath):
            keys, values = [], []
            for i, filename in enumerate(sorted(os.listdir(filepath))):
                # if filename == 'known_places.txt': continue
                print('path = ', os.path.join(filepath, filename))
                # with open(os.path.join(filepath, filename)) as file:
                #    l = [line.strip('\n') for line in file.readlines()]
                # fin = codecs.open(os.path.join(filepath, filename), mode='r', encoding='latin-1')
                with codecs.open(os.path.join(filepath, filename), mode='r', encoding='latin-1') as fin:
                    l = ['' + u' '.join(line.split()) for line in fin]
                logging.info('Init: %s (%d)' % (filename, len(l)))
                keys.extend(l)
                values.extend([[i]] * len(l))

            self.trie = marisa_trie.RecordTrie("h", zip(keys, values))
            pickle.dump(self.trie, open(savepath+'/'+savename, 'wb'))
        else:
            self.trie = pickle.load(open(savepath+'/'+savename, 'rb'))

        model = self.trie
        # model = KeyedVectors.load(filepath) if filepath.endswith('.gnsm')
        # else KeyedVectors.load_word2vec_format(filepath, binary=True)
        # dim = model.syn0.shape[1]

        super(NamedEntityTree, self).__init__(None, dim)
        logging.info('Init: %s (NE-vocab = %d, NE-labels = %d)' % (filepath, len(self.trie.keys()), dim))

    def _get(self, word, index=0, sentence=None):
        print('ERROR - should not be called')
        return self.get_entity_vectors(sentence)[index] if sentence else self.zero

    # Override
    def get_list(self, words):
        """
        :param words: a list of words.
        :type words: list of str
        :return: the list of embeddings for the corresponding words.
        :rtype: list of mxnet.nd.array
        """
        return self.get_entity_vectors(words) if self.dim == self.num_labels else self.get_expanded_entity_vectors(words)

        # return flat list of vectors
        # return [item for sublist in vectors for item in sublist]
        # return [vectors[i] for i in range(len(words))]
        # return [self.get(word) for word in words]

    def get_entity_vectors(self, words):
        """
        :param words
        :return: entity vector for each word
        """
        # entity_vectors = [mx.nd.zeros(self.num_labels) for _ in words]
        entity_vectors = [np.zeros(self.num_labels) for _ in words]
        for i, word in enumerate(words):
            entity = ''
            components = []
            for j in range(i, len(words)):
                if j > i:
                    entity += ' '
                entity += words[j]
                components.append(j)
                if entity not in self.trie:
                    if not self.trie.keys(entity):  # subtrie is empty
                        break
                    else:
                        continue
                for label in self.trie[entity]:
                    # print('entitiy = ', entity)
                    for component in components:
                        entity_vectors[component][label] += (j+1-i)*(j+1-i)
                        # entity_vectors[component][label] = 1
        entity_vectors = [normalize(vector) for vector in entity_vectors]
        return entity_vectors

    def get_expanded_entity_vectors(self, words):
        """
        :param words
        :return: entity vector for each word
        """
        entity_vectors = [np.zeros(3 * self.num_labels) for _ in words]
        for i, word in enumerate(words):
            entity = ''
            components = []
            for j in range(i, len(words)):
                if j > i:
                    entity += ' '
                entity += words[j]
                components.append(j)
                if entity not in self.trie:
                    if not self.trie.keys(entity):  # subtrie is empty
                        break
                    else:
                        continue
                for label in self.trie[entity]:
                    # print('entitiy = ', entity)
                    for ind in components:
                        # print(label[0], entity_vectors[ind][label])
                        if ind == i: entity_vectors[ind][3*label[0]] += (j+1-i)*(j+1-i)
                        elif ind == j: entity_vectors[ind][3*label[0] + 1] += (j+1-i)*(j+1-i)
                        else: entity_vectors[ind][3*label[0] + 2] += (j+1-i)*(j+1-i)
        entity_vectors = [normalize(vector) for vector in entity_vectors]
        return entity_vectors


def normalize(v):
    norm = sum(v)
    if norm < 0.0001:
        return v
    return v / norm


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