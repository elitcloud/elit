# coding: utf-8

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from collections import Counter
from typing import Sequence

import gluonnlp
import numpy as np

from elit.component.dep.common.k_means import KMeans
from elit.component.dep.common.savable import Savable
from elit.structure import DEP, LEM, Sentence


class ParserVocabulary(Savable):
    PAD, ROOT, UNK = 0, 1, 2
    """Padding, Root, Unknown"""

    def __init__(self, input_file, pret_embeddings=None, min_occur_count=2):
        """Vocabulary, holds word, tag and relation along with their id.
            Load from conll file
            Adopted from https://github.com/jcyk/Dynet-Biaffine-dependency-parser with some modifications

        Parameters
        ----------
        input_file : str or Sequence
            conll file
        pret_embeddings : tuple
            (embedding_name, source), used for gluonnlp.embedding.create(embedding_name, source)
        min_occur_count : int
            threshold of word frequency, those words with smaller frequency will be replaced by UNK
        """
        super().__init__()
        word_counter = Counter()
        tag_set = set()
        rel_set = set()
        if isinstance(input_file, list):
            documents = input_file
            for d in documents:
                for s in d:
                    for word, tag, head_rel in zip(s.tokens, s.part_of_speech_tags, s[DEP]):
                        rel = head_rel[1]
                        word.lower()
                        word_counter[word] += 1
                        tag_set.add(tag)
                        if rel != 'root':
                            rel_set.add(rel)
        else:
            with open(input_file) as f:
                for line in f:
                    info = line.strip().split()
                    if info:
                        if len(info) == 10:
                            arc_offset = 6
                            rel_offset = 7
                        elif len(info) == 8:
                            arc_offset = 5
                            rel_offset = 6
                        word, tag, head, rel = info[1].lower(), info[3], int(info[arc_offset]), info[rel_offset]
                        word_counter[word] += 1
                        tag_set.add(tag)
                        if rel != 'root':
                            rel_set.add(rel)

        self._id2word = ['<pad>', '<root>', '<unk>']
        self._id2tag = ['<pad>', '<root>', '<unk>']
        self._id2rel = ['<pad>', 'root']
        reverse = lambda x: dict(list(zip(x, list(range(len(x))))))
        for word, count in word_counter.most_common():
            if count > min_occur_count:
                self._id2word.append(word)

        self._pret_embeddings = pret_embeddings
        self._words_in_train_data = len(self._id2word)
        # print('#words in training set:', self._words_in_train_data)
        if pret_embeddings:
            self._add_pret_words(pret_embeddings)
        self._id2tag += list(tag_set)
        self._id2rel += list(rel_set)

        self._word2id = reverse(self._id2word)
        self._tag2id = reverse(self._id2tag)
        self._rel2id = reverse(self._id2rel)
        # print("Vocab info: #words %d, #tags %d #rels %d" % (self.vocab_size, self.tag_size, self.rel_size))

    def log_info(self, logger):
        """Print statistical information via the provided logger

        Parameters
        ----------
        logger : logging.Logger
            logger created using logging.getLogger()
        """
        logger.info('#words in training set: %d' % self._words_in_train_data)
        logger.info("Vocab info: #words %d, #tags %d #rels %d" % (self.vocab_size, self.tag_size, self.rel_size))

    def _add_pret_words(self, pret_embeddings):
        """Read pre-trained embedding file for extending vocabulary

        Parameters
        ----------
        pret_embeddings : tuple
            (embedding_name, source), used for gluonnlp.embedding.create(embedding_name, source)
        """
        words_in_train_data = set(self._id2word)
        pret_embeddings = gluonnlp.embedding.create(pret_embeddings[0], source=pret_embeddings[1])

        for idx, token in enumerate(pret_embeddings.idx_to_token):
            if token not in words_in_train_data:
                self._id2word.append(token)

    def has_pret_embs(self):
        """Check whether this vocabulary contains words from pre-trained embeddings

        Returns
        -------
        bool
            Whether this vocabulary contains words from pre-trained embeddings
        """
        return self._pret_embeddings is not None

    def get_pret_embs(self, word_dims=None):
        """Read pre-trained embedding file

        Parameters
        ----------
        word_dims : int or None
            vector size. Use `None` for auto-infer
        Returns
        -------
        numpy.ndarray
            T x C numpy NDArray
        """
        assert (self._pret_embeddings is not None), "No pretrained file provided."
        pret_embeddings = gluonnlp.embedding.create(self._pret_embeddings[0], source=self._pret_embeddings[1])
        embs = [None] * len(self._id2word)
        for idx, vec in enumerate(pret_embeddings.idx_to_vec):
            embs[idx] = vec.asnumpy()
        if word_dims is None:
            word_dims = len(pret_embeddings.idx_to_vec[0])
        for idx, emb in enumerate(embs):
            if emb is None:
                embs[idx] = np.zeros(word_dims)
        pret_embs = np.array(embs, dtype=np.float32)
        return pret_embs / np.std(pret_embs)

    def get_word_embs(self, word_dims):
        """Get randomly initialized embeddings when pre-trained embeddings are used, otherwise zero vectors

        Parameters
        ----------
        word_dims : int
            word vector size
        Returns
        -------
        numpy.ndarray
            T x C numpy NDArray
        """
        if self._pret_embeddings is not None:
            return np.random.randn(self.words_in_train, word_dims).astype(np.float32)
        return np.zeros((self.words_in_train, word_dims), dtype=np.float32)

    def get_tag_embs(self, tag_dims):
        """Randomly initialize embeddings for tag

        Parameters
        ----------
        tag_dims : int
            tag vector size

        Returns
        -------
        numpy.ndarray
            random embeddings
        """
        return np.random.randn(self.tag_size, tag_dims).astype(np.float32)

    def word2id(self, xs):
        """Map word(s) to its id(s)

        Parameters
        ----------
        xs : str or list
            word or a list of words

        Returns
        -------
        int or list
            id or a list of ids
        """
        if isinstance(xs, list):
            return [self._word2id.get(x, self.UNK) for x in xs]
        return self._word2id.get(xs, self.UNK)

    def id2word(self, xs):
        """Map id(s) to word(s)

        Parameters
        ----------
        xs : int
            id or a list of ids

        Returns
        -------
        str or list
            word or a list of words
        """
        if isinstance(xs, list):
            return [self._id2word[x] for x in xs]
        return self._id2word[xs]

    def rel2id(self, xs):
        """Map relation(s) to id(s)

        Parameters
        ----------
        xs : str or list
            relation

        Returns
        -------
        int or list
            id(s) of relation
        """
        if isinstance(xs, list):
            return [self._rel2id[x] for x in xs]
        return self._rel2id[xs]

    def id2rel(self, xs):
        """Map id(s) to relation(s)

        Parameters
        ----------
        xs : int
            id or a list of ids

        Returns
        -------
        str or list
            relation or a list of relations
        """
        if isinstance(xs, list):
            return [self._id2rel[x] for x in xs]
        return self._id2rel[xs]

    def tag2id(self, xs):
        """Map tag(s) to id(s)

        Parameters
        ----------
        xs : str or list
            tag or tags

        Returns
        -------
        int or list
            id(s) of tag(s)
        """
        if isinstance(xs, list):
            return [self._tag2id.get(x, self.UNK) for x in xs]
        return self._tag2id.get(xs, self.UNK)

    @property
    def words_in_train(self):
        """
        get #words in training set
        Returns
        -------
        int
            #words in training set
        """
        return self._words_in_train_data

    @property
    def vocab_size(self):
        return len(self._id2word)

    @property
    def tag_size(self):
        return len(self._id2tag)

    @property
    def rel_size(self):
        return len(self._id2rel)


class DataLoader(object):
    """
    Load CoNLL data
    Adopted from https://github.com/jcyk/Dynet-Biaffine-dependency-parser with some modifications
    """

    def __init__(self, input_file, n_bkts, vocab: ParserVocabulary):
        """Create a data loader for a data set

        Parameters
        ----------
        input_file : str or Sequence
            path to CoNLL file
        n_bkts : int
            number of buckets
        vocab : ParserVocabulary
            vocabulary object
        """
        self.vocab = vocab
        sents = []
        if isinstance(input_file, list):
            documents = input_file
            for d in documents:
                for s in d:  # type: Sentence
                    sent = [[ParserVocabulary.ROOT, ParserVocabulary.ROOT, 0, ParserVocabulary.ROOT]]
                    tokens = s.tokens
                    # if LEM in s:
                    #     tokens = s.lemmatized_tokens
                    for word, tag, head_rel in zip(tokens, s.part_of_speech_tags, s[DEP]):
                        head, rel = head_rel
                        sent.append([vocab.word2id(word.lower()), vocab.tag2id(tag), int(head), vocab.rel2id(rel)])
                    sents.append(sent)
        else:
            sent = [[ParserVocabulary.ROOT, ParserVocabulary.ROOT, 0, ParserVocabulary.ROOT]]
            with open(input_file) as f:
                for line in f:
                    info = line.strip().split()
                    if info:
                        arc_offset = 5
                        rel_offset = 6
                        if len(info) == 10:
                            arc_offset = 6
                            rel_offset = 7
                        # else:
                        #     raise RuntimeError('Illegal line: %s' % line)
                        assert info[rel_offset] in vocab._rel2id, 'Relation OOV: %s' % line
                        word, tag, head, rel = vocab.word2id(info[1].lower()), vocab.tag2id(info[3]), int(
                            info[arc_offset]), vocab.rel2id(info[rel_offset])
                        sent.append([word, tag, head, rel])
                    else:
                        sents.append(sent)
                        sent = [[ParserVocabulary.ROOT, ParserVocabulary.ROOT, 0, ParserVocabulary.ROOT]]
                if len(sent) > 1:  # last sent in file without '\n'
                    sents.append(sent)

        self.samples = len(sents)
        len_counter = Counter()
        for sent in sents:
            len_counter[len(sent)] += 1
        n_bkts = min(n_bkts, len(len_counter))
        self._bucket_lengths = KMeans(n_bkts, len_counter).splits
        self._buckets = [[] for i in range(n_bkts)]
        """bkt_idx x length x sent_idx x 4"""
        len2bkt = {}
        prev_length = -1
        for bkt_idx, length in enumerate(self._bucket_lengths):
            len2bkt.update(list(zip(list(range(prev_length + 1, length + 1)), [bkt_idx] * (length - prev_length))))
            prev_length = length

        self._record = []
        for sent in sents:
            bkt_idx = len2bkt[len(sent)]
            idx = len(self._buckets[bkt_idx])
            self._buckets[bkt_idx].append(sent)
            self._record.append((bkt_idx, idx))

        for bkt_idx, (bucket, length) in enumerate(zip(self._buckets, self._bucket_lengths)):
            self._buckets[bkt_idx] = np.zeros((length, len(bucket), 4), dtype=np.int32)
            for idx, sent in enumerate(bucket):
                self._buckets[bkt_idx][:len(sent), idx, :] = np.array(sent, dtype=np.int32)

    @property
    def idx_sequence(self):
        """Indices of sentences when enumerating data set from batches.
        Useful when retrieving the correct order of sentences

        Returns
        -------
        list
            List of ids ranging from 0 to #sent -1
        """
        return [x[1] for x in sorted(zip(self._record, list(range(len(self._record)))))]

    def get_batches(self, batch_size, shuffle=True):
        """Get batch iterator

        Parameters
        ----------
        batch_size : int
            size of one batch
        shuffle : bool
            whether to shuffle batches. Don't set to True when evaluating on dev or test set.
        Returns
        -------
        tuple
            word_inputs, tag_inputs, arc_targets, rel_targets
        """
        batches = []
        for bkt_idx, bucket in enumerate(self._buckets):
            bucket_size = bucket.shape[1]
            n_tokens = bucket_size * self._bucket_lengths[bkt_idx]
            n_splits = min(max(n_tokens // batch_size, 1), bucket_size)
            range_func = np.random.permutation if shuffle else np.arange
            for bkt_batch in np.array_split(range_func(bucket_size), n_splits):
                batches.append((bkt_idx, bkt_batch))

        if shuffle:
            np.random.shuffle(batches)

        for bkt_idx, bkt_batch in batches:
            word_inputs = self._buckets[bkt_idx][:, bkt_batch, 0]  # word_id x sent_id
            tag_inputs = self._buckets[bkt_idx][:, bkt_batch, 1]
            arc_targets = self._buckets[bkt_idx][:, bkt_batch, 2]
            rel_targets = self._buckets[bkt_idx][:, bkt_batch, 3]
            yield word_inputs, tag_inputs, arc_targets, rel_targets


def conll_8_to_10(src, dst):
    with open(src) as src, open(dst, 'w') as out:
        for line in src:
            info = line.strip().split()
            if info:
                arc_offset = 5
                rel_offset = 6
                if len(info) == 10:
                    arc_offset = 6
                    rel_offset = 7
                idx, word, lemma, tag, head, rel = info[0], info[1], info[2], info[3], info[arc_offset], info[
                    rel_offset]
                out.write('\t'.join([idx, word, lemma, tag, '_', '_', head, rel, '_', '_']))
            out.write('\n')


if __name__ == '__main__':
    conll_8_to_10('data/dat/en-ddr.trn', 'data/dat/en-ddr.trn.conllx')
    conll_8_to_10('data/dat/en-ddr.tst', 'data/dat/en-ddr.tst.conllx')
