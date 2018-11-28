# ========================================================================
# Copyright 2018 ELIT
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
# -*- coding: UTF-8 -*-
# Adopted from https://github.com/jcyk/Dynet-Biaffine-dependency-parser
# With some modifications
from collections import Counter
from typing import Sequence

import numpy as np

from elit.nlp.dep.common.savable import Savable
from elit.nlp.dep.parser.common.k_means import KMeans
from elit.structure import Document, DEP


class ConllWord(object):
    def __init__(self, id, form, lemma=None, cpos=None, pos=None, feats=None, head=None, relation=None, deps=None,
                 misc=None):
        self.id = id
        self.form = form
        self.cpos = cpos
        self.pos = pos
        self.head = head
        self.relation = relation

        self.lemma = lemma
        self.feats = feats
        self.deps = deps
        self.misc = misc

    def __str__(self):
        values = [str(self.id), self.form, self.lemma, self.cpos, self.pos, self.feats, str(self.head), self.relation,
                  self.deps, self.misc]
        return '\t'.join(['_' if v is None else v for v in values])


class ConllSentence(object):
    def __init__(self, words: list) -> None:
        super().__init__()
        self.words = words

    def __str__(self):
        return '\n'.join([word.__str__() for word in self.words])

    def __len__(self):
        return len(self.words)

    def __getitem__(self, index):
        return self.words[index]

    def __iter__(self):
        return (line for line in self.words)


class ParserVocabulary(Savable):
    """
    Vocabulary, holds word, tag and relation along with their id
    """
    PAD, ROOT, UNK = 0, 1, 2
    """Padding, Root, Unknown"""

    def __init__(self, input_file, pret_file=None, min_occur_count=2, documents: Sequence[Document] = None):
        """
        Load from conll file
        :param input_file: conll file
        :param pret_file: word vector file
        :param min_occur_count: threshold of word frequency
        """
        super().__init__()
        word_counter = Counter()
        tag_set = set()
        rel_set = set()
        if documents:
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
                        # else:
                        #     raise RuntimeError('Illegal line: %s' % line)
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

        self._pret_file = pret_file
        self._words_in_train_data = len(self._id2word)
        # print('#words in training set:', self._words_in_train_data)
        if pret_file:
            self._add_pret_words(pret_file)
        self._id2tag += list(tag_set)
        self._id2rel += list(rel_set)

        self._word2id = reverse(self._id2word)
        self._tag2id = reverse(self._id2tag)
        self._rel2id = reverse(self._id2rel)
        # print("Vocab info: #words %d, #tags %d #rels %d" % (self.vocab_size, self.tag_size, self.rel_size))

    def log_info(self, logger):
        logger.info('#words in training set: %d' % self._words_in_train_data)
        logger.info("Vocab info: #words %d, #tags %d #rels %d" % (self.vocab_size, self.tag_size, self.rel_size))

    def _add_pret_words(self, pret_file):
        words_in_train_data = set(self._id2word)
        with open(pret_file) as f:
            for line in f:
                line = line.strip().split()
                if line:
                    word = line[0]
                    if word not in words_in_train_data:
                        self._id2word.append(word)
                        # print 'Total words:', len(self._id2word)

    def has_pret_embs(self):
        return self._pret_file is not None

    def get_pret_embs(self, word_dims=None):
        assert (self._pret_file is not None), "No pretrained file provided."
        embs = [[]] * len(self._id2word)
        with open(self._pret_file) as f:
            dim = None
            for line in f:
                line = line.strip().split()
                if len(line) > 2:
                    if dim is None:
                        dim = len(line)
                    else:
                        if len(line) != dim:
                            continue
                    word, data = line[0], line[1:]
                    embs[self._word2id[word]] = data
        if word_dims is None:
            word_dims = len(data)
        for idx, emb in enumerate(embs):
            if not emb:
                embs[idx] = np.zeros(word_dims)
        pret_embs = np.array(embs, dtype=np.float32)
        return pret_embs / np.std(pret_embs)

    def get_word_embs(self, word_dims):
        if self._pret_file is not None:
            return np.random.randn(self.words_in_train, word_dims).astype(np.float32)
        return np.zeros((self.words_in_train, word_dims), dtype=np.float32)

    def get_tag_embs(self, tag_dims):
        return np.random.randn(self.tag_size, tag_dims).astype(np.float32)

    def word2id(self, xs):
        if isinstance(xs, list):
            return [self._word2id.get(x, self.UNK) for x in xs]
        return self._word2id.get(xs, self.UNK)

    def id2word(self, xs):
        if isinstance(xs, list):
            return [self._id2word[x] for x in xs]
        return self._id2word[xs]

    def rel2id(self, xs):
        if isinstance(xs, list):
            return [self._rel2id[x] for x in xs]
        return self._rel2id[xs]

    def id2rel(self, xs):
        if isinstance(xs, list):
            return [self._id2rel[x] for x in xs]
        return self._id2rel[xs]

    def tag2id(self, xs):
        if isinstance(xs, list):
            return [self._tag2id.get(x, self.UNK) for x in xs]
        return self._tag2id.get(xs, self.UNK)

    @property
    def words_in_train(self):
        """
        #words in training set
        :return:
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
    Load conll data
    """

    def __init__(self, input_file, n_bkts, vocab: ParserVocabulary, documents: Sequence[Document] = None):
        """
        Begin loading
        :param input_file: CoNLL file
        :param n_bkts: number of buckets
        :param vocab: vocabulary object
        """
        self.vocab = vocab
        sents = []
        if documents:
            for d in documents:
                for s in d:
                    sent = [[ParserVocabulary.ROOT, ParserVocabulary.ROOT, 0, ParserVocabulary.ROOT]]
                    for word, tag, head_rel in zip(s.tokens, s.part_of_speech_tags, s[DEP]):
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
        return [x[1] for x in sorted(zip(self._record, list(range(len(self._record)))))]

    def get_batches(self, batch_size, shuffle=True):
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


def get_word_id(word, w2i, i2w=None):
    """
    Get word id from a dictionary

    :param w2i:
    :param word:
    :return:
    """
    wid = w2i.get(word)
    if wid is None:
        wid = len(w2i)
        w2i[word] = wid
        if i2w is not None:
            i2w.append(word)
    return wid


def convert_conll_to_conllx(conll, conllx=None):
    if not conllx:
        conllx = conll + '.conllx'
    with open(conll) as src, open(conllx, 'w') as out:
        for line in src:
            cells = line.strip().split()
            if cells:
                cells = [cells[0], cells[1], cells[2], cells[3], '_', '_', cells[5], cells[6], '_', '_']
                out.write('\t'.join(cells) + '\n')
            else:
                out.write('\n')


if __name__ == '__main__':
    convert_conll_to_conllx('data/dat/en-ddr.trn')
    convert_conll_to_conllx('data/dat/en-ddr.tst')
    convert_conll_to_conllx('data/dat/en-ddr.dev')
