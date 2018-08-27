# -*- coding: UTF-8 -*-
# Adopted from https://github.com/jcyk/Dynet-Biaffine-dependency-parser
# With some modifications
from collections import Counter
from typing import Union, Sequence, Dict, Any
import numpy as np

from elit.dep.common.savable import Savable
from elit.util.structure import Document, DEP, HEA

from .k_means import KMeans


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
        char_set = set()
        if documents:
            for d in documents:
                for s in d:
                    for word, tag, head, rel in zip(s.tokens, s.part_of_speech_tags, s[HEA], s[DEP]):
                        char_set.update(word)
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
                        assert (len(info) == 10), 'Illegal line: %s' % line
                        word, tag, head, rel = info[1].lower(), info[3], int(info[6]), info[7]
                        char_set.update(info[1])
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

        # Map word to sequence of char
        self._id2char = ['\0', '\1', '\2'] + list(char_set)
        self._char2id = reverse(self._id2char)

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
        train = True
        try:
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
        except FileNotFoundError:
            train = False
        if word_dims is None:
            word_dims = len(data)
        for idx, emb in enumerate(embs):
            if not emb:
                embs[idx] = np.zeros(word_dims)
        pret_embs = np.array(embs, dtype=np.float32)
        return pret_embs / np.std(pret_embs) if train else pret_embs

    def get_word_embs(self, word_dims):
        if self._pret_file is not None:
            return np.random.randn(self.words_in_train, word_dims).astype(np.float32)
        return np.zeros((self.words_in_train, word_dims), dtype=np.float32)

    def get_char_embs(self, char_dims):
        return np.random.randn(len(self._id2char), char_dims).astype(np.float32)

    def get_tag_embs(self, tag_dims):
        return np.random.randn(self.tag_size, tag_dims).astype(np.float32)

    def word2id(self, xs):
        if isinstance(xs, list):
            return [self._word2id.get(x, self.UNK) for x in xs]
        return self._word2id.get(xs, self.UNK)

    def word2char_ids(self, word):
        char_ids = [self._char2id.get(c, self.UNK) for c in word]
        return char_ids

    def get_word2char_ids(self):
        return self._word2char_ids

    def char2id(self, char):
        return self._char2id.get(char, self.UNK)

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
        self._cased_word_id = dict()
        self._cased_id_word = []
        for special_word in ['\0', '\1', '\2']:
            self.cased_word_id(special_word)
        sents = []
        if documents:
            for d in documents:
                for s in d:
                    sent = [
                        [ParserVocabulary.ROOT, ParserVocabulary.ROOT, ParserVocabulary.ROOT, 0, ParserVocabulary.ROOT]]
                    for word, tag, head, rel in zip(s.tokens, s.part_of_speech_tags, s[HEA], s[DEP]):
                        sent.append(
                            [self.cased_word_id(word), vocab.word2id(word.lower()), vocab.tag2id(tag), int(head),
                             vocab.rel2id(rel)])
                    sents.append(sent)
        else:
            sent = [[ParserVocabulary.ROOT, ParserVocabulary.ROOT, ParserVocabulary.ROOT, 0, ParserVocabulary.ROOT]]
            with open(input_file) as f:
                for line in f:
                    info = line.strip().split()
                    if info:
                        assert (len(info) == 10), 'Illegal line: %s' % line
                        assert info[7] in vocab._rel2id, 'Relation OOV: %s' % line
                        word, tag, head, rel = vocab.word2id(info[1].lower()), vocab.tag2id(info[3]), int(
                            info[6]), vocab.rel2id(info[7])
                        sent.append([self.cased_word_id(info[1]), word, tag, head, rel])
                    else:
                        sents.append(sent)
                        sent = [
                            [ParserVocabulary.ROOT, ParserVocabulary.ROOT, ParserVocabulary.ROOT, 0,
                             ParserVocabulary.ROOT]]

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

        self._buckets_word2id = []
        for bkt_idx, (bucket, length) in enumerate(zip(self._buckets, self._bucket_lengths)):
            word2id_bucket = dict()
            for sent in bucket:
                for word in sent:
                    wid = word2id_bucket.get(word[0])
                    if wid is None:
                        wid = len(word2id_bucket)
                        word2id_bucket[word[0]] = wid
                    word[0] = wid
            self._buckets_word2id.append(word2id_bucket)
            self._buckets[bkt_idx] = np.zeros((length, len(bucket), 5), dtype=np.int32)
            for idx, sent in enumerate(bucket):
                self._buckets[bkt_idx][:len(sent), idx, :] = np.array(sent, dtype=np.int32)

    def cased_word_id(self, word):
        wid = self._cased_word_id.get(word)
        if wid is None:
            wid = len(self._cased_word_id)
            self._cased_word_id[word] = wid
            self._cased_id_word.append(word)
        return wid

    @property
    def idx_sequence(self):
        return [x[1] for x in sorted(zip(self._record, list(range(len(self._record)))))]

    def get_batches(self, batch_size, shuffle=True):
        batches = []
        for bkt_idx, bucket in enumerate(self._buckets):
            bucket_size = bucket.shape[1]
            n_tokens = bucket_size * self._bucket_lengths[bkt_idx]
            n_splits = max(n_tokens // batch_size, 1)
            range_func = np.random.permutation if shuffle else np.arange
            for bkt_batch in np.array_split(range_func(bucket_size), n_splits):
                batches.append((bkt_idx, bkt_batch))

        if shuffle:
            np.random.shuffle(batches)

        for bkt_idx, bkt_batch in batches:
            cased_word_inputs = self._buckets[bkt_idx][:, bkt_batch, 0]  # seq_len x batch_size
            word_inputs = self._buckets[bkt_idx][:, bkt_batch, 1]
            tag_inputs = self._buckets[bkt_idx][:, bkt_batch, 2]
            arc_targets = self._buckets[bkt_idx][:, bkt_batch, 3]
            rel_targets = self._buckets[bkt_idx][:, bkt_batch, 4]
            cased_w2i_batch = dict()
            cased_i2w_batch = []
            id_map = dict()
            for word_id in set(np.reshape(cased_word_inputs, (-1,)).tolist()):
                id_map[word_id] = get_word_id(self._cased_id_word[word_id], cased_w2i_batch, cased_i2w_batch)
            cased_word_inputs = np.copy(cased_word_inputs)
            cased_word_inputs = np.vectorize(id_map.get)(cased_word_inputs)
            # max_word_length = max(map(len, cased_w2i_batch))
            char_vocab_inputs = []
            for word in cased_i2w_batch:
                char_vocab_inputs.append([self.vocab.char2id(char) for char in word])

            yield char_vocab_inputs, cased_word_inputs, word_inputs, tag_inputs, arc_targets, rel_targets


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
