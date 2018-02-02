# -*- coding: UTF-8 -*-

from collections import Counter
import numpy as np

from .k_means import KMeans


class CoNLLSentence(object):
    """
    A CoNLL sentence
    """

    def __init__(self, words, tags, arcs, rels) -> None:
        """
        Create a sentence

        ----------
        :param words: list of words
        :param tags: list of pos-tags
        :param arcs: list of heads
        :param rels: list of relation types
        """
        length = len(words)
        self.array = [['_' for i in range(10)] for j in range(length)]
        for i in range(length):
            self.array[i][0] = i + 1
            self.array[i][1] = words[i]
            self.array[i][3] = self.array[i][4] = tags[i]
            self.array[i][6] = arcs[i]
            self.array[i][7] = rels[i]

    def __str__(self) -> str:
        return '\n'.join(["\t".join(str(i) for i in line) for line in self.array])

    def __len__(self):
        return len(self.array)

    def __getitem__(self, index):
        return self.array[index]

    def __iter__(self):
        return (line for line in self.array)


class Vocabulary(object):
    """
    Vocabulary, holds word, tag and relation along with their id.
    Adopted from: https://github.com/jcyk/Dynet-Biaffine-dependency-parser
    """
    PAD, ROOT, UNK = 0, 1, 2
    """Padding, Root, OOV"""

    def __init__(self, input_file, pret_file=None, min_occur_count=2):
        """
        Load from conll file
        :param input_file: conll file
        :param pret_file: word vector file
        :param min_occur_count: threshold of word frequency
        """
        word_counter = Counter()
        tag_set = set()
        rel_set = set()
        with open(input_file) as f:
            for line in f:
                info = line.strip().split()
                if info:
                    assert (len(info) == 10), 'Illegal line: %s' % line
                    word, tag, head, rel = info[1].lower(), info[3], int(info[6]), info[7]
                    word_counter[word] += 1
                    tag_set.add(tag)
                    if rel != 'root':
                        rel_set.add(rel)

        self._id2word = ['<pad>', '<root>', '<unk>']
        self._id2tag = ['<pad>', '<root>', '<unk>']
        self._id2rel = ['<pad>', 'root']
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

        reverse = lambda x: dict(list(zip(x, list(range(len(x))))))
        self._word2id = reverse(self._id2word)
        self._tag2id = reverse(self._id2tag)
        self._rel2id = reverse(self._id2rel)
        # print("Vocab info: #words %d, #tags %d #rels %d" % (self.vocab_size, self.tag_size, self.rel_size))

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

    def get_pret_embs(self):
        assert (self._pret_file is not None), "No pretrained file provided."
        embs = [[]] * len(self._id2word)
        with open(self._pret_file) as f:
            for line in f:
                line = line.strip().split()
                if line:
                    word, data = line[0], line[1:]
                    embs[self._word2id[word]] = data
        emb_size = len(data)
        for idx, emb in enumerate(embs):
            if not emb:
                embs[idx] = np.zeros(emb_size)
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

    def sentence2id(self, sentence):
        length = len(sentence)
        word_ids = np.ones((length + 1,), dtype=np.int32)  # append root=1
        tag_ids = np.ones((length + 1,), dtype=np.int32)
        for idx, (word, tag) in enumerate(sentence):
            word_ids[idx + 1] = self.word2id(word.lower())
            tag_ids[idx + 1] = self.tag2id(tag)
        return word_ids, tag_ids

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


class DataSet(object):
    """
    Load conll data.
    Adopted from: https://github.com/jcyk/Dynet-Biaffine-dependency-parser
    """

    def __init__(self, input_file, n_bkts, vocab):
        """
        Begin loading

        :param input_file: CoNLL file
        :param n_bkts: number of buckets
        :param vocab: vocabulary object
        """
        sents = []
        sent = [[Vocabulary.ROOT, Vocabulary.ROOT, 0, Vocabulary.ROOT]]
        with open(input_file) as f:
            for line in f:
                info = line.strip().split()
                if info:
                    assert (len(info) == 10), 'Illegal line: %s' % line
                    word, tag, head, rel = vocab.word2id(info[1].lower()), vocab.tag2id(info[3]), int(
                        info[6]), vocab.rel2id(info[7])
                    sent.append([word, tag, head, rel])
                else:
                    sents.append(sent)
                    sent = [[Vocabulary.ROOT, Vocabulary.ROOT, 0, Vocabulary.ROOT]]

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
            n_splits = max(n_tokens // batch_size, 1)
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
            yield word_inputs.T, tag_inputs.T, arc_targets.T, rel_targets.T
