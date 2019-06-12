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
import pickle
from collections import Counter

import gluonnlp
import numpy as np
from elit.component.dep.common.k_means import KMeans
from elit.component.dep.common.savable import Savable
from elit.structure import Sentence, POS, DEP, SENS, Document, SEM


class ParserVocabulary(Savable):
    PAD, ROOT, UNK = 0, 1, 2
    """Padding, Root, Unknown"""

    def __init__(self, input_file, pret_embeddings, min_occur_count=2, root='root', shared_vocab=None):
        """Vocabulary, holds word, tag and relation along with their id.
            Load from conll file
            Adopted from https://github.com/jcyk/Dynet-Biaffine-dependency-parser with some modifications

        Parameters
        ----------
        input_file : str
            conll file
        pret_embeddings : str
            word vector file (plain text)
        min_occur_count : int
            threshold of word frequency, those words with smaller frequency will be replaced by UNK
        """
        super().__init__()
        word_counter = Counter()
        tag_set = set()
        rel_set = set()

        if isinstance(input_file, str):
            with open(input_file) as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    cell = line.strip().split()
                    if cell:
                        word, tag = cell[1].lower(), cell[3]
                        word_counter[word] += 1
                        tag_set.add(tag)
                        rel_set.add(cell[6])
                        token = cell[7]
                        if token != '_':
                            token = token.split('|')
                            for edge in token:
                                pair = edge.split(':', 1)
                                assert len(pair) == 2, 'Illegal {}'.format(line)
                                head, rel = pair
                                if rel != root:
                                    rel_set.add(rel)
        else:
            documents = input_file
            for d in documents:
                for s in d:  # type: Sentence
                    tokens = s.tokens
                    # if LEM in s:
                    #     tokens = s.lemmatized_tokens
                    for word, tag, head_rel in zip(tokens, s.part_of_speech_tags, s[SEM]):
                        word_counter[word] += 1
                        tag_set.add(tag)
                        for (head, rel) in head_rel:
                            rel_set.add(rel)

        self._id2word = ['<pad>', '<root>', '<unk>']
        self._id2tag = ['<pad>', '<root>', '<unk>']
        self._id2rel = ['<pad>', root]
        self.root = root
        reverse = lambda x: dict(list(zip(x, list(range(len(x))))))
        for word, count in word_counter.most_common():
            if count > min_occur_count:
                self._id2word.append(word)

        self._pret_embeddings = pret_embeddings
        self._words_in_train_data = len(self._id2word)
        # print('#words in training set:', self._words_in_train_data)
        if shared_vocab:
            self._id2word = shared_vocab._id2word
            self._id2tag = shared_vocab._id2tag
            self._word2id = shared_vocab._word2id
            self._tag2id = shared_vocab._tag2id
        else:
            if pret_embeddings:
                self._add_pret_words(pret_embeddings)
            self._id2tag += list(sorted(tag_set))
            self._word2id = reverse(self._id2word)
            self._tag2id = reverse(self._id2tag)
        self._id2rel += list(sorted(rel_set))
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


class SDPDataLoader(object):
    """
    Load CoNLL data
    Adopted from https://github.com/jcyk/Dynet-Biaffine-dependency-parser with some modifications
    """

    def __init__(self, input_file, n_bkts, vocab, bert=None):
        """Create a data loader for a data set

        Parameters
        ----------
        input_file : str
            path to CoNLL file
        n_bkts : int
            number of buckets
        vocab : ParserVocabulary
            vocabulary object
        """
        self.vocab = vocab
        sents = []
        sent = [[ParserVocabulary.ROOT, ParserVocabulary.ROOT, [0], [ParserVocabulary.ROOT]]]
        if isinstance(input_file, str):
            with open(input_file) as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    info = line.strip().split()
                    if info:
                        word, tag = vocab.word2id(info[1].lower()), vocab.tag2id(info[3])
                        token = info[7]
                        hs, rs = [int(info[5])], []

                        def insert_rel(rel):
                            if rel not in vocab._rel2id:
                                rel = '<pad>'
                            rs.append(vocab.rel2id(rel))

                        insert_rel(info[6])

                        if token != '_':
                            token = token.split('|')
                            for edge in token:
                                head, rel = edge.split(':', 1)
                                head = int(head)
                                hs.append(head)
                                # assert rel in vocab._rel2id, 'Relation OOV: %s' % line
                                insert_rel(rel)
                        sent.append([word, tag, hs, rs])
                    else:
                        sents.append(sent)
                        sent = [[ParserVocabulary.ROOT, ParserVocabulary.ROOT, [0], [ParserVocabulary.ROOT]]]

                if len(sent) > 1:  # last sent in file without '\n'
                    sents.append(sent)
        else:
            documents = input_file
            for d in documents:
                for s in d:  # type: Sentence
                    sent = [[ParserVocabulary.ROOT, ParserVocabulary.ROOT, [0], [ParserVocabulary.ROOT]]]
                    tokens = s.tokens
                    # if LEM in s:
                    #     tokens = s.lemmatized_tokens
                    for word, tag, head_rel in zip(tokens, s.part_of_speech_tags, s[SEM]):
                        heads = [item[0] for item in head_rel]
                        rels = [vocab.rel2id(item[1]) for item in head_rel]
                        sent.append([vocab.word2id(word.lower()), vocab.tag2id(tag), heads, rels])
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

        self._record = []  # the bucket id of every sent and how many sents are there in that bucket
        for sent in sents:
            bkt_idx = len2bkt[len(sent)]
            idx = len(self._buckets[bkt_idx])
            self._buckets[bkt_idx].append(sent)
            self._record.append((bkt_idx, idx))

        for bkt_idx, (bucket, length) in enumerate(zip(self._buckets, self._bucket_lengths)):
            self._buckets[bkt_idx] = np.zeros((length, len(bucket), 2 + length * 2), dtype=np.int32)
            for idx, sent in enumerate(bucket):
                self._buckets[bkt_idx][:len(sent), idx, :2] = np.array([s[:2] for s in sent], dtype=np.int32)
                for wid, word in enumerate(sent):
                    arc, rel = word[-2], word[-1]
                    for a, r in zip(arc, rel):
                        self._buckets[bkt_idx][wid, idx, 2 + a] = 1
                        self._buckets[bkt_idx][wid, idx, 2 + length + a] = r
                    # self._buckets[bkt_idx][wid, idx, 2:length] =

        if bert is not None:
            with open(bert, 'rb') as f:
                self.bert = pickle.load(f)
            self.bert_dim = self.bert[0].shape[1]
        else:
            self.bert = None
            self.bert_dim = 0

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

        sent_idx = 0
        idx_seq = self.idx_sequence
        for bkt_idx, bkt_batch in batches:
            word_inputs = self._buckets[bkt_idx][:, bkt_batch, 0]  # word_id x sent_id
            tag_inputs = self._buckets[bkt_idx][:, bkt_batch, 1]
            arc_targets = self._buckets[bkt_idx][:, bkt_batch, 2: 2 + word_inputs.shape[0]]  # type: np.ndarray
            arc_targets = arc_targets.transpose((2, 0, 1))  # head x dep x batch
            rel_targets = self._buckets[bkt_idx][:, bkt_batch, 2 + word_inputs.shape[0]:]
            rel_targets = rel_targets.transpose((2, 0, 1))  # head x dep x batch
            if self.bert:
                seq_len = word_inputs.shape[0]
                bat_len = word_inputs.shape[1]
                batch_bert = np.zeros((seq_len, bat_len, self.bert_dim))
                for i in range(bat_len):
                    bert_sent = self.bert[idx_seq[sent_idx]]
                    batch_bert[1:1 + bert_sent.shape[0], i, :] = bert_sent
                    sent_idx += 1
                yield word_inputs, batch_bert, tag_inputs, arc_targets, rel_targets
            else:
                yield word_inputs, None, tag_inputs, arc_targets, rel_targets


class DepDataLoader(object):
    """
    Load conll data
    """

    def __init__(self, input_file, n_bkts, vocab: ParserVocabulary, bert=None):
        """
        Begin loading
        :param input_file: CoNLL file
        :param n_bkts: number of buckets
        :param vocab: vocabulary object
        """
        self.vocab = vocab
        sents = []
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

        if bert is not None:
            with open(bert, 'rb') as f:
                self.bert = pickle.load(f)
            self.bert_dim = self.bert[0].shape[1]
        else:
            self.bert = None
            self.bert_dim = 0

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

        sent_idx = 0
        idx_seq = self.idx_sequence
        for bkt_idx, bkt_batch in batches:
            word_inputs = self._buckets[bkt_idx][:, bkt_batch, 0]  # word_id x sent_id
            tag_inputs = self._buckets[bkt_idx][:, bkt_batch, 1]
            arc_targets = self._buckets[bkt_idx][:, bkt_batch, 2]
            rel_targets = self._buckets[bkt_idx][:, bkt_batch, 3]
            if self.bert:
                seq_len = word_inputs.shape[0]
                bat_len = word_inputs.shape[1]
                batch_bert = np.zeros((seq_len, bat_len, self.bert_dim))
                for i in range(bat_len):
                    bert_sent = self.bert[idx_seq[sent_idx]]
                    batch_bert[1:1 + bert_sent.shape[0], i, :] = bert_sent
                    sent_idx += 1
                yield word_inputs, batch_bert, tag_inputs, arc_targets, rel_targets
            else:
                yield word_inputs, None, tag_inputs, arc_targets, rel_targets


def split_train_dev(train, output_path):
    with open(train) as src, open(output_path + '.train.conllu', 'w') as train, open(output_path + '.dev.conllu',
                                                                                     'w') as dev:
        sents = src.read().split('\n\n')
        train.write("\n\n".join(s for s in sents if 'sent_id = 220' not in s))
        dev.write("\n\n".join(s for s in sents if 'sent_id = 220' in s))  # section 20 for dev


def slice_train_file(train, ratio):
    with open(train) as src, open(train.replace('.conllu', '.{}.conllu'.format(int(ratio * 100))), 'w') as out:
        sents = src.read().split('\n\n')
        out.write("\n\n".join(s for s in sents[:int(len(sents) * ratio)]))


def conll_to_sdp_document(path):
    def create_sentence() -> Sentence:
        sent = Sentence()
        sent[POS] = []
        sent[SEM] = []
        return sent

    sents = []
    with open(path) as src:
        sent = create_sentence()
        for line in src:
            info = line.strip().split()
            if info:
                assert (len(info) == 8), 'Illegal line: %s' % line
                word, tag, head, rel, extra = info[1], info[3], int(info[5]), info[6], info[7]
                sent.tokens.append(word)
                sent.part_of_speech_tags.append(tag)
                arcs = [(head, rel)]
                if extra != '_':
                    extra = extra.split('|')
                    for edge in extra:
                        head, rel = edge.split(':', 1)
                        head = int(head)
                        arcs.append((head, rel))
                sent[SEM].append(arcs)
            else:
                sents.append(sent)
                sent = create_sentence()
    return Document({SENS: sents})


if __name__ == '__main__':
    # vocab = ParserVocabulary('data/semeval15/en.pas.train.conllu',
    #                          pret_file='data/embedding/glove/glove.6B.100d.debug.txt', root='root')
    # vocab.save('data/model/pas/vocab-local.pkl')
    # print(vocab._id2rel)
    # data_loader = DataLoader(train_file, 2, vocab)
    # for data in 'dm', 'pas', 'psd':
    #     split_train_dev('data/semeval15/en.{}.conllu'.format(data), 'data/semeval15/en.{}'.format(data))
    # for data in 'dm', 'pas', 'psd':
    #     for ratio in range(10, 110, 10):
    #         ratio = ratio / 100
    #         slice_train_file('data/semeval15/en.{}.train.conllu'.format(data), ratio)
    # train_file = 'data/semeval15/en.dm.dev.conllu'
    # vocab = ParserVocabulary(train_file,
    #                          pret_file='data/embedding/glove/glove.6B.100d.debug.txt',
    #                          root='root')
    # # vocab.save('data/model/dm-debug/vocab-local.pkl')
    # data_loader = DataLoader(train_file, 2, vocab, bert='data/semeval15/en.dev.bert')
    # next(data_loader.get_batches(10, shuffle=False))
    # with open('data/semeval15/en.dm.train.conllu') as src:
    #     sents = src.read().split('\n\n')
    #     print(len(sents))
    conll_to_sdp_document('data/dat/en-ddr.debug.conll')
