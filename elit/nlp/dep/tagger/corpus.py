# -*- coding:utf-8 -*-
# Filename: corpus.py
# Author：hankcs
# Date: 2018-02-21 14:51
import numpy as np
import pickle
import os


class StringIdMapper(object):
    def __init__(self) -> None:
        super().__init__()
        self.s2i = dict()
        self.i2s = list()

    def get_id(self, s):
        return self.s2i.get(s)

    def get_str(self, i):
        return self.i2s[i]

    def ensure_id(self, s):
        i = self.s2i.get(s)
        if i is None:
            i = len(self.i2s)
            self.s2i[s] = i
            self.i2s.append(s)
        return i

    def get_string(self, i):
        return self.i2s[i]

    def size(self):
        return len(self.i2s)


class Vocabulary(object):
    UNK_TAG = "<UNK>"
    UNK_ID = 0
    START_TAG = "<START>"
    START_ID = 0
    END_TAG = "<STOP>"
    END_ID = 1

    def __init__(self, lower_case=True, use_chars=False) -> None:
        super().__init__()
        self.lower_case = lower_case
        self.use_chars = use_chars
        self.word_vocab = StringIdMapper()
        self.word_vocab.ensure_id(Vocabulary.UNK_TAG)
        if use_chars:
            self.char_vocab = StringIdMapper()
            self.char_vocab.ensure_id(Vocabulary.UNK_TAG)
        self.tag_vocab = StringIdMapper()
        Vocabulary.START_ID = self.tag_vocab.ensure_id(Vocabulary.START_TAG)
        Vocabulary.END_ID = self.tag_vocab.ensure_id(Vocabulary.END_TAG)
        self.num_words_in_train = 0
        self.pret_word_embs = None

    def locked(self):
        return self.num_words_in_train > 0

    def word_id(self, word: str):
        if self.lower_case:
            word = word.lower()
        return self.word_vocab.get_id(word)

    def word(self, word_id):
        return self.word_vocab.get_str(word_id)

    def tag(self, tag_id):
        return self.tag_vocab.get_str(tag_id)

    def ensure_word_id(self, word):
        if self.lower_case:
            word = word.lower()
        if self.locked():
            word_id = self.word_vocab.get_id(word)
            if word_id is None:
                return Vocabulary.UNK_ID
        return self.word_vocab.ensure_id(word)

    def ensure_word_ids(self, word: str):
        return (self.ensure_char_ids(word), self.ensure_word_id(word)) if self.use_chars else self.ensure_word_id(word)

    def ensure_char_ids(self, word):
        return [self.ensure_char_id(c) for c in word]

    def char_id(self, char):
        return self.char_vocab.get_id(char)

    def ensure_char_id(self, char):
        if self.locked():
            char_id = self.char_vocab.get_id(char)
            if char_id is None:
                return Vocabulary.UNK_ID
        return self.char_vocab.ensure_id(char)

    def tag_id(self, tag):
        return self.tag_vocab.get_id(tag)

    def ensure_tag_id(self, tag):
        return self.tag_vocab.ensure_id(tag)

    def tagset_size(self):
        return len(self.tag_vocab.i2s)

    def add_pret_words(self, pret_file, keep_oov=True):
        if self.pret_word_embs:
            return
        self.num_words_in_train = self.word_vocab.size()
        if not os.path.isfile(pret_file):
            return
        embs = [None] * self.num_words_in_train
        with open(pret_file) as f:
            emb_dim = None
            for line in f:
                line = line.strip().split()
                if len(line) > 2:
                    word, vector = line[0], line[1:]
                    word_id = self.word_vocab.get_id(word)
                    if not word_id:
                        if not keep_oov:
                            continue
                        word_id = self.word_vocab.ensure_id(word)
                        embs.append(None)
                    if not emb_dim:
                        emb_dim = len(vector)
                    elif len(vector) != emb_dim:
                        # print(line)
                        continue
                    embs[word_id] = vector
            emb_size = len(vector)
            for idx, emb in enumerate(embs):
                if not emb:
                    embs[idx] = np.zeros(emb_size)
            pret_embs = np.array(embs, dtype=np.float32)
            self.pret_word_embs = pret_embs / np.std(pret_embs)

    def save(self, path):
        pret_word_embs = self.pret_word_embs
        self.pret_word_embs = None
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        self.pret_word_embs = pret_word_embs

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)


class Word(object):
    def __init__(self, word, vocab) -> None:
        super().__init__()
        self.word = word


class Sentence(object):
    def __init__(self, words: list, tags: list) -> None:
        super().__init__()
        self.words = words
        self.tags = tags

    def to_str(self, vocab: Vocabulary):
        return ' '.join(vocab.word(word_id) + '/' + vocab.tag(tag_id) for word_id, tag_id in zip(self.words, self.tags))


class TSVCorpus(object):
    def __init__(self, path: str, vocab: Vocabulary) -> None:
        super().__init__()
        self.vocab = vocab
        self.sentences = []
        with open(path) as src:
            words = []
            tags = []
            for line in src:
                line = line.strip()
                if len(line) == 0:
                    self.sentences.append(Sentence(words, tags))
                    words = []
                    tags = []
                    continue
                cells = line.split()
                assert len(cells) == 2, 'Ill-formatted line: {}'.format(line)
                # each word is represented as (char_ids, word_id)
                words.append(vocab.ensure_word_ids(cells[0]))
                tags.append(vocab.ensure_tag_id(cells[1]))

    def __len__(self):
        return len(self.sentences)

    def __iter__(self):
        return self.sentences.__iter__()

    def __getitem__(self, idx: int) -> Sentence:
        return self.sentences[idx]


if __name__ == '__main__':
    vocab = Vocabulary()
    dev = TSVCorpus('data/pku/bmes/dev.txt', vocab)
    # vocab.add_pret_words('data/character/character-bi.vec', False)
    print(vocab.word_vocab.size())
    print(dev[0].to_str(vocab))
