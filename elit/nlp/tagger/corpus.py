# -*- coding:utf-8 -*-
# Filename: corpus.py
# Author：hankcs
# Date: 2018-02-21 14:51
import math
import random

import numpy as np
import pickle
import os
from typing import Sequence, Tuple, Dict, List
import mxnet as mx
import mxnet.ndarray as nd
from elit.nlp.dep.common.utils import make_sure_path_exists
from elit.nlp.tagger.mxnet_util import mxnet_prefer_gpu


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


class Dictionary:
    """
    This class holds a dictionary that maps strings to IDs, used to generate one-hot encodings of strings.
    """

    def __init__(self, add_unk=True):
        # init dictionaries
        self.item2idx = {}  # Dict[str, int]
        self.idx2item = []  # List[str]

        # in order to deal with unknown tokens, add <unk>
        if add_unk:
            self.add_item('<unk>')

    def add_item(self, item: str) -> int:
        """
        add string - if already in dictionary returns its ID. if not in dictionary, it will get a new ID.
        :param item: a string for which to assign an id
        :return: ID of string
        """
        # item = item.encode('utf-8')
        if item not in self.item2idx:
            self.idx2item.append(item)
            self.item2idx[item] = len(self.idx2item) - 1
        return self.item2idx[item]

    def get_idx_for_item(self, item: str) -> int:
        """
        returns the ID of the string, otherwise 0
        :param item: string for which ID is requested
        :return: ID of string, otherwise 0
        """
        # item = item.encode('utf-8')
        return self.item2idx.get(item, 0)

    def get_items(self) -> List[str]:
        items = []
        for item in self.idx2item:
            items.append(item)
        return items

    def __len__(self) -> int:
        return len(self.idx2item)

    def get_item_for_index(self, idx):
        return self.idx2item[idx]

    def save(self, savefile):
        import pickle
        with open(savefile, 'wb') as f:
            mappings = {
                'idx2item': self.idx2item,
                'item2idx': self.item2idx
            }
            pickle.dump(mappings, f)

    @classmethod
    def load_from_file(cls, filename: str):
        import pickle
        dictionary = Dictionary()  # Dictionary
        with open(filename, 'rb') as f:
            mappings = pickle.load(f, encoding='latin1')
            idx2item = mappings['idx2item']
            item2idx = mappings['item2idx']
            dictionary.item2idx = item2idx
            dictionary.idx2item = idx2item
        return dictionary

    @classmethod
    def load(cls, name: str):

        return Dictionary.load_from_file(name)

    @classmethod
    def create(cls, path: str, char=True):
        print('Creating dictionary from {}'.format(path))
        d = Dictionary()
        files = []
        if os.path.isdir(path):
            files = [os.path.join(path, f) for f in os.listdir(path) if
                     os.path.isfile(os.path.join(path, f)) and f.startswith('train_split')]
        elif os.path.isfile(path):
            files = [path]
        if len(files) == 0:
            raise IOError('No file found in {}'.format(path))
        for n, f in enumerate(files):
            print('{}/{}'.format(n + 1, len(files)))
            with open(f) as src:
                for line in src:
                    for c in line if char else line.split():
                        d.add_item(c)
        return d


class TextCorpus(object):
    def __init__(self, path, dictionary: Dictionary = None, forward: bool = True, character_level: bool = True):

        self.forward = forward
        self.split_on_char = character_level
        self.train_path = os.path.join(path, 'train')

        self.train_files = sorted(
            [f for f in os.listdir(self.train_path) if
             os.path.isfile(os.path.join(self.train_path, f)) and f.startswith('train_split')])

        if dictionary is None:
            dictionary = Dictionary.create(self.train_path)

        self.dictionary = dictionary  # Dictionary

        self.current_train_file_index = len(self.train_files)

        with mx.Context(mxnet_prefer_gpu()):
            self.valid = self.charsplit(os.path.join(path, 'valid.txt'),
                                        forward=forward,
                                        split_on_char=self.split_on_char)

            self.test = self.charsplit(os.path.join(path, 'test.txt'),
                                       forward=forward,
                                       split_on_char=self.split_on_char)

    @property
    def is_last_slice(self) -> bool:
        if self.current_train_file_index >= len(self.train_files) - 1:
            return True
        return False

    def get_next_train_slice(self):

        self.current_train_file_index += 1

        if self.current_train_file_index >= len(self.train_files):
            self.current_train_file_index = 0
            random.shuffle(self.train_files)

        current_train_file = self.train_files[self.current_train_file_index]

        train_slice = self.charsplit(os.path.join(self.train_path, current_train_file),
                                     expand_vocab=False,
                                     forward=self.forward,
                                     split_on_char=self.split_on_char)

        return train_slice

    def charsplit(self, path: str, expand_vocab=False, forward=True, split_on_char=True) -> nd.NDArray:

        """Tokenizes a text file on character basis."""
        assert os.path.exists(path)
        # print('loading {}'.format(path))

        #
        with open(path, 'r', encoding="utf-8") as f:
            tokens = 0
            for line in f:

                if split_on_char:
                    chars = list(line)
                else:
                    chars = line.split()

                # print(chars)
                tokens += len(chars)

                # Add chars to the dictionary
                if expand_vocab:
                    for char in chars:
                        self.dictionary.add_item(char)
                # if tokens % 1000000:
                #     print('\r{}m tokens'.format(tokens // 1000000), end='')

        # print('\nconverting to tensor...')

        def percent(current, total):
            log_every = math.ceil(total / 10000)
            if current % log_every == 0:
                print('\r%.2f%%' % (current / total) * 100, end='')
            elif total - current < 2:
                print('\r100%   ')

        if forward:
            # charsplit file content
            with open(path, 'r', encoding="utf-8") as f:
                token = 0
                id_list = [None] * tokens
                for line in f:
                    line = self.random_casechange(line)

                    if split_on_char:
                        chars = list(line)
                    else:
                        chars = line.split()

                    for char in chars:
                        if token >= tokens: break
                        id_list[token] = self.dictionary.get_idx_for_item(char)
                        token += 1
                    # percent(token, tokens)
                ids = nd.array(id_list)
        else:
            # charsplit file content
            with open(path, 'r', encoding="utf-8") as f:
                id_list = [None] * tokens
                token = tokens - 1
                for line in f:
                    line = self.random_casechange(line)

                    if split_on_char:
                        chars = list(line)
                    else:
                        chars = line.split()

                    for char in chars:
                        if token >= tokens: break
                        id_list[token] = self.dictionary.get_idx_for_item(char)
                        token -= 1
                    # percent(token, tokens)

        return ids

    @staticmethod
    def random_casechange(line: str) -> str:
        no = random.randint(0, 99)
        if no is 0:
            line = line.lower()
        if no is 1:
            line = line.upper()
        return line

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = nd.zeros((tokens), dtype='int')
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids


def make_language_model_dataset(text_file, output_folder):
    nline = 0
    with open(text_file) as src:
        for line in src:
            nline += 1
    print('{} lines in {}'.format(nline, text_file))
    make_sure_path_exists(os.path.join(output_folder, 'train'))
    train = int(0.8 * nline)
    split_every = int(train / 10)
    split = 0
    dev = int(0.9 * nline)
    out = None
    with open(text_file) as src:
        for n, line in enumerate(src):
            if n < train:
                if n % split_every == 0:
                    if out:
                        out.close()
                    split += 1
                    out = open(os.path.join(output_folder, 'train', 'train_split_{}'.format(split)), 'w')
            elif n == train:
                out.close()
                out = open(os.path.join(output_folder, 'valid.txt'), 'w')
            elif n == dev:
                out.close()
                out = open(os.path.join(output_folder, 'test.txt'), 'w')
            out.write(line)
    out.close()


if __name__ == '__main__':
    make_language_model_dataset('data/wiki/test.txt', 'data/wiki-debug')
