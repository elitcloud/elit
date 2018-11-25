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
# -*- coding:utf-8 -*-
# Filename: corpus.py
# Author：ported from PyTorch implementation of flair: https://github.com/zalandoresearch/flair to MXNet
# Date: 2018-02-21 14:51
import math
import os
import pickle
import random
import re
from collections import Counter, defaultdict
from enum import Enum
from typing import Sequence, Dict, List, Union

import mxnet as mx
import mxnet.ndarray as nd
import numpy as np

from elit.nlp.dep.common.utils import make_sure_path_exists
from elit.nlp.tagger.mxnet_util import mxnet_prefer_gpu
from elit.structure import Document, NER, POS, SENS
from elit.structure import Sentence as ElitSentence


def read_pretrained_embeddings(filename):
    word_to_embed = {}
    m = 0
    with open(filename, encoding='utf-8') as f:
        for line in f:
            split = line.split()
            if len(split) == 2:
                n, m = int(split[0]), int(split[1])
            if len(split) > 2:
                word = split[0]
                vec = split[1:]
                if m == 0:
                    m = len(vec)
                elif len(vec) != m:  # bad line
                    continue
                word_to_embed[word] = np.array(vec)
    return word_to_embed, m


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

    def __len__(self):
        return len(self.words)


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
    train = int(0.9999 * nline)
    split_every = int(train / 10000)
    split = 0
    dev = int(0.99995 * nline)
    out = None
    with open(text_file) as src:
        for n, line in enumerate(src):
            if n < train:
                if n % split_every == 0 and split < 10000:
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
            if line.strip():  # not blank
                out.write(line)
    out.close()


class Label:
    """
    This class represents a label of a sentence. Each label has a name and optional a confidence value. The confidence
    value needs to be between 0.0 and 1.0. Default value for the confidence is 1.0.
    """

    def __init__(self, name: str, confidence: float = 1.0):
        self.name = name
        self.confidence = confidence

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        if not name:
            raise ValueError('Incorrect label name provided. Label name needs to be set.')
        else:
            self._name = name

    @property
    def confidence(self):
        return self._confidence

    @confidence.setter
    def confidence(self, confidence):
        if 0.0 <= confidence <= 1.0:
            self._confidence = confidence
        else:
            self._confidence = 0.0

    def __str__(self):
        return "{} ({})".format(self._name, self._confidence)

    def __repr__(self):
        return "{} ({})".format(self._name, self._confidence)


class Token:
    """
    This class represents one word in a tokenized sentence. Each token may have any number of tags. It may also point
    to its head in a dependency tree.
    """

    def __init__(self,
                 text: str,
                 idx: int = None,
                 head_id: int = None,
                 whitespace_after: bool = True,
                 pos=None
                 ):
        self.text = text
        self.idx = idx
        self.head_id = head_id
        self.whitespace_after = whitespace_after

        self.sentence = None
        self._embeddings = {}
        self.tags = {}
        if pos:
            self.add_tag('pos', pos)

    def add_tag(self, tag_type: str, tag_value: str):
        self.tags[tag_type] = tag_value

    def get_tag(self, tag_type: str) -> str:
        if tag_type in self.tags:
            return self.tags[tag_type]
        return ''

    def get_head(self):
        return self.sentence.get_token(self.head_id)

    def __str__(self) -> str:
        return 'Token: %d %s' % (self.idx, self.text)

    def __repr__(self) -> str:
        return 'Token: %d %s' % (self.idx, self.text)

    def set_embedding(self, name: str, vector: nd.NDArray):
        assert isinstance(vector, nd.NDArray), 'set_embedding only supports nd.NDArray'
        assert vector.dtype == np.float32, 'dtype mismatch'
        self._embeddings[name] = vector

    def clear_embeddings(self):
        self._embeddings = {}

    def get_embedding(self) -> nd.NDArray:

        if len(self._embeddings) == 1:
            return list(self._embeddings.values())[0]
        embeddings = []
        for embed in sorted(self._embeddings.keys()):
            embeddings.append(self._embeddings[embed])

        if embeddings:
            return nd.concat(*embeddings, dim=0)

        return None

    @property
    def embedding(self):
        return self.get_embedding()


class Sentence:
    def __init__(self, text: str = None, use_tokenizer: bool = False, labels: Union[List[Label], List[str]] = None):
        self.tokens = []

        self.labels = []
        if labels is not None: self.add_labels(labels)

        self._embeddings = {}

        # if text is passed, instantiate sentence with tokens (words)
        if text is not None:

            # tokenize the text first if option selected
            if use_tokenizer:
                assert False, 'no implemented yet'
                # use segtok for tokenization
                # tokens = []
                # sentences = split_single(text)
                # for sentence in sentences:
                #     contractions = split_contractions(word_tokenizer(sentence))
                #     tokens.extend(contractions)
                #
                # # determine offsets for whitespace_after field
                # index = text.index
                # running_offset = 0
                # last_word_offset = -1
                # last_token = None
                # for word in tokens:
                #     token = Token(word)
                #     self.add_token(token)
                #     try:
                #         word_offset = index(word, running_offset)
                #     except:
                #         word_offset = last_word_offset + 1
                #     if word_offset - 1 == last_word_offset and last_token is not None:
                #         last_token.whitespace_after = False
                #     word_len = len(word)
                #     running_offset = word_offset + word_len
                #     last_word_offset = running_offset - 1
                #     last_token = token

            # otherwise assumes whitespace tokenized text
            else:
                # add each word in tokenized string as Token object to Sentence
                for word in text.split(' '):
                    if word:
                        token = Token(word)
                        self.add_token(token)

    def _infer_space_after(self):
        """
        Heuristics in case you wish to infer whitespace_after values for tokenized text. This is useful for some old NLP
        tasks (such as CoNLL-03 and CoNLL-2000) that provide only tokenized data with no info of original whitespacing.
        :return:
        """
        last_token = None
        quote_count = 0
        # infer whitespace after field

        for token in self.tokens:
            if token.text == '"':
                quote_count += 1
                if quote_count % 2 != 0:
                    token.whitespace_after = False
                elif last_token is not None:
                    last_token.whitespace_after = False

            if last_token is not None:

                if token.text in ['.', ':', ',', ';', ')', 'n\'t', '!', '?']:
                    last_token.whitespace_after = False

                if token.text.startswith('\''):
                    last_token.whitespace_after = False

            if token.text in ['(']:
                token.whitespace_after = False

            last_token = token
        return self

    def __getitem__(self, idx: int) -> Token:
        return self.tokens[idx]

    def __iter__(self):
        return iter(self.tokens)

    def add_label(self, label: Union[Label, str]):
        if type(label) is Label:
            self.labels.append(label)

        elif type(label) is str:
            self.labels.append(Label(label))

    def add_labels(self, labels: Union[List[Label], List[str]]):
        for label in labels:
            self.add_label(label)

    def get_label_names(self) -> List[str]:
        return [label.name for label in self.labels]

    def get_token(self, token_id: int) -> Token:
        for token in self.tokens:
            if token.idx == token_id:
                return token

    def add_token(self, token: Token):
        self.tokens.append(token)

        # set token idx if not set
        token.sentence = self
        if token.idx is None:
            token.idx = len(self.tokens)

    def set_embedding(self, name: str, vector):
        self._embeddings[name] = vector.cpu()

    def clear_embeddings(self, also_clear_word_embeddings: bool = True):
        self._embeddings = {}

        if also_clear_word_embeddings:
            for token in self:
                token.clear_embeddings()

    def cpu_embeddings(self):
        for name, vector in self._embeddings.items():
            self._embeddings[name] = vector.cpu()

    def get_embedding(self) -> nd.NDArray:
        embeddings = []
        for embed in sorted(self._embeddings.keys()):
            embedding = self._embeddings[embed]
            embeddings.append(embedding)

        if embeddings:
            return nd.concat(embeddings, dim=0)

        return None

    @property
    def embedding(self):
        return self.get_embedding()

    def to_tagged_string(self) -> str:
        list = []
        for token in self.tokens:
            list.append(token.text)

            tags = []
            for tag_type in token.tags.keys():

                if token.get_tag(tag_type) == '' or token.get_tag(tag_type) == 'O': continue
                tags.append(token.get_tag(tag_type))
            all_tags = '<' + '/'.join(tags) + '>'
            if all_tags != '<>':
                list.append(all_tags)
        return ' '.join(list)

    def convert_tag_scheme(self, tag_type: str = 'ner', target_scheme: str = 'iob', source_scheme='iob'):

        tags = []
        for token in self.tokens:
            token = token
            tags.append(token.get_tag(tag_type))

        if target_scheme == 'iob' and source_scheme == 'iob':
            iob2(tags)

        if target_scheme == 'iobes':
            if source_scheme == 'ioblu':
                tags = ioblu_iobes(tags)
            else:
                iob2(tags)
                tags = iob_iobes(tags)

        for index, tag in enumerate(tags):
            self.tokens[index].add_tag(tag_type, tag)

    def __repr__(self):
        return 'Sentence: "' + ' '.join([t.text for t in self.tokens]) + '" - %d Tokens' % len(self)

    def __copy__(self):
        s = Sentence()
        for token in self.tokens:
            nt = Token(token.text)
            for tag_type in token.tags:
                nt.add_tag(tag_type, token.get_tag(tag_type))

            s.add_token(nt)
        return s

    def __str__(self) -> str:
        return 'Sentence: "' + ' '.join([t.text for t in self.tokens]) + '" - %d Tokens' % len(self)

    def __len__(self) -> int:
        return len(self.tokens)

    def to_tokenized_string(self) -> str:
        return ' '.join([t.text for t in self.tokens])

    def to_plain_string(self):
        plain = ''
        for token in self.tokens:
            plain += token.text
            if token.whitespace_after: plain += ' '
        return plain.rstrip()


def iob2(tags):
    """
    Check that tags have a valid IOB format.
    Tags in IOB1 format are converted to IOB2.
    """
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
    return True


def iob_iobes(tags):
    """
    IOB -> IOBES
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags


def ioblu_iobes(tags):
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'L':
            new_tags.append(tag.replace('L-', 'E-'))
        elif tag.split('-')[0] == 'U':
            new_tags.append(tag.replace('U-', 'S-'))
        elif tag.split('-')[0] in ['I', 'B']:
            new_tags.append(tag)
        else:
            raise Exception('Invalid IOB format!')
    return new_tags


class TaggedCorpus:
    def __init__(self, train: List[Sentence], dev: List[Sentence], test: List[Sentence]):
        self.train = train
        self.dev = dev
        self.test = test

    def downsample(self, percentage: float = 0.1, only_downsample_train=False):

        self.train = self._downsample_to_proportion(self.train, percentage)
        if not only_downsample_train:
            self.dev = self._downsample_to_proportion(self.dev, percentage)
            self.test = self._downsample_to_proportion(self.test, percentage)

        return self

    def clear_embeddings(self):
        for sentence in self.get_all_sentences():
            for token in sentence.tokens:
                token.clear_embeddings()

    def get_all_sentences(self) -> List[Sentence]:
        all_sentences = []
        all_sentences.extend(self.train)
        all_sentences.extend(self.dev)
        all_sentences.extend(self.test)
        return all_sentences

    def make_tag_dictionary(self, tag_type: str) -> Dictionary:

        # Make the tag dictionary
        tag_dictionary = Dictionary()
        if tag_type == 'ner':
            tag_dictionary.add_item('O')
        for sentence in self.get_all_sentences():
            for token in sentence.tokens:
                token = token
                tag_dictionary.add_item(token.get_tag(tag_type))
        tag_dictionary.add_item('<START>')
        tag_dictionary.add_item('<STOP>')
        return tag_dictionary

    def make_label_dictionary(self) -> Dictionary:
        """
        Creates a dictionary of all labels assigned to the sentences in the corpus.
        :return: dictionary of labels
        """

        labels = set(self._get_all_label_names())

        label_dictionary = Dictionary(add_unk=False)
        for label in labels:
            label_dictionary.add_item(label)

        return label_dictionary

    def make_vocab_dictionary(self, max_tokens=-1, min_freq=1) -> Dictionary:
        """
        Creates a dictionary of all tokens contained in the corpus.
        By defining `max_tokens` you can set the maximum number of tokens that should be contained in the dictionary.
        If there are more than `max_tokens` tokens in the corpus, the most frequent tokens are added first.
        If `min_freq` is set the a value greater than 1 only tokens occurring more than `min_freq` times are considered
        to be added to the dictionary.
        :param max_tokens: the maximum number of tokens that should be added to the dictionary (-1 = take all tokens)
        :param min_freq: a token needs to occur at least `min_freq` times to be added to the dictionary (-1 = there is no limitation)
        :return: dictionary of tokens
        """
        tokens = self._get_most_common_tokens(max_tokens, min_freq)

        vocab_dictionary = Dictionary()
        for token in tokens:
            vocab_dictionary.add_item(token)

        return vocab_dictionary

    def _get_most_common_tokens(self, max_tokens, min_freq) -> List[str]:
        tokens_and_frequencies = Counter(self._get_all_tokens())
        tokens_and_frequencies = tokens_and_frequencies.most_common()

        tokens = []
        for token, freq in tokens_and_frequencies:
            if (min_freq != -1 and freq < min_freq) or (max_tokens != -1 and len(tokens) == max_tokens):
                break
            tokens.append(token)
        return tokens

    def _get_all_label_names(self) -> List[str]:
        return [label.name for sent in self.train for label in sent.labels]

    def _get_all_tokens(self) -> List[str]:
        tokens = list(map((lambda s: s.tokens), self.train))
        tokens = [token for sublist in tokens for token in sublist]
        return list(map((lambda t: t.text), tokens))

    def _downsample_to_proportion(self, list: List, proportion: float):

        counter = 0.0
        last_counter = None
        downsampled = []

        for item in list:
            counter += proportion
            if int(counter) != last_counter:
                downsampled.append(item)
                last_counter = int(counter)
        return downsampled

    def print_statistics(self):
        """
        Print statistics about the class distribution (only labels of sentences are taken into account) and sentence
        sizes.
        """

        self._print_statistics_for(self.train, "TRAIN")
        self._print_statistics_for(self.test, "TEST")
        self._print_statistics_for(self.dev, "DEV")

    @staticmethod
    def _print_statistics_for(sentences, name):
        if len(sentences) == 0:
            return

        classes_to_count = TaggedCorpus._get_classes_to_count(sentences)
        tokens_per_sentence = TaggedCorpus._get_tokens_per_sentence(sentences)

        print(name)
        print("total size: " + str(len(sentences)))
        for l, c in classes_to_count.items():
            print("size of class {}: {}".format(l, c))
        print("total # of tokens: " + str(sum(tokens_per_sentence)))
        print("min # of tokens: " + str(min(tokens_per_sentence)))
        print("max # of tokens: " + str(max(tokens_per_sentence)))
        print("avg # of tokens: " + str(sum(tokens_per_sentence) / len(sentences)))

    @staticmethod
    def _get_tokens_per_sentence(sentences):
        return list(map(lambda x: len(x.tokens), sentences))

    @staticmethod
    def _get_classes_to_count(sentences):
        classes_to_count = defaultdict(lambda: 0)
        for sent in sentences:
            for label in sent.labels:
                classes_to_count[label.name] += 1
        return classes_to_count

    def __str__(self) -> str:
        return 'TaggedCorpus: %d train + %d dev + %d test sentences' % (len(self.train), len(self.dev), len(self.test))


class NLPTask(Enum):
    # conll column format
    CONLL_03 = 'conll_03'
    CONLL_2000 = 'conll_2000'
    CONLL_03_GERMAN = 'conll_03-ger'
    ONTONER = 'onto-ner'
    FASHION = 'fashion'
    GERMEVAL = 'germeval'
    SRL = 'srl'
    WSD = 'wsd'

    # conll-u format
    UD_ENGLISH = 'ud_english'
    UD_GERMAN = 'ud_german'
    CONLL_12 = 'conll_12'
    PENN = 'penn'
    ONTONOTES = 'ontonotes'

    # text classification format
    IMDB = 'imdb'
    AG_NEWS = 'ag_news'


class NLPTaskDataFetcher:

    @staticmethod
    def fetch_data(task: NLPTask) -> TaggedCorpus:
        """
        Helper function to fetch a TaggedCorpus for a specific NLPTask. For this to work you need to first download
        and put into the appropriate folder structure the corresponsing NLP task data. The documents on
        https://github.com/zalandoresearch/flair give more info on how to do this. Alternatively, you can use this
        code to create your own data fetchers.
        :param task: specification of the NLPTask you wish to get
        :return: a TaggedCorpus consisting of train, dev and test data
        """

        data_folder = os.path.join('resources', 'tasks', str(task.value).lower())
        print("reading data from {}".format(data_folder))

        # the CoNLL 2000 task on chunking has three columns: text, pos and np (chunk)
        if task == NLPTask.CONLL_2000:
            columns = {0: 'text', 1: 'pos', 2: 'np'}

            return NLPTaskDataFetcher.fetch_column_corpus(data_folder,
                                                          columns,
                                                          train_file='train.txt',
                                                          test_file='test.txt',
                                                          tag_to_biloes='np')

        # many NER tasks follow the CoNLL 03 format with four colulms: text, pos, np and ner tag
        if task == NLPTask.CONLL_03 or task == NLPTask.ONTONER or task == NLPTask.FASHION:
            columns = {0: 'text', 1: 'pos', 2: 'np', 3: 'ner'}

            return NLPTaskDataFetcher.fetch_column_corpus(data_folder,
                                                          columns,
                                                          train_file='eng.train',
                                                          test_file='eng.testb',
                                                          dev_file='eng.testa',
                                                          tag_to_biloes='ner')

        # the CoNLL 03 task for German has an additional lemma column
        if task == NLPTask.CONLL_03_GERMAN:
            columns = {0: 'text', 1: 'lemma', 2: 'pos', 3: 'np', 4: 'ner'}

            return NLPTaskDataFetcher.fetch_column_corpus(data_folder,
                                                          columns,
                                                          train_file='deu.train',
                                                          test_file='deu.testb',
                                                          dev_file='deu.testa',
                                                          tag_to_biloes='ner')

        # the GERMEVAL task only has two columns: text and ner
        if task == NLPTask.GERMEVAL:
            columns = {1: 'text', 2: 'ner'}

            return NLPTaskDataFetcher.fetch_column_corpus(data_folder,
                                                          columns,
                                                          train_file='NER-de-train.tsv',
                                                          test_file='NER-de-test.tsv',
                                                          dev_file='NER-de-dev.tsv',
                                                          tag_to_biloes='ner')

        # WSD tasks may be put into this column format
        if task == NLPTask.WSD:
            columns = {0: 'text', 1: 'lemma', 2: 'pos', 3: 'sense'}

            return NLPTaskDataFetcher.fetch_column_corpus(data_folder,
                                                          columns,
                                                          train_file='semcor.tsv',
                                                          test_file='semeval2015.tsv')

        # the UD corpora follow the CoNLL-U format, for which we have a special reader
        if task == NLPTask.UD_ENGLISH:
            # get train, test and dev data
            sentences_train = NLPTaskDataFetcher.read_conll_ud(
                os.path.join(data_folder, 'en_ewt-ud-train.conllu'))
            sentences_test = NLPTaskDataFetcher.read_conll_ud(
                os.path.join(data_folder, 'en_ewt-ud-test.conllu'))
            sentences_dev = NLPTaskDataFetcher.read_conll_ud(
                os.path.join(data_folder, 'en_ewt-ud-dev.conllu'))

            return TaggedCorpus(sentences_train, sentences_dev, sentences_test)

        if task == NLPTask.UD_GERMAN:
            # get train, test and dev data
            sentences_train = NLPTaskDataFetcher.read_conll_ud(
                os.path.join(data_folder, 'de_gsd-ud-train.conllu'))
            sentences_test = NLPTaskDataFetcher.read_conll_ud(
                os.path.join(data_folder, 'de_gsd-ud-test.conllu'))
            sentences_dev = NLPTaskDataFetcher.read_conll_ud(
                os.path.join(data_folder, 'de_gsd-ud-dev.conllu'))

            return TaggedCorpus(sentences_train, sentences_dev, sentences_test)

        if task == NLPTask.ONTONOTES:
            # get train, test and dev data
            sentences_train = NLPTaskDataFetcher.read_conll_ud(
                os.path.join(data_folder, 'train.conllu'))
            sentences_test = NLPTaskDataFetcher.read_conll_ud(
                os.path.join(data_folder, 'test.conllu'))
            sentences_dev = NLPTaskDataFetcher.read_conll_ud(
                os.path.join(data_folder, 'dev.conllu'))

            return TaggedCorpus(sentences_train, sentences_dev, sentences_test)

        if task == NLPTask.CONLL_12:
            # get train, test and dev data
            sentences_train = NLPTaskDataFetcher.read_conll_ud(
                os.path.join(data_folder, 'train.propbank.conllu'))
            sentences_test = NLPTaskDataFetcher.read_conll_ud(
                os.path.join(data_folder, 'test.propbank.conllu'))
            sentences_dev = NLPTaskDataFetcher.read_conll_ud(
                os.path.join(data_folder, 'dev.propbank.conllu'))
            return TaggedCorpus(sentences_train, sentences_dev, sentences_test)

        if task == NLPTask.PENN:
            sentences_train = NLPTaskDataFetcher.read_conll_ud(
                os.path.join(data_folder, 'train.conll'))
            sentences_dev = NLPTaskDataFetcher.read_conll_ud(
                os.path.join(data_folder, 'valid.conll'))
            sentences_test = NLPTaskDataFetcher.read_conll_ud(
                os.path.join(data_folder, 'test.conll'))

            return TaggedCorpus(sentences_train, sentences_dev, sentences_test)

        # for text classifiers, we use our own special format
        if task == NLPTask.IMDB:
            sentences_train = NLPTaskDataFetcher.read_text_classification_file(
                os.path.join(data_folder, 'train.txt'))
            sentences_dev = NLPTaskDataFetcher.read_text_classification_file(
                os.path.join(data_folder, 'dev.txt'))
            sentences_test = NLPTaskDataFetcher.read_text_classification_file(
                os.path.join(data_folder, 'test.txt'))
            return TaggedCorpus(sentences_train, sentences_dev, sentences_test)

        # for text classifiers, we use our own special format
        if task == NLPTask.AG_NEWS:
            sentences_train = NLPTaskDataFetcher.read_text_classification_file(
                os.path.join(data_folder, 'train.txt'))
            sentences_dev = NLPTaskDataFetcher.read_text_classification_file(
                os.path.join(data_folder, 'dev.txt'))
            sentences_test = NLPTaskDataFetcher.read_text_classification_file(
                os.path.join(data_folder, 'test.txt'))
            return TaggedCorpus(sentences_train, sentences_dev, sentences_test)

    @staticmethod
    def fetch_column_corpus(
            data_folder: str,
            column_format: Dict[int, str],
            train_file: str,
            test_file: str,
            dev_file=None,
            tag_to_biloes=None,
            source_scheme='iob') -> TaggedCorpus:
        """
        Helper function to get a TaggedCorpus from CoNLL column-formatted task data such as CoNLL03 or CoNLL2000.

        :param data_folder: base folder with the task data
        :param column_format: a map specifying the column format
        :param train_file: the name of the train file
        :param test_file: the name of the test file
        :param dev_file: the name of the dev file, if None, dev data is sampled from train
        :param tag_to_biloes: whether to convert to BILOES tagging scheme
        :return: a TaggedCorpus with annotated train, dev and test data
        """

        # get train and test data
        sentences_train = NLPTaskDataFetcher.read_column_data(
            os.path.join(data_folder, train_file), column_format)
        sentences_test = NLPTaskDataFetcher.read_column_data(
            os.path.join(data_folder, test_file), column_format)

        if dev_file is not None:
            sentences_dev = NLPTaskDataFetcher.read_column_data(
                os.path.join(data_folder, dev_file), column_format)
        else:
            # sample dev data from train
            sentences_dev = [sentences_train[i] for i in NLPTaskDataFetcher.__sample()]
            sentences_train = [x for x in sentences_train if x not in sentences_dev]

        if tag_to_biloes is not None:
            longest = 0
            # convert tag scheme to iobes
            for sentence in sentences_train + sentences_test + sentences_dev:
                sentence  = sentence
                sentence.convert_tag_scheme(tag_type=tag_to_biloes, target_scheme='iobes', source_scheme=source_scheme)
                longest = max(longest, len(sentence))
            print('Longest sentence %d' % longest)

        return TaggedCorpus(sentences_train, sentences_dev, sentences_test)

    @staticmethod
    def convert_elit_documents(docs: Sequence[Document]) -> List[Sentence]:
        dataset = []
        for d in docs:
            for s in d.sentences:
                sentence = Sentence()

                for word in s.tokens:
                    t = Token(word)
                    sentence.add_token(t)

                if POS in s:
                    for t, pos in zip(sentence, s.part_of_speech_tags):
                        t.add_tag('pos', pos)

                if NER in s:
                    for t in sentence:
                        t.add_tag('ner', 'O')
                    for start, end, label in s[NER]:
                        if end - start == 0:
                            sentence.tokens[start].tags['ner'] = 'S-' + label
                        else:
                            sentence.tokens[start].tags['ner'] = 'B-' + label
                            for i in range(start + 1, end - 1):
                                sentence.tokens[i].tags['ner'] = 'I-' + label
                            sentence.tokens[end - 1].tags['ner'] = 'E-' + label
                sentence._infer_space_after()
                dataset.append(sentence)
        return dataset

    @staticmethod
    def read_column_data(path_to_column_file: str,
                         column_name_map: Dict[int, str],
                         infer_whitespace_after: bool = True):
        """
        Reads a file in column format and produces a list of Sentence with tokenlevel annotation as specified in the
        column_name_map. For instance, by passing "{0: 'text', 1: 'pos', 2: 'np', 3: 'ner'}" as column_name_map you
        specify that the first column is the text (lexical value) of the token, the second the PoS tag, the third
        the chunk and the forth the NER tag.
        :param path_to_column_file: the path to the column file
        :param column_name_map: a map of column number to token annotation name
        :param infer_whitespace_after: if True, tries to infer whitespace_after field for Token
        :return: list of sentences
        """
        sentences = []

        lines = open(path_to_column_file).read().strip().split('\n')

        # most data sets have the token text in the first column, if not, pass 'text' as column
        text_column = 0
        for column in column_name_map:
            if column_name_map[column] == 'text':
                text_column = column

        sentence = Sentence()
        for line in lines:

            if line.startswith('#'):
                continue

            if line == '':
                if len(sentence) > 0:
                    sentence._infer_space_after()
                    sentences.append(sentence)
                sentence  = Sentence()

            else:
                fields = re.split(r"\s+", line)
                token = Token(fields[text_column])
                for column in column_name_map:
                    if len(fields) > column:
                        if column != text_column:
                            token.add_tag(column_name_map[column], fields[column])
                sentence.add_token(token)

        if len(sentence.tokens) > 0:
            sentence._infer_space_after()
            sentences.append(sentence)

        return sentences

    @staticmethod
    def read_conll_ud(path_to_conll_file: str) -> List[Sentence]:
        """
       Reads a file in CoNLL-U format and produces a list of Sentence with full morphosyntactic annotation
       :param path_to_conll_file: the path to the conll-u file
       :return: list of sentences
       """
        sentences = []

        lines = open(path_to_conll_file, encoding='utf-8'). \
            read().strip().split('\n')

        sentence  = Sentence()
        for line in lines:

            fields = re.split(r"\s+", line)
            if line == '':
                if len(sentence) > 0:
                    sentences.append(sentence)
                sentence  = Sentence()

            elif line.startswith('#'):
                continue
            elif '.' in fields[0]:
                continue
            elif '-' in fields[0]:
                continue
            else:
                token = Token(fields[1], head_id=int(fields[6]))
                token.add_tag('lemma', str(fields[2]))
                token.add_tag('upos', str(fields[3]))
                token.add_tag('pos', str(fields[4]))
                token.add_tag('dependency', str(fields[7]))

                for morph in str(fields[5]).split('|'):
                    if not "=" in morph: continue;
                    token.add_tag(morph.split('=')[0].lower(), morph.split('=')[1])

                if len(fields) > 10 and str(fields[10]) == 'Y':
                    token.add_tag('frame', str(fields[11]))

                sentence.add_token(token)

        if len(sentence.tokens) > 0: sentences.append(sentence)

        return sentences

    @staticmethod
    def read_text_classification_file(path_to_file):
        """
        Reads a data file for text classification. The file should contain one document/text per line.
        The line should have the following format:
        __label__<class_name> <text>
        If you have a multi class task, you can have as many labels as you want at the beginning of the line, e.g.,
        __label__<class_name_1> __label__<class_name_2> <text>
        :param path_to_file: the path to the data file
        :return: list of sentences
        """
        label_prefix = '__label__'
        sentences = []

        with open(path_to_file) as f:
            lines = f.readlines()

            for line in lines:
                words = line.split()

                labels = []
                l_len = 0

                for i in range(len(words)):
                    if words[i].startswith(label_prefix):
                        l_len += len(words[i]) + 1
                        label = words[i].replace(label_prefix, "")
                        labels.append(label)
                    else:
                        break

                text = line[l_len:].strip()

                if text and labels:
                    sentences.append(Sentence(text, labels=labels, use_tokenizer=True))

        return sentences

    @staticmethod
    def __sample():
        sample = [7199, 2012, 7426, 1374, 2590, 4401, 7659, 2441, 4209, 6997, 6907, 4789, 3292, 4874, 7836, 2065, 1804,
                  2409,
                  6353, 86, 1412, 5431, 3275, 7696, 3325, 7678, 6888, 5326, 5782, 3739, 4972, 6350, 7167, 6558, 918,
                  6444,
                  5368, 731, 244, 2029, 6200, 5088, 4688, 2580, 2153, 5477, 714, 1570, 6651, 5724, 4090, 167, 1689,
                  6166,
                  7304, 3705, 256, 5689, 6282, 707, 5390, 1367, 4167, 16, 6554, 5093, 3944, 5008, 3510, 1741, 1, 4464,
                  173,
                  5362, 6827, 35, 1662, 3136, 1516, 3826, 1575, 6771, 5965, 1449, 7806, 632, 5870, 3566, 1434, 2361,
                  6348,
                  5140, 7765, 4800, 6541, 7910, 2021, 1041, 3171, 2137, 495, 2249, 7334, 4806, 844, 3848, 7396, 3861,
                  1337,
                  430, 1325, 36, 2289, 720, 4182, 3955, 3451, 192, 3715, 3144, 1140, 2874, 6728, 4877, 1876, 2551, 2910,
                  260,
                  7767, 7206, 5577, 6707, 3392, 1830, 842, 5264, 4042, 3572, 331, 6995, 2307, 5664, 2878, 1115, 1880,
                  1548,
                  3740, 860, 1799, 2099, 7359, 4648, 2264, 1018, 5417, 3052, 2480, 2256, 6672, 6647, 1272, 1986, 7063,
                  4071,
                  3199, 3652, 1797, 1693, 2008, 4138, 7428, 3083, 1494, 4911, 728, 1556, 7651, 2535, 2160, 4014, 1438,
                  6148,
                  551, 476, 4198, 3835, 1489, 6404, 7346, 1178, 607, 7693, 4146, 6655, 4355, 1571, 522, 5835, 622, 1267,
                  6778, 5236, 5211, 5039, 3836, 1751, 1019, 6952, 7610, 7677, 4224, 1485, 4101, 5793, 6708, 5741, 4630,
                  5857,
                  6959, 847, 4375, 3458, 4936, 6887, 5, 3150, 5551, 4840, 2618, 7456, 7600, 5995, 5270, 5496, 4316,
                  1479,
                  517, 2940, 2337, 7461, 3296, 4133, 491, 6408, 7609, 4290, 5028, 7471, 6337, 488, 5033, 5967, 1209,
                  5511,
                  5449, 3837, 4760, 4490, 6550, 2676, 371, 3962, 4507, 5268, 4285, 5257, 859, 14, 4487, 5669, 6594,
                  6544,
                  7427, 5624, 4882, 7425, 2378, 1498, 931, 7253, 2638, 2897, 5670, 6463, 5300, 6802, 4229, 7076, 6848,
                  6414,
                  1465, 7243, 989, 7204, 1926, 1255, 1794, 2115, 3975, 6987, 3166, 105, 3856, 3272, 3977, 4097, 2612,
                  2869,
                  6022, 153, 3357, 2439, 6491, 766, 3840, 2683, 5074, 159, 5407, 3029, 4815, 1782, 4970, 6250, 5377,
                  6473,
                  5151, 4687, 798, 5214, 3364, 6412, 7125, 3495, 2385, 4476, 863, 5493, 5830, 938, 2979, 7808, 4830,
                  4180,
                  1565, 4818, 702, 1442, 4673, 6920, 2089, 1930, 2036, 1436, 6632, 1006, 5256, 5666, 6401, 3415, 4693,
                  5890,
                  7124, 3853, 884, 4650, 4550, 7406, 3394, 6715, 6754, 3932, 599, 1816, 3273, 5016, 2918, 526, 6883,
                  3089,
                  64, 1305, 7442, 6837, 783, 4536, 100, 4951, 2933, 3750, 3232, 7150, 1934, 3576, 2900, 7883, 964, 4025,
                  28,
                  1732, 382, 166, 6053, 6320, 2058, 652, 3182, 6836, 4547, 419, 1600, 6891, 6235, 7208, 7190, 7144,
                  3133,
                  4775, 4892, 895, 4428, 7929, 7297, 7773, 5325, 2799, 5645, 1192, 1672, 2540, 6812, 5441, 2681, 342,
                  333,
                  2161, 593, 5463, 1568, 5252, 4194, 2280, 2423, 2118, 7455, 4553, 5960, 3163, 7147, 4305, 5599, 2775,
                  5334,
                  4727, 6926, 2189, 7778, 7245, 2066, 1259, 2074, 7866, 7403, 4642, 5490, 3563, 6923, 3934, 5728, 5425,
                  2369,
                  375, 3578, 2732, 2675, 6167, 6726, 4211, 2241, 4585, 4272, 882, 1821, 3904, 6864, 5723, 4708, 3226,
                  7151,
                  3911, 4274, 4945, 3719, 7467, 7712, 5068, 7181, 745, 2846, 2695, 3707, 1076, 1077, 2698, 5699, 1040,
                  6338,
                  631, 1609, 896, 3607, 6801, 3593, 1698, 91, 639, 2826, 2937, 493, 4218, 5958, 2765, 4926, 4546, 7400,
                  1909,
                  5693, 1871, 1687, 6589, 4334, 2748, 7129, 3332, 42, 345, 709, 4685, 6624, 377, 3204, 2603, 7183, 6123,
                  4249, 1531, 7, 703, 6978, 2856, 7871, 7290, 369, 582, 4704, 4979, 66, 1139, 87, 5166, 967, 2727, 5920,
                  6806, 5997, 1301, 5826, 1805, 4347, 4870, 4213, 4254, 504, 3865, 189, 6393, 7281, 2907, 656, 6617,
                  1807,
                  6258, 3605, 1009, 3694, 3004, 2870, 7710, 2608, 400, 7635, 4392, 3055, 942, 2952, 3441, 902, 5892,
                  574,
                  5418, 6212, 1602, 5619, 7094, 1168, 3877, 3888, 1618, 6564, 455, 4581, 3258, 2606, 4643, 2454, 2763,
                  5332,
                  6158, 940, 2343, 7902, 3438, 6117, 2198, 3842, 4773, 1492, 2424, 7662, 6559, 1196, 3203, 5286, 6764,
                  3829,
                  4746, 1117, 2120, 1378, 5614, 4871, 4024, 5489, 3312, 1094, 1838, 3964, 3151, 4545, 5795, 1739, 4920,
                  5690,
                  2570, 3530, 2751, 1426, 2631, 88, 7728, 3741, 5654, 3157, 5557, 6668, 7309, 7313, 807, 4376, 4512,
                  6786,
                  7898, 2429, 3890, 2418, 2243, 2330, 4561, 6119, 2864, 5570, 2485, 5499, 4983, 6257, 3692, 1563, 1939,
                  126,
                  3299, 2811, 7933, 465, 5976, 3712, 4478, 7671, 3143, 1947, 6133, 1928, 5725, 5747, 1107, 163, 3610,
                  3723,
                  1496, 7477, 53, 6548, 5548, 4357, 4963, 5896, 5361, 7295, 7632, 3559, 6740, 6312, 6890, 3303, 625,
                  7681,
                  7174, 6928, 1088, 2133, 4276, 5299, 4488, 5354, 3044, 3321, 409, 6218, 2255, 829, 2129, 673, 1588,
                  6824,
                  1297, 6996, 4324, 7423, 5209, 7617, 3041, 78, 5518, 5392, 4967, 3704, 497, 858, 1833, 5108, 6095,
                  6039,
                  6705, 5561, 5888, 3883, 1048, 1119, 1292, 5639, 4358, 2487, 1235, 125, 4453, 3035, 3304, 6938, 2670,
                  4322,
                  648, 1785, 6114, 6056, 1515, 4628, 5036, 37, 1226, 6081, 4473, 953, 5009, 217, 5952, 755, 2604, 3060,
                  3322,
                  6087, 604, 2260, 7897, 3129, 616, 1593, 69, 230, 1526, 6349, 6452, 4235, 1752, 4288, 6377, 1229, 395,
                  4326,
                  5845, 5314, 1542, 6483, 2844, 7088, 4702, 3300, 97, 7817, 6804, 471, 3624, 3773, 7057, 2391, 22, 3293,
                  6619, 1933, 6871, 164, 7796, 6744, 1589, 1802, 2880, 7093, 906, 389, 7892, 976, 848, 4076, 7818, 5556,
                  3507, 4740, 4359, 7105, 2938, 683, 4292, 1849, 3121, 5618, 4407, 2883, 7502, 5922, 6130, 301, 4370,
                  7019,
                  3009, 425, 2601, 3592, 790, 2656, 5455, 257, 1500, 3544, 818, 2221, 3313, 3426, 5915, 7155, 3110,
                  4425,
                  5255, 2140, 5632, 614, 1663, 1787, 4023, 1734, 4528, 3318, 4099, 5383, 3999, 722, 3866, 1401, 1299,
                  2926,
                  1360, 1916, 3259, 2420, 1409, 2817, 5961, 782, 1636, 4168, 1344, 4327, 7780, 7335, 3017, 6582, 4623,
                  7198,
                  2499, 2139, 3821, 4822, 2552, 4904, 4328, 6666, 4389, 3687, 1014, 7829, 4802, 5149, 4199, 1866, 1992,
                  2893,
                  6957, 3099, 1212, 672, 4616, 758, 6421, 2281, 6528, 3148, 4197, 1317, 4258, 1407, 6618, 2562, 4448,
                  6137,
                  6151, 1817, 3278, 3982, 5144, 3311, 3453, 1722, 4912, 3641, 5560, 2234, 6645, 3084, 4890, 557, 1455,
                  4152,
                  5784, 7221, 3078, 6961, 23, 4281, 6012, 156, 5109, 6984, 6140, 6730, 4965, 7123, 85, 2912, 5192, 1425,
                  1993, 4056, 598]
        return sample


def get_chunks(seq: List[str]):
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == 'O' and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_start, i, chunk_type)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != 'O':
            tok_chunk_class, tok_chunk_type = tok.split('-')
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_start, i, chunk_type)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass

    # end condition
    if chunk_type is not None:
        chunk = (chunk_start, len(seq), chunk_type)
        chunks.append(chunk)

    return chunks


def conll_to_documents(path, headers={0: 'text', 1: 'pos', 2: 'np', 3: 'ner'}) -> List[Document]:
    sents = NLPTaskDataFetcher.read_column_data(path, headers)
    elit_sents = []
    has_ner = 'ner' in headers.values()
    for s in sents:
        sent = ElitSentence()
        sent[POS] = []
        for t in s.tokens:
            sent.tokens.append(t.text)
            sent[POS].append(t.tags['pos'])
        if has_ner:
            ner_tags = [t.tags['ner'] for t in s.tokens]
            sent[NER] = get_chunks(ner_tags)
        elit_sents.append(sent)
    return [Document({SENS: elit_sents})]


if __name__ == '__main__':
    make_language_model_dataset('data/raw/jumbo.txt', 'data/raw')
    # use your own data path
    # dataset = conll_to_documents('data/conll-03/debug/eng.dev')
    # corpus = NLPTaskDataFetcher.convert_elit_documents(dataset)
    # w2v = read_pretrained_embeddings('data/embedding/glove/glove.6B.100d.debug.txt')
    pass