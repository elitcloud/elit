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
import abc
import inspect
import os
import re
from typing import List, Tuple

from pkg_resources import resource_filename

from elit.component import Component
from elit.structure import TOK, OFF, Document, Sentence, SID
from elit.util.io import read_word_set, read_concat_word_dict
from elit.util.string import *

__author__ = "Jinho D. Choi, Gary Lai"


class Tokenizer(Component):
    def decode(self, input_text: str, init_offset: int = 0, segment: int = 2, **kwargs) -> Document:
        """
        :param input_text: the input text.
        :param init_offset: the initial offset of the first token.
        :param segment: 0 - no segmentation, 1 - newline segmentation, 2 - rule-based segmentation, 3 - newline and rule-based segmentation.
        :return: the dictionary contains ('tok' = list of tokens) and ('off' = list of offsets);
                 see the comments for :meth:`Tokenizer.offsets` for more details about the offsets.
        """
        document = Document()

        if segment == 0:
            tokens, offsets = self.tokenize(input_text, init_offset)
            document.add_sentence(Sentence({TOK: tokens, OFF: offsets}))
        elif segment == 2:
            tokens, offsets = self.tokenize(input_text, init_offset)
            document.add_sentences(self.segment(tokens, offsets))
        elif segment == 1 or segment == 3:
            indices = [i for i, c in enumerate(input_text) if i == 0 or (c == '\n' and input_text[i - 1] != '\n')]
            indices.append(len(input_text))

            for i in range(1, len(indices)):
                bidx = indices[i - 1]
                eidx = indices[i]
                tokens, offsets = self.tokenize(input_text[bidx:eidx], bidx+init_offset)
                if tokens:
                    if segment == 1: document.add_sentence(Sentence({TOK: tokens, OFF: offsets}))
                    else: document.add_sentences(self.segment(tokens, offsets))

        for i, sentence in enumerate(document.sentences): sentence[SID] = i
        return document

    @abc.abstractmethod
    def tokenize(self, input_text: str, init_offset: int = 0) -> Tuple[List[str], List[Tuple[int, int]]]:
        """
        :param input_text: the input text.
        :param init_offset: the initial offset of the first token.
        :return: the dictionary contains ('tok' = list of tokens) and ('off' = list of offsets);
                 see the comments for :meth:`Tokenizer.offsets` for more details about the offsets.
        """
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))

    @classmethod
    def get_offsets(cls, input_text: str, tokens: List[str], init_offset=0) -> List[Tuple[int, int]]:
        """
        :param input_text: the input text.
        :param tokens: the list of tokens split from the input text.
        :param init_offset: the initial offset of the first token.
        :return: the list of (begin, end) offsets, where the begin (inclusive) and the end (exclusive) offsets indicate
        the caret positions of the first and the last characters of the corresponding token, respectively.
        e.g., text = 'Hello, world!', tokens = ['Hello', ',', 'world', '!'] -> [(0, 5), (5, 6), (7, 12), (12, 13)]
        """

        def get_offset(token):
            nonlocal end
            begin = input_text.index(token, end)
            end = begin + len(token)
            return begin + init_offset, end + init_offset

        end = 0
        return [get_offset(token) for token in tokens]

    @classmethod
    def segment(cls, tokens: List[str], offsets: List[Tuple[int, int]]) -> List[Sentence]:
        """
        :param tokens: the list of input tokens.
        :param offsets: the list of offsets, where each offset is a tuple of (begin, end).
        :return: the list of sentences segmented from the input text.
        """

        def sentence(bidx: int, eidx: int) -> Sentence:
            return Sentence({TOK: tokens[bidx:eidx], OFF: offsets[bidx:eidx]})

        right_quote = True
        sentences = []
        begin = 0

        for i, token in enumerate(tokens):
            t = token[0]
            if t == '"': right_quote = not right_quote

            if begin == i:
                if sentences and (is_right_bracket(t) or t == u'\u201D' or t == '"' and right_quote):
                    d = sentences[-1]
                    d[TOK].append(token)
                    d[OFF].append(offsets[i])
                    begin = i + 1
            elif all(is_final_mark(c) for c in token):
                sentences.append(sentence(begin, i + 1))
                begin = i + 1

        if begin < len(tokens):
            sentences.append(sentence(begin, len(tokens)))

        return sentences


class SpaceTokenizer(Tokenizer):
    """
    :class:`SpaceTokenizer` splits tokens by white-spaces.
    """

    def __init__(self):
        super(SpaceTokenizer, self).__init__()

    def save(self, model_path: str, **kwargs):
        """ Not supported. """
        pass

    def decode(self, input_text: str, init_offset: int = 0, segment: int = 1, **kwargs) -> Document:
        return super().decode(input_text, init_offset, segment, **kwargs)

    def train(self, trn_data, dev_data, model_path: str, **kwargs) -> float:
        """ Not supported. """
        pass

    def evaluate(self, data, **kwargs):
        """ Not supported. """
        pass

    def save(self, model_path: str, **kwargs):
        """ Not supported. """
        pass

    def decode(self, input_text: str, init_offset: int = 0, segment: int = 1, **kwargs) -> Document:
        return super().decode(input_text, init_offset, segment, **kwargs)

    def train(self, trn_data, dev_data, model_path: str, **kwargs) -> float:
        """ Not supported. """
        pass

    def evaluate(self, data, **kwargs):
        """ Not supported. """
        pass

    def load(self, model_path: str, **kwargs):
        """ Not supported. """
        pass

    def tokenize(self, input_text: str, init_offset: int = 0) -> Tuple[List[str], List[Tuple[int, int]]]:
        """
        :param input_text: the input text.
        :param init_offset: the initial offset of the first token.
        :return: the dictionary contains ('tok' = list of tokens) and ('off' = list of offsets);
                 see the comments for :meth:`Tokenizer.offsets` for more details about the offsets.
        """
        tokens = input_text.split()
        return tokens, self.get_offsets(input_text, tokens, init_offset)


class EnglishTokenizer(Tokenizer):
    def __init__(self):
        """
        :class:`EnglishTokenizer` splits the input text into linguistic tokens.
        """
        super(EnglishTokenizer, self).__init__()

        # _inflection_lexicons
        resource_root = 'elit.resources.tokenizer'

        self.ABBREVIATION_PERIOD = read_word_set(resource_filename(
            resource_root, 'english_abbreviation_period.txt'))
        self.APOSTROPHE_FRONT = read_word_set(resource_filename(
            resource_root, 'english_apostrophe_front.txt'))
        self.MAP_CONCAT_WORD = read_concat_word_dict(resource_filename(
            resource_root, 'english_concat_words.txt'))
        self.HYPHEN_PREFIX = read_word_set(resource_filename(
            resource_root, 'english_hyphen_prefix.txt'))
        self.HYPHEN_SUFFIX = read_word_set(resource_filename(
            resource_root, 'english_hyphen_suffix.txt'))

        # regular expressions
        self.RE_NETWORK_PROTOCOL = re.compile(
            r'((http|https|ftp|sftp|ssh|ssl|telnet|smtp|pop3|imap|imap4|sip)(://))')
        """
        :abc:
        <3 </3 <\3
        (: ): \\: *: $: (-: (^: (= (;
        :) :( =) B) 8) :-) :^) :3 :D :p :| :(( :---)
        """
        self.RE_EMOTICON = re.compile(
            r'(:\w+:|<[\\/]?3|[()\\|*$][-^]?[:=;]|[:=;B8]([-^]+)?[3DOPp@$*()\\/|]+)(\W|$)')
        """
        jinho@elit.cloud
        jinho.choi@elit.cloud
        choi@demo.elit.cloud
        jinho:choi@127.0.0.1
        """
        self.RE_EMAIL = re.compile(
            r'[\w\-.]+(:\S+)?@(([A-Za-z0-9\-]+\.)+[A-Za-z]{2,12}|\d{1,3}(\.\d{1,3}){3})')
        """
        &arrow;
        &#123; &#x123; &#X123;
        """
        self.RE_HTML_ENTITY = re.compile(r'&([A-Za-z]+|#[Xx]?\d+);')
        """
        [1] (1a) {A} <a1> [***] [A.a] [A.1] [1.a] ((---))
        """
        self.RE_LIST_ITEM = re.compile(
            r'(([\[({<]+)(\d+[A-Za-z]?|[A-Za-z]\d*|\W+)(\.(\d+|[A-Za-z]))*([\])\}>])+)')
        """
        don't don’t I'll HE'S
        """
        self.RE_APOSTROPHE = re.compile(
            r'(?i)[a-z](n[\'\u2019]t|[\'\u2019](ll|nt|re|ve|[dmstz]))(\W|$)')
        """
        a.b.c 1-2-3
        """
        self.RE_ABBREVIATION = re.compile(r'[A-Za-z0-9]([.-][A-Za-z0-9])*$')
        """
        10kg 1cm
        """
        self.RE_UNIT = re.compile(
            r'(?i)(\d)([acdfkmnpyz]?[mg]|[ap]\.m|ch|cwt|d|drc|ft|fur|gr|h|in|lb|lea|mi|ms|oz|pg|qtr|yd)$')
        """
        hello.World
        """
        self.RE_FINAL_MARK_IN_BETWEEN = re.compile(
            r'([A-Za-z]{3,})([.?!]+)([A-Za-z]{3,})$')

    def save(self, model_path: str, **kwargs):
        """ Not supported. """
        pass

    def train(self, trn_data, dev_data, model_path: str, **kwargs) -> float:
        """ Not supported. """
        pass

    def evaluate(self, data, **kwargs):
        """ Not supported. """
        pass

    def load(self, model_path, *args, **kwargs):
        self.ABBREVIATION_PERIOD = read_word_set(
            os.path.join(model_path, 'english_abbreviation_period.txt'))
        self.APOSTROPHE_FRONT = read_word_set(
            os.path.join(model_path, 'english_apostrophe_front.txt'))
        self.MAP_CONCAT_WORD = read_concat_word_dict(
            os.path.join(model_path, 'english_concat_words.txt'))
        self.HYPHEN_PREFIX = read_word_set(
            os.path.join(model_path, 'english_hyphen_prefix.txt'))
        self.HYPHEN_SUFFIX = read_word_set(
            os.path.join(model_path, 'english_hyphen_suffix.txt'))

    def tokenize(self, input_text: str, init_offset: int = 0) -> Tuple[List[str], List[Tuple[int, int]]]:
        """
        :param input_text: the input text.
        :param init_offset: the initial offset of the first token.
        :return: the dictionary contains ('tok' = list of tokens) and ('off' = list of offsets);
                 see the comments for :meth:`Tokenizer.offsets` for more details about the offsets.
        """
        tokens = []
        offsets = []

        # no valid token in the input text
        if not input_text or input_text.isspace():
            return tokens, offsets

        # skip beginning and ending spaces
        begin = next(i for i, c in enumerate(input_text) if not c.isspace())
        last = len(input_text) - next(i for i, c in enumerate(reversed(input_text)) if not c.isspace())

        # search for in-between spaces
        for end, c in enumerate(input_text[begin + 1:last], begin + 1):
            if c.isspace():
                self.tokenize_aux(tokens, offsets, input_text, begin, end, init_offset)
                begin = end + 1

        self.tokenize_aux(tokens, offsets, input_text, begin, last, init_offset)
        return tokens, offsets

    def tokenize_aux(self, tokens, offsets, text, begin, end, offset):
        if begin >= end or end > len(text): return False
        token = text[begin:end]

        # handle special cases
        if self.tokenize_trivial(tokens, offsets, token, begin, end, offset):
            return True
        if self.tokenize_regex(tokens, offsets, text, begin, end, offset, token):
            return True
        if self.tokenize_symbol(tokens, offsets, text, begin, end, offset, token):
            return True

        # add the token as it is
        self.add_token(tokens, offsets, token, begin, end, offset)
        return True

    def tokenize_trivial(self, tokens, offsets, token, begin, end, offset):
        if end - begin == 1 or token.isalnum():
            self.add_token(tokens, offsets, token, begin, end, offset)
            return True

        return False

    def tokenize_regex(self, tokens, offsets, text, begin, end, offset, token):
        def group(regex, gid=0):
            m = regex.search(token)
            if m:
                idx = begin + m.start(gid)
                lst = begin + m.end(gid)

                self.tokenize_aux(tokens, offsets, text, begin, idx, offset)
                self.add_token(tokens, offsets, m.group(gid), idx, lst, offset)
                self.tokenize_aux(tokens, offsets, text, lst, end, offset)
                return True

            return False

        def hyperlink():
            m = self.RE_NETWORK_PROTOCOL.search(token)
            if m:
                if m.start() > 0:
                    idx = begin + m.start()
                    self.tokenize_aux(tokens, offsets, text, begin, idx, offset)
                    self.add_token(tokens, offsets, token[m.start():], idx, end, offset)
                else:
                    self.add_token(tokens, offsets, token, begin, end, offset)
                return True

            return False

        # split by regular expressions
        if group(self.RE_HTML_ENTITY):
            return True
        if group(self.RE_EMAIL):
            return True
        if hyperlink():
            return True
        if group(self.RE_EMOTICON, 1):
            return True
        if group(self.RE_LIST_ITEM):
            return True
        if group(self.RE_APOSTROPHE, 1):
            return True
        return False

    def tokenize_symbol(self, tokens, offsets, text, begin, end, offset, token):
        def index_last_sequence(i, c):
            final_mark = is_final_mark(c)

            for j, d in enumerate(token[i + 1:], i + 1):
                if final_mark:
                    if not is_final_mark(d):
                        return j
                elif c != d:
                    return j

            return len(token)

        def skip(i, c):
            if c == '.' or c == '+':  # .1, +1
                return self.is_digit(token, i + 1)
            if c == '-':  # -1
                return i == 0 and self.is_digit(token, i + 1)
            if c == ',':  # 1,000,000
                return self.is_digit(token, i - 1) and self.is_digit(token, i + 1, i + 4) and not self.is_digit(token, i + 4)
            if c == ':':
                # 1:2
                return self.is_digit(token, i - 1) and self.is_digit(token, i + 1)
            if is_single_quote(c):
                return self.is_digit(token, i + 1, i + 3) and not self.is_digit(token, i + 3)  # '97
            return False

        def split(i, c, p0, p1):
            if p0(c):
                j = index_last_sequence(i, c)

                if p1(i, j):
                    idx = begin + i
                    lst = begin + j

                    self.tokenize_aux(tokens, offsets, text, begin, idx, offset)
                    self.add_token(tokens, offsets, token[i:j], idx, lst, offset)
                    self.tokenize_aux(tokens, offsets, text, lst, end, offset)
                    return True

            return False

        def separator_0(c):
            return c in {',', ';', ':', '~', '&', '|', '/'} or is_bracket(c) or is_arrow(c) or is_double_quote(c) or is_hyphen(c)

        def edge_symbol_0(c):
            return is_single_quote(c) or is_final_mark(c)

        def currency_like_0(c):
            return c == '#' or is_currency(c)

        def edge_symbol_1(i, j):
            return i + 1 < j or i == 0 or j == len(token) or is_punct(token[i - 1]) or is_punct(token[j])

        def currency_like_1(i, j):
            return i + 1 < j or j == len(token) or token[j].isdigit()

        # split by symbols
        for i, c in enumerate(token):
            if skip(i, c):
                continue
            if split(i, c, separator_0, lambda i, j: True):
                return True
            if split(i, c, edge_symbol_0, edge_symbol_1):
                return True
            if split(i, c, currency_like_0, currency_like_1):
                return True
        return False

    def add_token(self, tokens, offsets, token, begin, end, offset):
        if not self.concat_token(tokens, offsets, token, end) and not self.split_token(tokens, offsets, token, begin, end, offset):
            self.add_token_aux(tokens, offsets, token, begin, end, offset)

    def concat_token(self, tokens, offsets, token, end):
        def apostrophe_front(prev, curr):
            return len(prev) == 1 and is_single_quote(prev) and curr in self.APOSTROPHE_FRONT

        def abbreviation(prev, curr):
            return curr == '.' and (self.RE_ABBREVIATION.match(prev) or prev in self.ABBREVIATION_PERIOD)

        def acronym(prev, curr, next):
            return len(curr) == 1 and curr in {'&', '|', '/'} and (len(prev) <= 2 and len(next) <= 2 or prev.isupper() and next.isupper())

        def hyphenated(prev, curr, next):
            p = len(prev)

            if len(curr) == 1 and is_hyphen(curr):
                if self.is_digit(prev, p - 3, p) and (p == 3 or is_hyphen(prev[p - 4])) and next.isdigit():
                    # 000-0000, 000-000-0000
                    return True
                if prev[-1].isalnum() and (len(prev) == 1 or is_hyphen(prev[p - 2])) and len(next) == 1 and next.isalnum():
                    # p-u-s-h
                    return True
                return (prev in self.HYPHEN_PREFIX and next.isalnum()) or (next in self.HYPHEN_SUFFIX and prev.isalnum())

            return False

        def coloned(prev, curr, next):
            if prev == 're' and curr == ':' and next == 'invent':
                # re:Invent
                return True

            return False

        def no_dot_digit(prev, curr, next):
            if prev == 'no' and curr == '.' and next[0].isdigit():
                t, o = tokens.pop(), offsets.pop()
                tokens[-1] += t
                offsets[-1] = (offsets[-1][0], o[1])
                return True

            return False

        # concatenate split tokens if necessary
        if tokens:
            prev = tokens[-1].lower()
            curr = token.lower()

            if apostrophe_front(prev, curr) or abbreviation(prev, curr):
                tokens[-1] += token
                offsets[-1] = (offsets[-1][0], end)
                return True

        if len(tokens) >= 2:
            prev = tokens[-2].lower()
            curr = tokens[-1].lower()
            next = token.lower()

            if acronym(tokens[-2], curr, token) or hyphenated(prev, curr, next) or coloned(prev, curr, next):
                tokens[-2] += tokens[-1] + token
                offsets[-2] = (offsets[-2][0], end)
                del tokens[-1]
                del offsets[-1]
                return True

            no_dot_digit(prev, curr, next)

        return False

    def split_token(self, tokens, offsets, token, begin, end, offset):
        def unit():
            m = self.RE_UNIT.search(token)
            if m:
                idx = begin + m.start(2)
                self.add_token_aux(tokens, offsets, token[:m.start(2)], begin, idx, offset)
                self.add_token_aux(tokens, offsets, m.group(2), idx, end, offset)
                return True
            return False

        def concat_words():
            t = self.MAP_CONCAT_WORD.get(token.lower(), None)
            if t:
                i = 0
                for j in t:
                    self.add_token_aux(tokens, offsets, token[i:j], begin + i, begin + j, offset)
                    i = j
                return True
            return False

        def final_mark():
            m = self.RE_FINAL_MARK_IN_BETWEEN.match(token)
            if m:
                for i in range(1, 4):
                    self.add_token_aux(tokens, offsets, m.group(i), begin + m.start(i), begin + m.end(i), offset)
                return True
            return False

        return unit() or concat_words() or final_mark()

    @staticmethod
    def add_token_aux(tokens, offsets, token, begin, end, offset):
        tokens.append(token)
        offsets.append((begin + offset, end + offset))

    @staticmethod
    def is_digit(token, i, j=None):
        if 0 <= i < len(token):
            if j is None:
                return token[i].isdigit()
            if i < j <= len(token):
                return token[i:j].isdigit()
        return False
