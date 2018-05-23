# ========================================================================
# Copyright 2018 Emory University
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
import os
import re

from elit.util.string import *
from elit.nlp.structure import TOKEN, OFFSET
from elitsdk.sdk import Component
from pkg_resources import resource_filename

__author__ = "Jinho D. Choi, Gary Lai"


class Tokenizer(Component):
    def __init__(self):
        super(Tokenizer, self).__init__()
        pass

    @abc.abstractmethod
    def decode(self, input_data, offset=0, **kwargs):
        """

        :type input_data: str
        :type offset: int
        :param input_data: the input text.
        :param offset: the starting offset.
        :return: the tuple of (tokens, offsets[, custom values]*); see the comments for Tokenizer.offsets() for more details about the offsets.
        :rtype: json
        """
        pass

    def load(self, model_path, *args, **kwargs):
        """

        :param model_path:
        :param args:
        :param kwargs:
        :return:
        """
        pass

    def train(self, trn_data, dev_data, *args, **kwargs):
        """

        :param trn_data:
        :param dev_data:
        :param args:
        :param kwargs:
        :return:
        """
        pass

    def save(self, model_path, *args, **kwargs):
        """

        :param model_path:
        :param args:
        :param kwargs:
        :return:
        """
        pass

    @staticmethod
    def offsets(input_data, tokens, offset=0):
        """

        :type input_data: str
        :type tokens: list of str
        :type offset: int
        :param input_data: the input text.
        :param tokens: the list of tokens split from the input text.
        :param offset: offset of tokens
        :return: the list of (begin, end) offsets, where the begin (inclusive) and the end (exclusive) offsets indicate the caret positions of the first and the last characters of the corresponding token, respectively. e.g., text = 'Hello, world!', tokens = ['Hello', ',', 'world', '!'] -> [(0, 5), (5, 6), (7, 12), (12, 13)]
        :rtype: json
        """

        def get_offset(token):
            nonlocal end
            begin = input_data.index(token, end)
            end = begin + len(token)
            return begin + offset, end + offset

        end = 0
        return [get_offset(token) for token in tokens]


class SpaceTokenizer(Tokenizer):

    def __init__(self):
        super(SpaceTokenizer, self).__init__()

    def decode(self, input_data, offset=0, **kwargs):
        """

        :param input_data:
        :param offset:
        :param kwargs:
        :return:
        """
        tokens = input_data.split()
        return tokens, self.offsets(input_data, tokens, offset)


class EnglishTokenizer(Tokenizer):

    def __init__(self):
        super(EnglishTokenizer, self).__init__()
        self.SET_ABBREVIATION_PERIOD = self.read_word_set(
            resource_filename('elit.resources.tokenizer', 'english_abbreviation_period.txt'))
        self.SET_APOSTROPHE_FRONT = self.read_word_set(
            resource_filename('elit.resources.tokenizer', 'english_apostrophe_front.txt'))
        self.MAP_CONCAT_WORD = self.read_concat_word_dict(
            resource_filename('elit.resources.tokenizer', 'english_concat_words.txt'))
        self.SET_HYPHEN_PREFIX = self.read_word_set(
            resource_filename('elit.resources.tokenizer', 'english_hyphen_prefix.txt'))
        self.SET_HYPHEN_SUFFIX = self.read_word_set(
            resource_filename('elit.resources.tokenizer', 'english_hyphen_suffix.txt'))
        # regular expressions
        self.RE_NETWORK_PROTOCOL = re.compile(
            r"((http|https|ftp|sftp|ssh|ssl|telnet|smtp|pop3|imap|imap4|sip)(://))")
        """
        # :abc:
        # <3 </3 <\3
        # (: ): \: *: $: (-: (^: (= (;
        # :) :( =) B) 8) :-) :^) :3 :D :p :| :(( :---)
        """
        self.RE_EMOTICON = re.compile(
            r"(:\w+:|<[\\/]?3|[\(\)\\\|\*\$][-\^]?[:\=\;]|[:\=\;B8]([-\^]+)?[3DOPp\@\$\*\(\)\\/\|]+)(\W|$)")
        """
        jinho@elit.cloud
        jinho.choi@elit.cloud
        choi@demo.elit.cloud
        jinho:choi@127.0.0.1
        """
        self.RE_EMAIL = re.compile(
            r"[\w\-\.]+(:\S+)?@(([A-Za-z0-9\-]+\.)+[A-Za-z]{2,12}|\d{1,3}(\.\d{1,3}){3})")
        """
        &arrow;
        &#123; &#x123; &#X123;
        """
        self.RE_HTML_ENTITY = re.compile(r"&([A-Za-z]+|#[Xx]?\d+);")
        """
        # [1] (1a) {A} <a1> [***] [A.a] [A.1] [1.a] ((---))
        """
        self.RE_LIST_ITEM = re.compile(
            r"(([\[\(\{\<]+)(\d+[A-Za-z]?|[A-Za-z]\d*|\W+)(\.(\d+|[A-Za-z]))*([\]\)\}\>])+)")
        """
        don't don’t I'll HE'S
        """
        self.RE_APOSTROPHE = re.compile(
            r"(?i)[a-z](n['\u2019]t|['\u2019](ll|nt|re|ve|[dmstz]))(\W|$)")
        """
        a.b.c 1-2-3
        """
        self.RE_ABBREVIATION = re.compile(r"[A-Za-z0-9]([\.-][A-Za-z0-9])*$")
        """
        10kg 1cm
        """
        self.RE_UNIT = re.compile(
            r"(?i)(\d)([acdfkmnpyz]?[mg]|[ap]\.m|ch|cwt|d|drc|ft|fur|gr|h|in|lb|lea|mi|ms|oz|pg|qtr|yd)$")
        """
        hello.World
        """
        self.RE_FINAL_MARK_IN_BETWEEN = re.compile(
            r"([A-Za-z]{3,})([\.\?\!]+)([A-Za-z]{3,})$")

    def load(self, model_path, *args, **kwargs):
        self.SET_ABBREVIATION_PERIOD = self.read_word_set(
            os.path.join(model_path, 'english_abbreviation_period.txt'))
        self.SET_APOSTROPHE_FRONT = self.read_word_set(
            os.path.join(model_path, 'english_apostrophe_front.txt'))
        self.MAP_CONCAT_WORD = self.read_concat_word_dict(
            os.path.join(model_path, 'english_concat_words.txt'))
        self.SET_HYPHEN_PREFIX = self.read_word_set(
            os.path.join(model_path, 'english_hyphen_prefix.txt'))
        self.SET_HYPHEN_SUFFIX = self.read_word_set(
            os.path.join(model_path, 'english_hyphen_suffix.txt'))

    def decode(self, input_data, offset=0, **kwargs):
        tokens = []
        offsets = []

        # no valid token in the input text
        if not input_data or input_data.isspace():
            return tokens, offsets

        # skip beginning and ending spaces
        begin = next(i for i, c in enumerate(input_data) if not c.isspace())
        last = len(input_data) - next(
            i for i, c in enumerate(reversed(input_data)) if not c.isspace())

        # search for in-between spaces
        for end, c in enumerate(input_data[begin + 1:last], begin + 1):
            if c.isspace():
                self.tokenize_aux(tokens, offsets, input_data,
                                  begin, end, offset)
                begin = end + 1

        self.tokenize_aux(tokens, offsets, input_data, begin, last, offset)
        return tokens, offsets

    def tokenize_aux(self, tokens, offsets, text, begin, end, offset):
        if begin >= end or end > len(text):
            return False
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
                    self.tokenize_aux(tokens, offsets, text,
                                      begin, idx, offset)
                    self.add_token(tokens, offsets,
                                   token[m.start():], idx, end, offset)
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
                return self.is_digit(token, i - 1) and self.is_digit(token, i + 1,
                                                                     i + 4) and not self.is_digit(
                    token, i + 4)
            if c == ':':
                # 1:2
                return self.is_digit(token, i - 1) and self.is_digit(token, i + 1)
            if is_single_quote(c):
                return self.is_digit(token, i + 1, i + 3) and not self.is_digit(
                    token,
                    i + 3)  # '97
            return False

        def split(i, c, p0, p1):
            if p0(c):
                j = index_last_sequence(i, c)

                if p1(i, j):
                    idx = begin + i
                    lst = begin + j

                    self.tokenize_aux(tokens, offsets, text,
                                      begin, idx, offset)
                    self.add_token(tokens, offsets,
                                   token[i:j], idx, lst, offset)
                    self.tokenize_aux(tokens, offsets, text, lst, end, offset)
                    return True

            return False

        def separator_0(c):
            return c in {',', ';', ':', '~', '&', '|', '/'} or \
                is_bracket(c) or is_arrow(
                    c) or is_double_quote(c) or is_hyphen(c)

        def edge_symbol_0(c):
            return is_single_quote(c) or is_final_mark(c)

        def currency_like_0(c):
            return c == '#' or is_currency(c)

        def edge_symbol_1(i, j):
            return i + 1 < j or i == 0 or j == len(token) or is_punct(token[i - 1]) or is_punct(
                token[j])

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
        if not self.concat_token(tokens, offsets, token, end) and \
                not self.split_token(tokens, offsets, token, begin, end, offset):
            self.add_token_aux(tokens, offsets, token, begin, end, offset)

    def concat_token(self, tokens, offsets, token, end):
        def apostrophe_front(prev, curr):
            return len(prev) == 1 and is_single_quote(prev) and curr in self.SET_APOSTROPHE_FRONT

        def abbreviation(prev, curr):
            return curr == '.' and (
                self.RE_ABBREVIATION.match(prev) or prev in self.SET_ABBREVIATION_PERIOD)

        def acronym(prev, curr, next):
            return len(curr) == 1 and curr in {'&', '|', '/'} and (
                len(prev) <= 2 and len(next) <= 2 or prev.isupper() and next.isupper())

        def hyphenated(prev, curr, next):
            p = len(prev)

            if len(curr) == 1 and is_hyphen(curr):
                if self.is_digit(prev, p - 3, p) and (p == 3 or is_hyphen(
                        prev[p - 4])) and next.isdigit():
                    # 000-0000, 000-000-0000
                    return True
                if prev[-1].isalnum() and (len(prev) == 1 or is_hyphen(prev[p - 2])) and len(
                        next) == 1 and next.isalnum():
                    # p-u-s-h
                    return True
                return (prev in self.SET_HYPHEN_PREFIX and next.isalnum()) or (
                    next in self.SET_HYPHEN_SUFFIX and prev.isalnum())

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

            if acronym(tokens[-2], curr, token) or hyphenated(prev, curr, next):
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
                self.add_token_aux(
                    tokens, offsets, token[:m.start(2)], begin, idx, offset)
                self.add_token_aux(
                    tokens, offsets, m.group(2), idx, end, offset)
                return True
            return False

        def concat_words():
            t = self.MAP_CONCAT_WORD.get(token.lower(), None)
            if t:
                i = 0
                for j in t:
                    self.add_token_aux(
                        tokens, offsets, token[i:j], begin + i, begin + j, offset)
                    i = j
                return True
            return False

        def final_mark():
            m = self.RE_FINAL_MARK_IN_BETWEEN.match(token)
            if m:
                for i in range(1, 4):
                    self.add_token_aux(tokens, offsets, m.group(i), begin + m.start(i),
                                       begin + m.end(i), offset)
                return True
            return False

        return unit() or concat_words() or final_mark()

    @staticmethod
    def read_word_set(filename):
        fin = codecs.open(filename, mode='r', encoding='utf-8')
        s = set(line.strip() for line in fin)
        print('Init: %s(keys=%d)' % (filename, len(s)))
        return s

    @staticmethod
    def read_concat_word_dict(filename):
        def key_value(line):
            l = [i for i, c in enumerate(line) if c == ' ']
            l = [i - o for o, i in enumerate(l)]
            line = line.replace(' ', '')
            l.append(len(line))
            return line, l

        fin = codecs.open(filename, mode='r', encoding='utf-8')
        d = dict(key_value(line.strip()) for line in fin)
        print('Init: %s(keys=%d)' % (filename, len(d)))
        return d

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


class Segmenter(Component):
    @abc.abstractmethod
    def decode(self, input_data, offsets=0, **kwargs):
        pass

    def load(self, model_path, *args, **kwargs):
        pass

    def train(self, trn_data, dev_data, *args, **kwargs):
        pass

    def save(self, model_path, *args, **kwargs):
        pass


class EnglishSegmenter(Segmenter):
    def decode(self, input_data, offsets=0, **kwargs):
        def sentence(begin, end):
            return {TOKEN: input_data[begin:end], OFFSET: offsets[begin:end]}

        sentences = []
        begin = 0
        right_quote = True

        for i, token in enumerate(input_data):
            t = token[0]
            if t == '"':
                right_quote = not right_quote

            if begin == i:
                if sentences and (is_right_bracket(t) or t == u'\u201D' or t == '"' and right_quote):
                    d = sentences[-1]
                    d[TOKEN].append(token)
                    d[OFFSET].append(offsets[i])
                    begin = i + 1
            elif all(is_final_mark(c) for c in token):
                sentences.append(sentence(begin, i + 1))
                begin = i + 1

        if begin < len(input_data):
            sentences.append(sentence(begin, len(input_data)))

        return sentences
