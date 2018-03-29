# ========================================================================
# Copyright 2017 Emory University
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

from elit.nlp.structure import TOKEN, OFFSET
from elit.util.string import *
from elit.util.string import is_right_bracket, is_final_mark

__author__ = 'Jinho D. Choi'


class Tokenizer(object):
    @abc.abstractmethod
    def decode(self, text, offset=0):
        """
        :param text: the input text.
        :type text: str
        :param offset: the starting offset.
        :type offset: int
        :return:
            the tuple of (tokens, offsets[, custom values]*);
            see the comments for Tokenizer.offsets() for more details about the offsets.
        :rtype: (list of str, list of (int, int), *args)
        """
        return

    @staticmethod
    def offsets(text, tokens, offset=0):
        """
        :param text: the input text.
        :type text: str
        :param tokens: the list of tokens split from the input text.
        :type tokens: list of str
        :return:
            the list of (begin, end) offsets, where the begin (inclusive) and the end (exclusive) offsets indicate
            the caret positions of the first and the last characters of the corresponding token, respectively.
            e.g., text = 'Hello, world!', tokens = ['Hello', ',', 'world', '!'] -> [(0, 5), (5, 6), (7, 12), (12, 13)]
        :rtype list of (int, int)
        """
        def get_offset(token):
            nonlocal end
            begin = text.index(token, end)
            end = begin + len(token)
            return (begin+offset, end+offset)

        end = 0
        return [get_offset(token) for token in tokens]


class SpaceTokenizer(Tokenizer):
    """
    Tokenize by only spaces.
    """
    def decode(self, text, offset=0):
        tokens = text.split()
        return tokens, Tokenizer.offsets(text, tokens, offset)


class EnglishTokenizer(Tokenizer):
    """
    The default tokenizer for English.
    """
    def __init__(self, resource_dir):
        """
        :param resource_dir: the path to the directory containing resources for tokenization.
        :type resource_dir: str
        """
        # from resources
        self.SET_ABBREVIATION_PERIOD = read_word_set(os.path.join(resource_dir, 'english_abbreviation_period.txt'))
        self.SET_APOSTROPHE_FRONT = read_word_set(os.path.join(resource_dir, 'english_apostrophe_front.txt'))
        self.MAP_CONCAT_WORD = read_concat_word_dict(os.path.join(resource_dir, 'english_concat_words.txt'))
        self.SET_HYPHEN_PREFIX = read_word_set(os.path.join(resource_dir, 'english_hyphen_prefix.txt'))
        self.SET_HYPHEN_SUFFIX = read_word_set(os.path.join(resource_dir, 'english_hyphen_suffix.txt'))

        # regular expressions
        self.RE_NETWORK_PROTOCOL = re.compile(r"((http|https|ftp|sftp|ssh|ssl|telnet|smtp|pop3|imap|imap4|sip)(://))")
        """
        # :abc:
        # <3 </3 <\3
        # (: ): \: *: $: (-: (^: (= (;
        # :) :( =) B) 8) :-) :^) :3 :D :p :| :(( :---)
        """
        self.RE_EMOTICON = re.compile(r"(:\w+:|<[\\/]?3|[\(\)\\\|\*\$][-\^]?[:\=\;]|[:\=\;B8]([-\^]+)?[3DOPp\@\$\*\(\)\\/\|]+)(\W|$)")
        """
        jinho@elit.cloud
        jinho.choi@elit.cloud
        choi@demo.elit.cloud
        jinho:choi@127.0.0.1
        """
        self.RE_EMAIL = re.compile(r"[\w\-\.]+(:\S+)?@(([A-Za-z0-9\-]+\.)+[A-Za-z]{2,12}|\d{1,3}(\.\d{1,3}){3})")
        """
        &arrow;
        &#123; &#x123; &#X123;
        """
        self.RE_HTML_ENTITY = re.compile(r"&([A-Za-z]+|#[Xx]?\d+);")
        """
        # [1] (1a) {A} <a1> [***] [A.a] [A.1] [1.a] ((---))
        """
        self.RE_LIST_ITEM = re.compile(r"(([\[\(\{\<]+)(\d+[A-Za-z]?|[A-Za-z]\d*|\W+)(\.(\d+|[A-Za-z]))*([\]\)\}\>])+)")
        """
        don't don’t I'll HE'S
        """
        self.RE_APOSTROPHE = re.compile(r"(?i)[a-z](n['\u2019]t|['\u2019](ll|nt|re|ve|[dmstz]))(\W|$)")
        """
        a.b.c 1-2-3
        """
        self.RE_ABBREVIATION = re.compile(r"[A-Za-z0-9]([\.-][A-Za-z0-9])*$")
        """
        10kg 1cm
        """
        self.RE_UNIT = re.compile(r"(?i)(\d)([acdfkmnpyz]?[mg]|[ap]\.m|ch|cwt|d|drc|ft|fur|gr|h|in|lb|lea|mi|ms|oz|pg|qtr|yd)$")
        """
        hello.World
        """
        self.RE_FINAL_MARK_IN_BETWEEN = re.compile(r"([A-Za-z]{3,})([\.\?\!]+)([A-Za-z]{3,})$")

    def decode(self, text, offset=0):
        tokens = []
        offsets = []

        # no valid token in the input text
        if not text or text.isspace(): return tokens, offsets

        # skip beginning and ending spaces
        begin = next(i for i, c in enumerate(text) if not c.isspace())
        last = len(text) - next(i for i, c in enumerate(reversed(text)) if not c.isspace())

        # search for in-between spaces
        for end, c in enumerate(text[begin+1:last], begin+1):
            if c.isspace():
                self.tokenize_aux(tokens, offsets, text, begin, end, offset)
                begin = end + 1

        self.tokenize_aux(tokens, offsets, text, begin, last, offset)
        return tokens, offsets

    def tokenize_aux(self, tokens, offsets, text, begin, end, offset):
        if begin >= end or end > len(text): return False
        token = text[begin:end]

        # handle special cases
        if self.tokenize_trivial(tokens, offsets, token, begin, end, offset): return True
        if self.tokenize_regex(tokens, offsets, text, begin, end, offset, token): return True
        if self.tokenize_symbol(tokens, offsets, text, begin, end, offset, token): return True

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
        if group(self.RE_HTML_ENTITY): return True
        if group(self.RE_EMAIL): return True
        if hyperlink(): return True
        if group(self.RE_EMOTICON, 1): return True
        if group(self.RE_LIST_ITEM): return True
        if group(self.RE_APOSTROPHE, 1): return True
        return False

    def tokenize_symbol(self, tokens, offsets, text, begin, end, offset, token):
        def index_last_sequence(i, c):
            final_mark = is_final_mark(c)

            for j, d in enumerate(token[i+1:], i+1):
                if final_mark:
                    if not is_final_mark(d): return j
                elif c != d:
                    return j

            return len(token)

        def skip(i, c):
            if c == '.' or c == '+': return is_digit(token, i+1)                                                 # .1, +1
            if c == '-': return i == 0 and is_digit(token, i+1)                                                  # -1
            if c == ',': return is_digit(token, i-1) and is_digit(token, i+1, i+4) and not is_digit(token, i+4)  # 1,000,000
            if c == ':': return is_digit(token, i-1) and is_digit(token, i+1)                                    # 1:2
            if is_single_quote(c): return is_digit(token, i+1, i+3) and not is_digit(token, i+3)                 # '97
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
            return c in {',', ';', ':', '~', '&', '|', '/'} or \
                   is_bracket(c) or is_arrow(c) or is_double_quote(c) or is_hyphen(c)

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
            if skip(i, c): continue
            if split(i, c, separator_0, lambda i, j: True): return True
            if split(i, c, edge_symbol_0, edge_symbol_1): return True
            if split(i, c, currency_like_0, currency_like_1): return True

        return False

    def add_token(self, tokens, offsets, token, begin, end, offset):
        if not self.concat_token(tokens, offsets, token, end) and \
           not self.split_token(tokens, offsets, token, begin, end, offset):
            add_token_aux(tokens, offsets, token, begin, end, offset)

    def concat_token(self, tokens, offsets, token, end):
        def apostrophe_front(prev, curr):
            return len(prev) == 1 and is_single_quote(prev) and curr in self.SET_APOSTROPHE_FRONT

        def abbreviation(prev, curr):
            return curr == '.' and (self.RE_ABBREVIATION.match(prev) or prev in self.SET_ABBREVIATION_PERIOD)

        def acronym(prev, curr, next):
            return len(curr) == 1 and curr in {'&', '|', '/'} and (len(prev) <= 2 and len(next) <= 2 or prev.isupper() and next.isupper())

        def hyphenated(prev, curr, next):
            p = len(prev)

            if len(curr) == 1 and is_hyphen(curr):
                if is_digit(prev, p-3, p) and (p == 3 or is_hyphen(prev[p-4])) and next.isdigit(): return True  # 000-0000, 000-000-0000
                if prev[-1].isalnum() and (len(prev) == 1 or is_hyphen(prev[p-2])) and len(next) == 1 and next.isalnum(): return True  # p-u-s-h
                return (prev in self.SET_HYPHEN_PREFIX and next.isalnum()) or (next in self.SET_HYPHEN_SUFFIX and prev.isalnum())

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
                add_token_aux(tokens, offsets, token[:m.start(2)], begin, idx, offset)
                add_token_aux(tokens, offsets, m.group(2), idx, end, offset)
                return True
            return False

        def concat_words():
            t = self.MAP_CONCAT_WORD.get(token.lower(), None)
            if t:
                i = 0
                for j in t:
                    add_token_aux(tokens, offsets, token[i:j], begin + i, begin + j, offset)
                    i = j
                return True
            return False

        def final_mark():
            m = self.RE_FINAL_MARK_IN_BETWEEN.match(token)
            if m:
                for i in range(1, 4):
                    add_token_aux(tokens, offsets, m.group(i), begin + m.start(i), begin + m.end(i), offset)
                return True
            return False

        return unit() or concat_words() or final_mark()


def read_word_set(filename):
    fin = codecs.open(filename, mode='r', encoding='utf-8')
    s = set(line.strip() for line in fin)
    print('Init: %s(keys=%d)' % (filename, len(s)))
    return s


def read_concat_word_dict(filename):
    def key_value(line):
        l = [i for i, c in enumerate(line) if c == ' ']
        l = [i-o for o, i in enumerate(l)]
        line = line.replace(' ', '')
        l.append(len(line))
        return line, l

    fin = codecs.open(filename, mode='r', encoding='utf-8')
    d = dict(key_value(line.strip()) for line in fin)
    print('Init: %s(keys=%d)' % (filename, len(d)))
    return d


def add_token_aux(tokens, offsets, token, begin, end, offset):
    tokens.append(token)
    offsets.append((begin+offset, end+offset))


def is_digit(token, i, j=None):
    if 0 <= i < len(token):
        if j is None: return token[i].isdigit()
        if i < j <= len(token): return token[i:j].isdigit()
    return False


class Segmenter(object):
    @abc.abstractmethod
    def decode(self, tokens, offsets):
        """
        :param tokens: the input tokens.
        :type tokens: list of str
        :param offsets: the offsets of the corresponding tokens in the original text.
        :type offsets: list of (int, int)
        :return: the list of sentences, where each sentence is a dictionary containing tokens and offsets as keys.
        """
        return


class EnglishSegmenter(Segmenter):
    def decode(self, tokens, offsets):
        def sentence(begin, end):
            return {TOKEN: tokens[begin:end], OFFSET: offsets[begin:end]}

        sentences = []
        begin = 0
        right_quote = True

        for i, token in enumerate(tokens):
            t = token[0]
            if t == '"': right_quote = not right_quote

            if begin == i:
                if sentences and (is_right_bracket(t) or t == u'\u201D' or t == '"' and right_quote):
                    d = sentences[-1]
                    d[TOKEN].append(token)
                    d[OFFSET].append(offsets[i])
                    begin = i + 1
            elif all(is_final_mark(c) for c in token):
                sentences.append(sentence(begin, i + 1))
                begin = i + 1

        if begin < len(tokens):
            sentences.append(sentence(begin, len(tokens)))

        return sentences
