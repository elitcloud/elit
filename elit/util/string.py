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
import string

__author__ = 'Jinho D. Choi'


def is_range(c, begin, end):
    return begin <= c <= end


def is_single_quote(c):
    return c in {'\'', '`'} or is_range(c, u'\u2018', u'\u201B')


def is_double_quote(c):
    return c == '"' or is_range(c, u'\u201C', u'\u201F')


def is_left_bracket(c):
    return c in {'(', '{', '[', '<'}


def is_right_bracket(c):
    return c in {')', '}', ']', '>'}


def is_bracket(c):
    return is_left_bracket(c) or is_right_bracket(c)


def is_hyphen(c):
    return c == '-' or is_range(c, u'\u2010', u'\u2014')


def is_arrow(c):
    return is_range(c, u'\u2190', u'\u21FF') or is_range(c, u'\u27F0', u'\u27FF') or is_range(c, u'\u2900', u'\u297F')


def is_currency(c):
    return c == '$' or is_range(c, u'\u00A2', u'\u00A5') or is_range(c, u'\u20A0', u'\u20CF')


def is_final_mark(c):
    return c in {'.', '?', '!', u'\u203C'} or is_range(c, u'\u2047', u'\u2049')


def is_punct(c):
    return c in string.punctuation


def collapse_digits(s):
    def get(i, c):
        if i+1 < len(s) and digits[i+1]:
            if is_currency(c) or c in {'.', '-', '+', '#'}:
                return ''
            if i-1 >= 0 and digits[i-1] and (is_hyphen(c) or c in {',', ':', '/', '='}):
                return ''
        elif i-1 >= 0 and digits[i-1]:
            if c == '%':
                return ''
        elif digits[i]:
            if t and t[0] == '0':
                return ''

        t[0] = '0' if digits[i] else c
        return t[0]

    t = ['']
    digits = [c.isdigit() for c in s]
    return ''.join([get(i, c) for i, c in enumerate(s)])
