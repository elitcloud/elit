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
from elit.component.tokenize import SpaceTokenizer, EnglishTokenizer
import unittest

__author__ = 'Jinho D. Choi'


class TestSpaceTokenizer(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestSpaceTokenizer, self).__init__(*args, **kwargs)
        self.tokenize = SpaceTokenizer().tokenize

    def test(self):
        test(self, '  Hello\t\r, world  !\n\n', ['Hello', ',', 'world', '!'], [(2, 7), (9, 10), (11, 16), (18, 19)])


class TestEnglishTokenizer(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestEnglishTokenizer, self).__init__(*args, **kwargs)
        self.tokenize = EnglishTokenizer('../../../resources/tokenizer').tokenize

    def test_space(self):
        # empty string
        test(self, '', [], [])

        # only spaces
        test(self, ' \t\n\r', [], [])

        # no space
        test(self, 'abc', ['abc'], [(0, 3)])

        # strip
        test(self, ' A\tBC  D12 \n', ['A', 'BC', 'D12'], [(1, 2), (3, 5), (7, 10)])

    def test_regex(self):


        s = '  I’m a boy.'
        print(self.tokenize(s))


def test(t, s, gold_tokens, gold_offsets):
    tokens, offsets = t.tokenize(s)
    t.assertEqual(tokens, gold_tokens)
    t.assertEqual(offsets, gold_offsets)


if __name__ == '__main__':
    unittest.main()


