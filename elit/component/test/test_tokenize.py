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

        # multiple consecutive spaces
        test(self, ' A\tBC  D12 \n', ['A', 'BC', 'D12'], [(1, 2), (3, 5), (7, 10)])

    def test_regex(self):
        # html entity
        test(self, 'ab&larr;cd&#8592;&#x2190;ef&rarr;',
             ['ab', '&larr;', 'cd', '&#8592;', '&#x2190;', 'ef', '&rarr;'],
             [(0, 2), (2, 8), (8, 10), (10, 17), (17, 25), (25, 27), (27, 33)])

        # email address
        test(self, 'a;jinho@elit.com,b;jinho.choi@elit.com,choi@elit.emory.edu,jinho:choi@0.0.0.0',
             ['a', ';', 'jinho@elit.com', ',', 'b', ';', 'jinho.choi@elit.com', ',', 'choi@elit.emory.edu', ',', 'jinho:choi@0.0.0.0'],
             [(0, 1), (1, 2), (2, 16), (16, 17), (17, 18), (18, 19), (19, 38), (38, 39), (39, 58), (58, 59), (59, 77)])

        # hyperlink
        test(self, 'a:http://ab sftp://ef abeftp://',
             ['a', ':', 'http://ab', 'sftp://ef', 'abe', 'ftp://'],
             [(0, 1), (1, 2), (2, 11), (12, 21), (22, 25), (25, 31)])

        # emoticon
        test(self, ':-) A:-( :). B:smile::sad: C:):(! :)., :-))) :---( Hi:).',
             [':-)', 'A', ':-(', ':)', '.', 'B', ':smile:', ':sad:', 'C', ':)', ':(', '!', ':)', '.', ',', ':-)))', ':---(', 'Hi', ':)', '.'],
             [(0, 3), (4, 5), (5, 8), (9, 11), (11, 12), (13, 14), (14, 21), (21, 26), (27, 28), (28, 30), (30, 32), (32, 33), (34, 36), (36, 37), (37, 38), (39, 44), (45, 50), (51, 53), (53, 55), (55, 56)])

        # list item
        test(self, '[a](1)(1.a)[11.22.a.33](A.1)[a1][hello]{22}',
             ['[a]', '(1)', '(1.a)', '[11.22.a.33]', '(A.1)', '[a1]', '[', 'hello', ']', '{22}'],
             [(0, 3), (3, 6), (6, 11), (11, 23), (23, 28), (28, 32), (32, 33), (33, 38), (38, 39), (39, 43)])

        # apostrophe
        test(self, "don't he's can't does'nt 0's DON'T ab'cd",
             ['do', "n't", 'he', "'s", 'ca', "n't", 'does', "'nt", "0's", 'DO', "N'T", "ab'cd"],
             [(0, 2), (2, 5), (6, 8), (8, 10), (11, 13), (13, 16), (17, 21), (21, 24), (25, 28), (29, 31), (31, 34), (35, 40)])

    def test_symbol(self):




        s = ".1 +1 -1 1,000,000.00 1,00 '97 '90s '1990 10:30 1:2 a:b"
        print(self.tokenize(s))
        # s = "1997-2012,1990's-2000S '97-2012 '12-14's"
        # print(self.tokenize(s))
        s = "aa;;;b: c\"\"d\"\" 0'''s ''a''a'' 'a'a'a'.?!.."
        print(self.tokenize(s))
        s = "#happy2018,@Jinho_Choi: ab@cde"
        print(self.tokenize(s))
        s = "#1 $1 "
        print(self.tokenize(s))



def test(t, s, gold_tokens, gold_offsets):
    tokens, offsets = t.tokenize(s)
    t.assertEqual(tokens, gold_tokens)
    t.assertEqual(offsets, gold_offsets)


if __name__ == '__main__':
    unittest.main()


