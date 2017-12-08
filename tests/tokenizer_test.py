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
import unittest

from elit.nlp.task.tokenize import SpaceTokenizer, EnglishTokenizer

__author__ = 'Jinho D. Choi'


class TestSpaceTokenizer(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestSpaceTokenizer, self).__init__(*args, **kwargs)
        self.tok = SpaceTokenizer()

    def test_space_tokenizer(self):
        # hello world
        tokenizer_helper(self, '  Hello\t\r, world  !\n\n', ['Hello', ',', 'world', '!'],
                         [(2, 7), (9, 10), (11, 16), (18, 19)])


class TestEnglishTokenizer(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestEnglishTokenizer, self).__init__(*args, **kwargs)
        self.tok = EnglishTokenizer('../resources/tokenize')

    def test_space(self):
        # empty string
        tokenizer_helper(self, '', [], [])

        # only spaces
        tokenizer_helper(self, ' \t\n\r', [], [])

        # no space
        tokenizer_helper(self, 'abc', ['abc'], [(0, 3)])

        # multiple consecutive spaces
        tokenizer_helper(self, ' A\tBC  D12 \n', ['A', 'BC', 'D12'], [(1, 2), (3, 5), (7, 10)])

    def test_regex(self):
        # html entity
        tokenizer_helper(self, 'ab&larr;cd&#8592;&#x2190;ef&rarr;',
                         ['ab', '&larr;', 'cd', '&#8592;', '&#x2190;', 'ef', '&rarr;'],
                         [(0, 2), (2, 8), (8, 10), (10, 17), (17, 25), (25, 27), (27, 33)])

        # email address
        tokenizer_helper(self,
                         'a;jinho@elit.com,b;jinho.choi@elit.com,choi@elit.emory.edu,jinho:choi@0.0.0.0',
                         ['a', ';', 'jinho@elit.com', ',', 'b', ';', 'jinho.choi@elit.com', ',',
                          'choi@elit.emory.edu', ',', 'jinho:choi@0.0.0.0'],
                         [(0, 1), (1, 2), (2, 16), (16, 17), (17, 18), (18, 19), (19, 38), (38, 39),
                          (39, 58), (58, 59), (59, 77)])

        # hyperlink
        tokenizer_helper(self, 'a:http://ab sftp://ef abeftp://',
                         ['a', ':', 'http://ab', 'sftp://ef', 'abe', 'ftp://'],
                         [(0, 1), (1, 2), (2, 11), (12, 21), (22, 25), (25, 31)])

        # emoticon
        tokenizer_helper(self, ':-) A:-( :). B:smile::sad: C:):(! :)., :-))) :---( Hi:).',
                         [':-)', 'A', ':-(', ':)', '.', 'B', ':smile:', ':sad:', 'C', ':)', ':(',
                          '!', ':)', '.', ',', ':-)))', ':---(', 'Hi', ':)', '.'],
                         [(0, 3), (4, 5), (5, 8), (9, 11), (11, 12), (13, 14), (14, 21), (21, 26),
                          (27, 28), (28, 30), (30, 32), (32, 33), (34, 36), (36, 37), (37, 38),
                          (39, 44), (45, 50), (51, 53), (53, 55), (55, 56)])

        # list item
        tokenizer_helper(self, '[a](1)(1.a)[11.22.a.33](A.1)[a1][hello]{22}',
                         ['[a]', '(1)', '(1.a)', '[11.22.a.33]', '(A.1)', '[a1]', '[', 'hello', ']',
                          '{22}'],
                         [(0, 3), (3, 6), (6, 11), (11, 23), (23, 28), (28, 32), (32, 33), (33, 38),
                          (38, 39), (39, 43)])

        # apostrophe
        tokenizer_helper(self, "don't he's can't does'nt 0's DON'T ab'cd",
                         ['do', "n't", 'he', "'s", 'ca', "n't", 'does', "'nt", "0's", 'DO', "N'T",
                          "ab'cd"],
                         [(0, 2), (2, 5), (6, 8), (8, 10), (11, 13), (13, 16), (17, 21), (21, 24),
                          (25, 28), (29, 31), (31, 34), (35, 40)])

    def test_symbol(self):
        # handle symbols with digits
        tokenizer_helper(self, ".1 +1 -1 1,000,000.00 1,00 '97 '90s '1990 10:30 1:2 a:b",
                         ['.1', '+1', '-1', '1,000,000.00', '1', ',', '00', "'97", "'90s", "'",
                          '1990', '10:30', '1:2', 'a', ':', 'b'],
                         [(0, 2), (3, 5), (6, 8), (9, 21), (22, 23), (23, 24), (24, 26), (27, 30),
                          (31, 35), (36, 37), (37, 41), (42, 47), (48, 51), (52, 53), (53, 54),
                          (54, 55)])

        # separator and edge symbols
        tokenizer_helper(self, "aa;;;b: c\"\"d\"\" 0'''s ''a''a'' 'a'a'a'.?!..",
                         ['aa', ';;;', 'b', ':', 'c', '""', 'd', '""', '0', "'''", 's', "''", 'a',
                          "''", 'a', "''", "'", "a'a'a", "'", '.?!..'],
                         [(0, 2), (2, 5), (5, 6), (6, 7), (8, 9), (9, 11), (11, 12), (12, 14),
                          (15, 16), (16, 19), (19, 20), (21, 23), (23, 24), (24, 26), (26, 27),
                          (27, 29), (30, 31), (31, 36), (36, 37), (37, 42)])

        # currency like
        tokenizer_helper(self, '#happy2018,@Jinho_Choi: ab@cde #1 $1 ',
                         ['#happy2018', ',', '@Jinho_Choi', ':', 'ab@cde', '#', '1', '$', '1'],
                         [(0, 10), (10, 11), (11, 22), (22, 23), (24, 30), (31, 32), (32, 33),
                          (34, 35), (35, 36)])

        # hyphen
        tokenizer_helper(self, "1997-2012,1990's-2000S '97-2012 '12-14's",
                         ['1997', '-', '2012', ',', "1990's", '-', '2000S', "'97", '-', '2012',
                          "'12", '-', "14's"],
                         [(0, 4), (4, 5), (5, 9), (9, 10), (10, 16), (16, 17), (17, 22), (23, 26),
                          (26, 27), (27, 31), (32, 35), (35, 36), (36, 40)])

    def test_concat(self):
        # apostrophe for abbreviation
        tokenizer_helper(self, "'CAUSE 'tis ' em 'happy",
                         ["'CAUSE", "'tis", "'em", "'", 'happy'],
                         [(0, 6), (7, 11), (12, 16), (17, 18), (18, 23)])

        # abbreviation
        tokenizer_helper(self, 'A.B. a.B.c AB.C. Ph.D. 1.2. A-1.',
                         ['A.B.', 'a.B.c', 'AB.C', '.', 'Ph.D.', '1.2.', 'A-1.'],
                         [(0, 4), (5, 10), (11, 15), (15, 16), (17, 22), (23, 27), (28, 32)])

        # acronym
        tokenizer_helper(self, 'ab&cd AB|CD 1/2 a-1 1-2',
                         ['ab&cd', 'AB|CD', '1/2', 'a-1', '1-2'],
                         [(0, 5), (6, 11), (12, 15), (16, 19), (20, 23)])

        # hyphenated
        tokenizer_helper(self,
                         'mis-predict mic-predict book - able book-es 000-0000 000-000-000 p-u-s-h-1-2',
                         ['mis-predict', 'mic', '-', 'predict', 'book-able', 'book', '-', 'es',
                          '000-0000', '000-000-000', 'p-u-s-h-1-2'],
                         [(0, 11), (12, 15), (15, 16), (16, 23), (24, 35), (36, 40), (40, 41),
                          (41, 43), (44, 52), (53, 64), (65, 76)])

        # No. 1
        tokenizer_helper(self, 'No. 1 No. a No.',
                         ['No.', '1', 'No', '.', 'a', 'No', '.'],
                         [(0, 3), (4, 5), (6, 8), (8, 9), (10, 11), (12, 14), (14, 15)])

    def test_split(self):
        # unit
        tokenizer_helper(self, '20mg 100cm 1st 11a.m. 10PM',
                         ['20', 'mg', '100', 'cm', '1st', '11', 'a.m.', '10', 'PM'],
                         [(0, 2), (2, 4), (5, 8), (8, 10), (11, 14), (15, 17), (17, 21), (22, 24),
                          (24, 26)])

        # concatenated word
        tokenizer_helper(self, "whadya DON'CHA",
                         ['wha', 'd', 'ya', 'DO', "N'", 'CHA'],
                         [(0, 3), (3, 4), (4, 6), (7, 9), (9, 11), (11, 14)])

        # final mark
        tokenizer_helper(self, 'Mbaaah.Please hello.!?world',
                         ['Mbaaah', '.', 'Please', 'hello', '.!?', 'world'],
                         [(0, 6), (6, 7), (7, 13), (14, 19), (19, 22), (22, 27)])

    def test_offset(self):
        tokenizer_helper(self, 'Hello, world!', ['Hello', ',', 'world', '!'],
                         [(5, 10), (10, 11), (12, 17), (17, 18)], 5)

        # def test_segment(self):
        #     tokens, offsets = self.tok.tokenize('. "1st sentence." 2nd sentence? "3rd sentence!"')
        #     self.assertEqual(self.tok.segment(tokens, offsets),
        #     [{'tokens': ['.', '"', '1st', 'sentence', '.', '"'], 'offsets': [(0, 1), (2, 3), (3, 6), (7, 15), (15, 16), (16, 17)]},
        #      {'tokens': ['2nd', 'sentence', '?'], 'offsets': [(18, 21), (22, 30), (30, 31)]},
        #      {'tokens': ['"', '3rd', 'sentence', '!', '"'], 'offsets': [(32, 33), (33, 36), (37, 45), (45, 46), (46, 47)]}])


# Change to tokenizer_helper because function name with test cause test framework confused.
def tokenizer_helper(t, s, gold_tokens, gold_offsets, offset=0):
    tokens, offsets = t.tok.decode(s, offset)
    t.assertEqual(tokens, gold_tokens)
    t.assertEqual(offsets, gold_offsets)


if __name__ == '__main__':
    unittest.main()
