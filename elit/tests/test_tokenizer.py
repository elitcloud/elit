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
import pytest

__author__ = 'Jinho D. Choi'


@pytest.mark.parametrize('input, offset, expected', [
    ('  Hello\t\r, world  !\n\n', 0,
     {'sens': [{'sid': 0, 'tok': ['Hello', ',', 'world', '!'], 'off': [(2, 7), (9, 10), (11, 16), (18, 19)]}]}),
    ('  Hello\t\r, world  !\n\n', 1,
     {'sens': [{'sid': 0, 'tok': ['Hello', ',', 'world', '!'], 'off': [(3, 8), (10, 11), (12, 17), (19, 20)]}]})
])
def test_space_tokenizer(space_tokenizer, input, offset, expected):
    result = space_tokenizer.decode(input, offset, segment=2)
    assert result == expected


def test_eng_tokenizer_load(english_tokenizer):
    assert len(english_tokenizer.ABBREVIATION_PERIOD) == 159
    assert len(english_tokenizer.APOSTROPHE_FRONT) == 7
    assert len(english_tokenizer.HYPHEN_PREFIX) == 156
    assert len(english_tokenizer.HYPHEN_SUFFIX) == 39


@pytest.mark.parametrize('input, expected', [
    ('', {'sens': []}),
    (' \t\n\r', {'sens': []}),
    ('abc', {'sens': [{'sid': 0, 'tok': ['abc'], 'off': [(0, 3)]}]}),
    (' A\tBC  D12 \n', {'sens': [{'sid': 0, 'tok': ['A', 'BC', 'D12'], 'off': [(1, 2), (3, 5), (7, 10)]}]})
])
def test_eng_tokenizer_space(english_tokenizer, input, expected):
    result = english_tokenizer.decode(input)
    assert result == expected


@pytest.mark.parametrize('input, offset, expected', [
    ('Hello, world!', 5,
     {'sens': [{'sid': 0, 'tok': ['Hello', ',', 'world', '!'], 'off': [(5, 10), (10, 11), (12, 17), (17, 18)]}]}),
])
def test_eng_tokenizer_offset(english_tokenizer, input, offset, expected):
    result = english_tokenizer.decode(input, offset)
    assert result == expected


@pytest.mark.parametrize('input, expected', [
    # html entity
    ('ab&larr;cd&#8592;&#x2190;ef&rarr;',
     {'sens': [{'sid': 0, 'tok': ['ab', '&larr;', 'cd', '&#8592;', '&#x2190;', 'ef', '&rarr;'],
               'off': [(0, 2), (2, 8), (8, 10), (10, 17), (17, 25), (25, 27), (27, 33)]}]}
     ),
    # email address
    ('a;jinho@elit.com,b;jinho.choi@elit.com,choi@elit.emory.edu,jinho:choi@0.0.0.0',
     {'sens': [{'sid': 0,
               'tok': ['a', ';', 'jinho@elit.com', ',', 'b', ';', 'jinho.choi@elit.com', ',', 'choi@elit.emory.edu',
                       ',', 'jinho:choi@0.0.0.0'],
               'off': [(0, 1), (1, 2), (2, 16), (16, 17), (17, 18), (18, 19), (19, 38), (38, 39), (39, 58), (58, 59),
                       (59, 77)]}]}
     ),
    # hyperlink
    ('a:http://ab sftp://ef abeftp://',
     {'sens': [{'sid': 0, 'tok': ['a', ':', 'http://ab', 'sftp://ef', 'abe', 'ftp://'],
               'off': [(0, 1), (1, 2), (2, 11), (12, 21), (22, 25), (25, 31)]}]}
     ),
    # emoticon
    (':-) A:-( :). B:smile::sad: C:):(! :)., :-))) :---( Hi:).',
     {'sens': [{'sid': 0, 'tok': [':-)', 'A', ':-(', ':)', '.'], 'off': [(0, 3), (4, 5), (5, 8), (9, 11), (11, 12)]},
              {'sid': 1, 'tok': ['B', ':smile:', ':sad:', 'C', ':)', ':(', '!'],
               'off': [(13, 14), (14, 21), (21, 26), (27, 28), (28, 30), (30, 32), (32, 33)]},
              {'sid': 2, 'tok': [':)', '.'], 'off': [(34, 36), (36, 37)]},
              {'sid': 3, 'tok': [',', ':-)))', ':---(', 'Hi', ':)', '.'],
               'off': [(37, 38), (39, 44), (45, 50), (51, 53), (53, 55), (55, 56)]}]}
     ),
    # list item
    ('[a](1)(1.a)[11.22.a.33](A.1)[a1][hello]{22}',
     {'sens': [{'sid': 0, 'tok': ['[a]', '(1)', '(1.a)', '[11.22.a.33]', '(A.1)', '[a1]', '[', 'hello', ']', '{22}'],
               'off': [(0, 3), (3, 6), (6, 11), (11, 23), (23, 28), (28, 32), (32, 33), (33, 38), (38, 39), (39, 43)]}]}

     ),
    ("don't he's can't does'nt 0's DON'IN ab'cd",
     {'sens': [{'sid': 0, 'tok': ['do', "n't", 'he', "'s", 'ca', "n't", 'does', "'nt", "0's", "DON'IN", "ab'cd"],
               'off': [(0, 2), (2, 5), (6, 8), (8, 10), (11, 13), (13, 16), (17, 21), (21, 24), (25, 28), (29, 35), (36, 41)]}]}
     ),
])
def test_eng_tokenizer_regex(english_tokenizer, input, expected):
    result = english_tokenizer.decode(input)
    assert result == expected


@pytest.mark.parametrize('input, expected', [
    # handle symbols with digits
    (".1 +1 -1 1,000,000.00 1,00 '97 '90s '1990 10:30 1:2 a:b",
     {'sens': [{'sid': 0,
               'tok': ['.1', '+1', '-1', '1,000,000.00', '1', ',', '00', "'97", "'90s", "'", '1990', '10:30', '1:2',
                       'a', ':', 'b'],
               'off': [(0, 2), (3, 5), (6, 8), (9, 21), (22, 23), (23, 24), (24, 26), (27, 30), (31, 35), (36, 37),
                       (37, 41), (42, 47), (48, 51), (52, 53), (53, 54), (54, 55)]}]}
     ),
    # separator and edge symbols
    ("aa;;;b: c\"\"d\"\" 0'''s ''a''a'' 'a'a'a'.?!..",
     {'sens': [{'sid': 0,
               'tok': ['aa', ';;;', 'b', ':', 'c', '""', 'd', '""', '0', "'''", 's', "''", 'a', "''", 'a', "''", "'",
                       "a'a'a", "'", '.?!..'],
               'off': [(0, 2), (2, 5), (5, 6), (6, 7), (8, 9), (9, 11), (11, 12), (12, 14), (15, 16), (16, 19),
                       (19, 20), (21, 23), (23, 24), (24, 26), (26, 27), (27, 29), (30, 31), (31, 36), (36, 37),
                       (37, 42)]}]}
     ),
    # currency like
    ('#happy2018,@Jinho_Choi: ab@cde #1 $1 ',
     {'sens': [{'sid': 0, 'tok': ['#happy2018', ',', '@Jinho_Choi', ':', 'ab@cde', '#', '1', '$', '1'],
               'off': [(0, 10), (10, 11), (11, 22), (22, 23), (24, 30), (31, 32), (32, 33), (34, 35), (35, 36)]}]}
     ),
    # hyphen
    ("1997-2012,1990's-2000S '97-2012 '12-14's",
     {'sens': [{'sid': 0,
               'tok': ['1997', '-', '2012', ',', "1990's", '-', '2000S', "'97", '-', '2012', "'12", '-', "14's"],
               'off': [(0, 4), (4, 5), (5, 9), (9, 10), (10, 16), (16, 17), (17, 22), (23, 26), (26, 27), (27, 31),
                       (32, 35), (35, 36), (36, 40)]}]}
     )
])
def test_eng_tokenizer_symbol(english_tokenizer, input, expected):
    result = english_tokenizer.decode(input)
    assert result == expected


@pytest.mark.parametrize('input, expected', [
    # apostrophe for abbreviation
    ("'CAUSE 'tis ' em 'happy",
     {'sens': [{'sid': 0, 'tok': ["'CAUSE", "'tis", "'em", "'", 'happy'],
               'off': [(0, 6), (7, 11), (12, 16), (17, 18), (18, 23)]}]}),
    # abbreviation
    ('A.B. a.B.c AB.C. Ph.D. 1.2. A-1.',
     {'sens': [{'tok': ['A.B.', 'a.B.c', 'AB.C', '.'], 'off': [(0, 4), (5, 10), (11, 15), (15, 16)], 'sid': 0},
               {'tok': ['Ph.D.', '1.2.', 'A-1.'], 'off': [(17, 22), (23, 27), (28, 32)], 'sid': 1}]}
     ),
    # acronym
    ('ab&cd AB|CD 1/2 a-1 1-2',
     {'sens': [{'sid': 0, 'tok': ['ab&cd', 'AB|CD', '1/2', 'a-1', '1-2'],
               'off': [(0, 5), (6, 11), (12, 15), (16, 19), (20, 23)]}]}),
    # hyphenated
    ('mis-predict mic-predict book - able book-es 000-0000 000-000-000 p-u-s-h-1-2',
     {'sens': [{'sid': 0,
               'tok': ['mis-predict', 'mic', '-', 'predict', 'book-able', 'book', '-', 'es', '000-0000', '000-000-000',
                       'p-u-s-h-1-2'],
               'off': [(0, 11), (12, 15), (15, 16), (16, 23), (24, 35), (36, 40), (40, 41), (41, 43), (44, 52),
                       (53, 64), (65, 76)]}]}
     ),
    # No. 1
    ('No. 1 No. a No.',
     {'sens': [{'sid': 0, 'tok': ['No.', '1', 'No', '.'], 'off': [(0, 3), (4, 5), (6, 8), (8, 9)]},
              {'sid': 1, 'tok': ['a', 'No', '.'], 'off': [(10, 11), (12, 14), (14, 15)]}]})
])
def test_eng_tokenizer_concat(english_tokenizer, input, expected):
    result = english_tokenizer.decode(input)
    assert result == expected


@pytest.mark.parametrize('input, expected', [
    # unit
    ('20mg 100cm 1st 11a.m. 10PM',
     {'sens': [{'sid': 0, 'tok': ['20', 'mg', '100', 'cm', '1st', '11', 'a.m.', '10', 'PM'],
               'off': [(0, 2), (2, 4), (5, 8), (8, 10), (11, 14), (15, 17), (17, 21), (22, 24), (24, 26)]}]}
     ),
    # concatenated word
    ("whadya DON'CHA",
     {'sens': [{'sid': 0, 'tok': ['wha', 'd', 'ya', 'DO', "N'", 'CHA'], 'off': [(0, 3), (3, 4), (4, 6), (7, 9), (9, 11), (11, 14)]}]}
     ),
    # final mark
    ('Mbaaah.Please hello.!?world',
     {'sens': [{'tok': ['Mbaaah', '.'], 'off': [(0, 6), (6, 7)], 'sid': 0}, {'tok': ['Please', 'hello', '.!?'], 'off': [(7, 13), (14, 19), (19, 22)], 'sid': 1},
               {'tok': ['world'], 'off': [(22, 27)], 'sid': 2}]}
     ),
])
def test_eng_tokenizer_split(english_tokenizer, input, expected):
    result = english_tokenizer.decode(input)
    assert result == expected
