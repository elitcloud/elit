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
import pytest

__author__ = 'Jinho D. Choi'


@pytest.fixture(scope="module")
def space_tok():
    from elit.tokenizer import SpaceTokenizer
    return SpaceTokenizer()


space_tokenizer_test_data = [
    ('  Hello\t\r, world  !\n\n', 0,
     {'tok': ['Hello', ',', 'world', '!'], 'offset': [(2, 7), (9, 10), (11, 16), (18, 19)]}),
    ('  Hello\t\r, world  !\n\n', 1,
     {'tok': ['Hello', ',', 'world', '!'], 'offset': [(3, 8), (10, 11), (12, 17), (19, 20)]})
]


@pytest.mark.parametrize('input, offset, expected', space_tokenizer_test_data)
def test_pace_tokenizer(space_tok, input, offset, expected):
    result = space_tok.decode(input, offset)
    assert result['tok'] == expected['tok']
    assert result['offset'] == expected['offset']


@pytest.fixture(scope="module")
def eng_tok():
    from elit.tokenizer import EnglishTokenizer
    return EnglishTokenizer()


def test_eng_tokenizer_load(eng_tok):
    assert len(eng_tok.ABBREVIATION_PERIOD) == 159
    assert len(eng_tok.APOSTROPHE_FRONT) == 7
    assert len(eng_tok.HYPHEN_PREFIX) == 156
    assert len(eng_tok.HYPHEN_SUFFIX) == 39


@pytest.mark.parametrize('input, expected', [
    ('', {'tok': [], 'offset': []}),
    (' \t\n\r', {'tok': [], 'offset': []}),
    ('abc', {'tok': ['abc'], 'offset': [(0, 3)]}),
    (' A\tBC  D12 \n', {'tok': ['A', 'BC', 'D12'],
                        'offset': [(1, 2), (3, 5), (7, 10)]})
])
def test_eng_tokenizer_space(eng_tok, input, expected):
    result = eng_tok.decode(input)
    assert result['tok'] == expected['tok']
    assert result['offset'] == expected['offset']


@pytest.mark.parametrize('input, offset, expected', [
    ('Hello, world!', 5,  {'tok': ['Hello', ',', 'world', '!'], 'offset': [
     (5, 10), (10, 11), (12, 17), (17, 18)]}),
])
def test_eng_tokenizer_offset(eng_tok, input, offset, expected):
    result = eng_tok.decode(input, offset)
    assert result['tok'] == expected['tok']
    assert result['offset'] == expected['offset']


@pytest.mark.parametrize('input, expected', [
    # html entity
    ('ab&larr;cd&#8592;&#x2190;ef&rarr;',
     {'tok': ['ab', '&larr;', 'cd', '&#8592;', '&#x2190;', 'ef', '&rarr;'],
      'offset': [(0, 2), (2, 8), (8, 10), (10, 17), (17, 25), (25, 27), (27, 33)]
      }),
    # email address
    ('a;jinho@elit.com,b;jinho.choi@elit.com,choi@elit.emory.edu,jinho:choi@0.0.0.0',
     {'tok': ['a', ';', 'jinho@elit.com', ',', 'b', ';', 'jinho.choi@elit.com', ',',
              'choi@elit.emory.edu', ',', 'jinho:choi@0.0.0.0'],
      'offset': [(0, 1), (1, 2), (2, 16), (16, 17), (17, 18), (18, 19), (19, 38), (38, 39),
                 (39, 58), (58, 59), (59, 77)]
      }),
    # hyperlink
    ('a:http://ab sftp://ef abeftp://',
     {'tok': ['a', ':', 'http://ab', 'sftp://ef', 'abe', 'ftp://'],
      'offset': [(0, 1), (1, 2), (2, 11), (12, 21), (22, 25), (25, 31)]
      }),
    # emoticon
    (':-) A:-( :). B:smile::sad: C:):(! :)., :-))) :---( Hi:).',
     {'tok': [':-)', 'A', ':-(', ':)', '.', 'B', ':smile:', ':sad:', 'C', ':)', ':(', '!', ':)',
              '.', ',', ':-)))', ':---(', 'Hi', ':)', '.'],
      'offset': [(0, 3), (4, 5), (5, 8), (9, 11), (11, 12), (13, 14), (14, 21), (21, 26),
                 (27, 28), (28, 30), (30, 32), (32,
                                                33), (34, 36), (36, 37), (37, 38),
                 (39, 44), (45, 50), (51, 53), (53, 55), (55, 56)]
      }),
    # list item
    ('[a](1)(1.a)[11.22.a.33](A.1)[a1][hello]{22}',
     {'tok': ['[a]', '(1)', '(1.a)', '[11.22.a.33]', '(A.1)', '[a1]', '[', 'hello', ']', '{22}'],
      'offset': [(0, 3), (3, 6), (6, 11), (11, 23), (23, 28), (28, 32), (32, 33), (33, 38),
                 (38, 39), (39, 43)]
      }),
    ("don't he's can't does'nt 0's DON'T ab'cd",
     {'tok': ['do', "n't", 'he', "'s", 'ca', "n't", 'does', "'nt", "0's", 'DO', "N'T",
              "ab'cd"],
      'offset': [(0, 2), (2, 5), (6, 8), (8, 10), (11, 13), (13, 16), (17, 21), (21, 24),
                 (25, 28), (29, 31), (31, 34), (35, 40)]
      }),
])
def test_eng_tokenizer_regex(eng_tok, input, expected):
    result = eng_tok.decode(input)
    assert result['tok'] == expected['tok']
    assert result['offset'] == expected['offset']


@pytest.mark.parametrize('input, expected', [
    # handle symbols with digits
    (".1 +1 -1 1,000,000.00 1,00 '97 '90s '1990 10:30 1:2 a:b",
     {'tok': ['.1', '+1', '-1', '1,000,000.00', '1', ',', '00', "'97", "'90s", "'",
                          '1990', '10:30', '1:2', 'a', ':', 'b'],
      'offset': [(0, 2), (3, 5), (6, 8), (9, 21), (22, 23), (23, 24), (24, 26), (27, 30),
                 (31, 35), (36, 37), (37, 41), (42,
                                                47), (48, 51), (52, 53), (53, 54),
                 (54, 55)]
      }),
    # separator and edge symbols
    ("aa;;;b: c\"\"d\"\" 0'''s ''a''a'' 'a'a'a'.?!..",
     {'tok': ['aa', ';;;', 'b', ':', 'c', '""', 'd', '""', '0', "'''", 's', "''", 'a',
              "''", 'a', "''", "'", "a'a'a", "'", '.?!..'],
      'offset': [(0, 2), (2, 5), (5, 6), (6, 7), (8, 9), (9, 11), (11, 12), (12, 14),
                 (15, 16), (16, 19), (19, 20), (21,
                                                23), (23, 24), (24, 26), (26, 27),
                 (27, 29), (30, 31), (31, 36), (36, 37), (37, 42)]
      }),
    # currency like
    ('#happy2018,@Jinho_Choi: ab@cde #1 $1 ',
     {'tok': ['#happy2018', ',', '@Jinho_Choi', ':', 'ab@cde', '#', '1', '$', '1'],
      'offset': [(0, 10), (10, 11), (11, 22), (22, 23), (24, 30), (31, 32), (32, 33),
                 (34, 35), (35, 36)]
      }),
    # hyphen
    ("1997-2012,1990's-2000S '97-2012 '12-14's",
     {'tok': ['1997', '-', '2012', ',', "1990's", '-', '2000S', "'97", '-', '2012',
                      "'12", '-', "14's"],
      'offset': [(0, 4), (4, 5), (5, 9), (9, 10), (10, 16), (16, 17), (17, 22), (23, 26),
                 (26, 27), (27, 31), (32, 35), (35, 36), (36, 40)]
      })
])
def test_eng_tokenizer_symbol(eng_tok, input, expected):
    result = eng_tok.decode(input)
    assert result['tok'] == expected['tok']
    assert result['offset'] == expected['offset']


@pytest.mark.parametrize('input, expected', [
    # apostrophe for abbreviation
    ("'CAUSE 'tis ' em 'happy", {
        'tok': ["'CAUSE", "'tis", "'em", "'", 'happy'],
        'offset': [(0, 6), (7, 11), (12, 16), (17, 18), (18, 23)]
    }),
    # abbreviation
    ('A.B. a.B.c AB.C. Ph.D. 1.2. A-1.', {
        'tok': ['A.B.', 'a.B.c', 'AB.C', '.', 'Ph.D.', '1.2.', 'A-1.'],
        'offset': [(0, 4), (5, 10), (11, 15), (15, 16), (17, 22), (23, 27), (28, 32)]
    }),
    # acronym
    ('ab&cd AB|CD 1/2 a-1 1-2', {
        'tok': ['ab&cd', 'AB|CD', '1/2', 'a-1', '1-2'],
        'offset': [(0, 5), (6, 11), (12, 15), (16, 19), (20, 23)]
    }),
    # hyphenated
    ('mis-predict mic-predict book - able book-es 000-0000 000-000-000 p-u-s-h-1-2', {
        'tok': ['mis-predict', 'mic', '-', 'predict', 'book-able', 'book', '-', 'es', '000-0000', '000-000-000', 'p-u-s-h-1-2'],
        'offset': [(0, 11), (12, 15), (15, 16), (16, 23), (24, 35), (36, 40), (40, 41), (41, 43), (44, 52), (53, 64), (65, 76)]
    }),
    # No. 1
    ('No. 1 No. a No.', {
        'tok': ['No.', '1', 'No', '.', 'a', 'No', '.'],
        'offset': [(0, 3), (4, 5), (6, 8), (8, 9), (10, 11), (12, 14), (14, 15)]
    })
])
def test_eng_tokenizer_concat(eng_tok, input, expected):
    result = eng_tok.decode(input)
    assert result['tok'] == expected['tok']
    assert result['offset'] == expected['offset']


@pytest.mark.parametrize('input, expected', [
    # unit
    ('20mg 100cm 1st 11a.m. 10PM', {
        'tok': ['20', 'mg', '100', 'cm', '1st', '11', 'a.m.', '10', 'PM'],
        'offset':[(0, 2), (2, 4), (5, 8), (8, 10), (11, 14), (15, 17), (17, 21), (22, 24), (24, 26)]
    }),
    # concatenated word
    ("whadya DON'CHA", {
        'tok': ['wha', 'd', 'ya', 'DO', "N'", 'CHA'],
        'offset':[(0, 3), (3, 4), (4, 6), (7, 9), (9, 11), (11, 14)]
    }),
    # final mark
    ('Mbaaah.Please hello.!?world', {
        'tok': ['Mbaaah', '.', 'Please', 'hello', '.!?', 'world'],
        'offset':[(0, 6), (6, 7), (7, 13), (14, 19), (19, 22), (22, 27)]
    }),
])
def test_eng_tokenizer_split(eng_tok, input, expected):
    result = eng_tok.decode(input)
    assert result['tok'] == expected['tok']
    assert result['offset'] == expected['offset']


@pytest.fixture(scope="module")
def eng_seg():
    from elit.segmenter import EnglishSegmenter
    return EnglishSegmenter()


@pytest.mark.parametrize('input, expected', [
    ('hello world',
     [{
         'sid': 0,
         'tok': ['hello', 'world'],
         'offset': [(0, 5), (6, 11)]
     }]
    ),
    # unit
    ('. "1st sentence." 2nd sentence? "3rd sentence!"',
     [{
         'sid': 0,
         'tok': ['.', '"', '1st', 'sentence', '.', '"'],
         'offset': [(0, 1), (2, 3), (3, 6), (7, 15), (15, 16), (16, 17)]
     }, {
         'sid': 1,
         'tok': ['2nd', 'sentence', '?'],
         'offset': [(18, 21), (22, 30), (30, 31)]
     }, {
         'sid': 2,
         'tok': ['"', '3rd', 'sentence', '!', '"'],
         'offset': [(32, 33), (33, 36), (37, 45), (45, 46), (46, 47)]
     }])
])
def test_eng_seg(eng_tok, eng_seg, input, expected):
    result = eng_tok.decode(input)
    print(result['tok'])
    assert eng_seg.decode(result['tok'], result['offset']) == expected
