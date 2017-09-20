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
__author__ = 'Jinho D. Choi'

import json
from enum import Enum
import itertools
from io import StringIO
from elit.tokenizer import english_tokenizer


DOC_DELIM = '<@#DOC$%>'

FLAG_INPUT_FORMAT = 0
FLAG_DOCUMENT_SPLIT = 1
FLAG_TOKENIZATION = 1
FLAG_SEGMENTATION = 2
FLAG_SENTIMENT = 3


def to_dict(tokens, segments, sentiment):
    d = {'tokens': [token[0] for token in tokens],
         'offsets': [(token[1], token[2]) for token in tokens],
         'segments': segments, 'sentiment': sentiment}

    return d




class Language(Enum):
    English = 'en'


class NLPDecoder:
    def __init__(self, resource_dir, lang=Language.English):
        if lang == Language.English:
            english_tokenizer.init(resource_dir)
            self.tokenize = english_tokenizer.tokenize
            self.segment = english_tokenizer.segment
        else:
            raise ValueError('Unsupported language: '+str(lang))

    def read_document(self, stream, flag):



    def read_document(self, stream, flag):
        buf = []

        for line in stream:
            if line.strip():
                buf


                return line


        # 1: line
        if flag[FLAG_DOCUMENT_SPLIT] == '1':
            return self.red_next_line(stream)

        # 0: delim

    #
    #
    #
    #
    #
    #
    # def read_next_line(self, stream):
    #     for line in stream:
    #         line = line.strip()
    #         if line: return line
    #
    #     return None
    #
    # def decode_file(self, filename, flag='1111'):
    #     documents = []
    #
    #     if flag[FLAG_INPUT_FORMAT] == '0': # raw
    #         fin = open(filename, buffering=(2 << 16) + 8)
    #         documents.append()



    def decode_aux(self, text, flag='1111'):
        tokens = self.tokenize(text, flag[FLAG_TOKENIZATION] == '0')
        segments = self.segment(tokens) if flag[FLAG_SEGMENTATION] == '1' else [(0, len(tokens))]
        sentiment = '1'
        d = to_dict(tokens, segments, sentiment)

        js = json.dumps(d)
        print(str(js))

        return d





# import os
# print(os.getcwd())
# nd = NLPDecoder(resource_dir='/Users/jdchoi/workspace/elit/resources/tokenizer')
# nd.decode('Hello wrold! Ph.D. Choi is here.  How are you?')
# ./elit/resources/


# import io
# buf = io.StringIO('a\nb\nc')
# for i,line in enumerate(buf):
#     print(line.strip())
#     if (i == 1): break
#
# print('sdklfjsklfs')
# for i,line in enumerate(buf):
#     print(line.strip())
#     if (i == 1): break

# import sys
# fin = open('/Users/jdchoi/Documents/Data/experiments/general-en/dev/google_email.dev')
# for line in fin:
#     sys.stdout.write(line)
# fin.close()
# for line in buf:
#     sys.stdout.write(line)

import io
stream = io.StringIO('a\nb\n\nc\nd\ne')
l = list(itertools.takewhile(lambda x: x.strip(), [x for x in stream]))
print(str(l))
l = list(itertools.takewhile(lambda x: x.strip(), [x for x in stream]))
print(str(l))