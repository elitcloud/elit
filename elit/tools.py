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
from io import StringIO
from elit.tokenizer import english_tokenizer

DOC_MAX_SIZE = 10485760
DOC_DELIM = '@#DOC$%'

FLAG_INPUT_FORMAT = 0
FLAG_TOKENIZATION = 1
FLAG_SEGMENTATION = 2
FLAG_SENTIMENT = 3

KEY_FORMS = 'forms'
KEY_OFFSETS = 'offsets'
KEY_SENTIMENT = 'sentiment'


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

    ############################## DECODE ##############################

    def decode(self, flag, istream, ostream=None):
        """
        :param flag:
        :param istream: either StringIO or File
        :param ostream: either StringIO or File
        :return:
        """
        if ostream is not None: ostream.write('[')

        d = self.decode_raw(flag, istream, ostream) if flag[FLAG_INPUT_FORMAT] == '0' \
                                                    else self.decode_line(flag, istream, ostream)

        if ostream is not None: ostream.write(']')
        return d

    def decode_raw(self, flag, istream, ostream=None):
        def decode():
            d = self.text_to_sentences(flag, ''.join(lines))
            if ostream is None:
                documents.append(d)
            else:
                ostream.write(str(json.dumps(d)) + ',')

        documents = []
        offset = 0
        lines = []

        for line in istream:
            if line.strip() == DOC_DELIM:
                decode()
                offset = 0
                lines.clear()
            elif offset + len(line) <= DOC_MAX_SIZE:
                offset += len(line)
                lines.append(line)

        if lines: decode()
        return documents

    def decode_line(self, flag, istream, ostream=None):
        def decode():
            if ostream is None:
                documents.append(sentences)
            else:
                ostream.write(str(json.dumps(sentences)) + ',')

        documents = []
        sentences = []
        offset = 0

        for line in istream:
            if line.strip() == DOC_DELIM:
                decode()
                offset = 0
                sentences = []
            elif offset + len(line) <= DOC_MAX_SIZE:
                d = self.text_to_sentences(flag, line, offset)
                offset += len(line)
                sentences.extend(d)

        if sentences: decode()
        return documents

    ############################## CONVERSION ##############################

    def text_to_sentences(self, flag, text, offset=0):
        tokens = self.tokenize(text, flag[FLAG_TOKENIZATION] == '0')

        if flag[FLAG_SEGMENTATION] == '0':
            return [self.tokens_to_sentence(flag, tokens, offset)]

        return self.tokens_to_sentences(flag, tokens, self.segment(tokens), offset)

    def tokens_to_sentences(self, flag, tokens, segments, offset):
        return [self.tokens_to_sentence(flag, tokens[segments[i]:segments[i+1]], offset) for i in range(0, len(segments)-1)]

    def tokens_to_sentence(self, flag, tokens, offset):
        sentence = {KEY_FORMS: [token[0] for token in tokens],
                    KEY_OFFSETS: [(token[1] + offset, token[2] + offset) for token in tokens]}

        if flag[FLAG_SENTIMENT] == '1': self.sentiment_analyze(sentence)
        return sentence

    ############################## COMPONENTS ##############################

    def sentiment_analyze(self, sentence):
        sentence[KEY_SENTIMENT] = 1.0







# import os
# print(os.getcwd())
# nd = NLPDecoder(resource_dir='/Users/jdchoi/workspace/elit/resources/tokenizer')
# flag = '0110'
# input_text = 'This is an example of the\n raw format. It assumes no\n segmentation for the input text.'
# input_text = 'The first sentence in the first segment.\nThe second sentence in the first segment\nThe third sentence in the second segment.'
# input_text = 'This is the first document.\nContents of the first document are here.\n@#DOC$%\nThis is the second document.\nThe delimiter is not required for the last document.'
#
# istream = StringIO(input_text)
# ostream = open('tmp.json', 'w')
# d = nd.decode(flag, istream, ostream)
# j = json.dumps(d)
# print(j)
