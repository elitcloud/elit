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
import os
import ujson
from elit.lexicon import Word2Vec
from elit.sentiment.sentiment_analyzer import SSTSentimentAnalyzer, SemEvalSentimentAnalyzer

__author__ = 'Jinho D. Choi'

import json
from enum import Enum
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
KEY_SENTIMENT_ATTENTION = 'sentiment_attention'

class Language(Enum):
    English = 'en'


class NLPDecoder:
    def __init__(self, resource_dir, lang=Language.English):
        if lang == Language.English:
            # tokenizer
            english_tokenizer.init(os.path.join(resource_dir, 'tokenizer'))
            self.tokenize = english_tokenizer.tokenize
            self.segment = english_tokenizer.segment

            # embedding
            self.twitter_emb = Word2Vec(os.path.join(resource_dir, 'embedding/w2v-400-twitter.gnsm'))
            self.amazon_emb = Word2Vec(os.path.join(resource_dir, 'embedding/w2v-400-amazon-review.gnsm'))

            # sentiment analyzers
            self.sentiment_semeval = SemEvalSentimentAnalyzer(self.twitter_emb, os.path.join(resource_dir, 'sentiment/sentiment-semeval17-400-v2'))
            self.sentiment_sst = SSTSentimentAnalyzer(self.amazon_emb, os.path.join(resource_dir, 'sentiment/sentiment-sst-400-v2'))
        else:
            raise ValueError('Unsupported language: ' + str(lang))

    ############################## DECODE ##############################

    def decode(self, flag, istream, ostream=None):
        """
        :param flag:
        :param istream: either StringIO or File
        :param ostream: either StringIO or File
        :return:
        """
        if ostream is not None:
            ostream.write('[')

        d = self.decode_raw(flag, istream, ostream) if flag[FLAG_INPUT_FORMAT] == '0' \
            else self.decode_line(flag, istream, ostream)

        if ostream is not None:
            ostream.write(']')
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

        if lines:
            decode()
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

        if sentences:
            decode()
        return documents

    ############################## CONVERSION ##############################

    def text_to_sentences(self, flag, text, offset=0):
        tokens = self.tokenize(text, flag[FLAG_TOKENIZATION] == '0')

        sentences = [self.tokens_to_sentence(flag, tokens, offset)] if flag[FLAG_SEGMENTATION] == '0' else \
                     self.tokens_to_sentences(flag, tokens, self.segment(tokens), offset)

        if flag[FLAG_SENTIMENT] != '0':
            self.sentiment_analyze(int(flag[FLAG_SENTIMENT]), sentences)

        return sentences

    def tokens_to_sentence(self, flag, tokens, offset):
        sentence = {KEY_FORMS: [token[0] for token in tokens],
                    KEY_OFFSETS: [(token[1] + offset, token[2] + offset) for token in tokens]}

        return sentence

    def tokens_to_sentences(self, flag, tokens, segments, offset):
        return [self.tokens_to_sentence(flag, tokens[segments[i]:segments[i + 1]], offset) for i in
                range(0, len(segments) - 1)]

    ############################## COMPONENTS ##############################

    def sentiment_analyze(self, flag, sentences):
        analyzer = self.sentiment_semeval if flag % 2 == 1 else self.sentiment_sst
        attn = flag > 2

        sens = [sentence[KEY_FORMS] for sentence in sentences]
        y, att, raw_att = analyzer.decode(sens, attn=attn)

        for i, sentence in enumerate(sentences):
            sentence[KEY_SENTIMENT] = y[i].tolist()
            if attn: sentence[KEY_SENTIMENT_ATTENTION] = att[i].tolist()


# from io import StringIO
# flag = '0114'
# input_text = 'This is a film well worth seeing, talking and singing heads and all. The sort of movie that gives tastelessness a bad rap. Marisa Tomei is good, but Just A Kiss is just a mess.'
# nd = NLPDecoder(resource_dir='/Users/jdchoi/workspace/elit/resources/')
# istream = StringIO(input_text)
# d = nd.decode(flag, istream, None)
# j = ujson.dumps(d)
# print(j)

# json_text = '[[{"forms":["This","is","an","example","of","the","raw","format","."],"offsets":[[0,4],[5,7],[8,10],[11,18],[19,21],[22,25],[27,30],[31,37],[37,38]]},{"forms":["It","assumes","no","segmentation","for","the","input","text","."],"offsets":[[39,41],[42,49],[50,52],[54,66],[67,70],[71,74],[75,80],[81,85],[85,86]]}]]'
# d = ujson.load(StringIO(json_text))
# print(d)
# print(type(d))