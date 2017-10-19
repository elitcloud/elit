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
import os
import json
from elit.lexicon import Word2Vec
from elit.configuration import *
from elit.sentiment.sentiment_analyzer import MovieSentimentAnalyzer, TwitterSentimentAnalyzer
from elit.tokenizer import english_tokenizer


__author__ = 'Jinho D. Choi'


DOC_MAX_SIZE = 10485760
DOC_DELIM = '@#DOC$%'

KEY_TOKENS = 'tokens'
KEY_OFFSETS = 'offsets'
KEY_SENTIMENT = 'sentiment'
KEY_SENTIMENT_ATTENTION = 'sentiment_attention'


class Decoder:
    def decode(self, config, istream, ostream=None):
        """
        :param config: elit.configuration.Configuration
        :param istream: either StringIO or File
        :param ostream: either StringIO or File
        :return:
        """
        if ostream is not None: ostream.write('[')

        d = self.decode_raw(config, istream, ostream) if config.is_input_format(INPUT_FORMAT_RAW) \
            else self.decode_line(config, istream, ostream)

        if ostream is not None: ostream.write(']')
        return d

    def decode_raw(self, config, istream, ostream=None):
        def decode():
            d = self.text_to_sentences(config, ''.join(lines))
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

    def decode_line(self, config, istream, ostream=None):
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
                d = self.text_to_sentences(config, line, offset)
                offset += len(line)
                sentences.extend(d)

        if sentences: decode()
        return documents

    ############################## CONVERSION ##############################

    @abc.abstractmethod
    def text_to_sentences(self, config, text, offset=0):
        return

    def tokens_to_sentences(self, config, tokens, segments, offset):
        return [self.tokens_to_sentence(config, tokens[segments[i]:segments[i + 1]], offset) for i in
                range(0, len(segments) - 1)]

    def tokens_to_sentence(self, config, tokens, offset):
        sentence = {KEY_TOKENS: [token[0] for token in tokens],
                    KEY_OFFSETS: [(token[1] + offset, token[2] + offset) for token in tokens]}

        return sentence


class EnglishDecoder(Decoder):
    def __init__(self, resource_dir, config):
        # init tokenizer
        if config.is_tokenize(TOKENIZE_DEFAULT):
            english_tokenizer.init(os.path.join(resource_dir, 'tokenizer'))
            self.tokenize = english_tokenizer.tokenize
            self.segment = english_tokenizer.segment

        # init sentiment analyzer: twitter
        if config.is_sentiment(SENTIMENT_TWITTER):
            self.emb_twit = Word2Vec(os.path.join(resource_dir, 'embedding/w2v-400-twitter.gnsm'))
            self.sentiment_twit = TwitterSentimentAnalyzer(
                self.emb_twit, os.path.join(resource_dir, 'sentiment/sentiment-semeval17-400-v2'))

        # init sentiment analyzer: movie review
        if config.is_sentiment(SENTIMENT_MOVIE):
            self.emb_mov = Word2Vec(os.path.join(resource_dir, 'embedding/w2v-400-amazon-review.gnsm'))
            self.sentiment_mov = MovieSentimentAnalyzer(
                self.emb_mov, os.path.join(resource_dir, 'sentiment/sentiment-sst-400-v2'))

    ############################## CONVERSION ##############################

    def text_to_sentences(self, config, text, offset=0):
        tokens = self.tokenize(text, config.is_tokenize(FLAG_FALSE))

        sentences = [self.tokens_to_sentence(config, tokens, offset)] if config.is_segment(FLAG_FALSE) else \
                     self.tokens_to_sentences(config, tokens, self.segment(tokens), offset)

        if not config.is_sentiment(FLAG_FALSE):
            self.sentiment_analyze(config, sentences)

        return sentences

    ############################## COMPONENTS ##############################

    def sentiment_analyze(self, config, sentences):
        analyzer = self.sentiment_twit if config.sentiment.startswith(SENTIMENT_TWITTER) else self.sentiment_mov
        attn = config.is_sentiment(SENTIMENT_TWITTER_ATTENTION) or config.is_sentiment(SENTIMENT_MOVIE_ATTENTION)

        sens = [sentence[KEY_TOKENS] for sentence in sentences]
        y, att, raw_att = analyzer.decode(sens, attn=attn)

        for i, sentence in enumerate(sentences):
            sentence[KEY_SENTIMENT] = y[i].tolist()
            if attn: sentence[KEY_SENTIMENT_ATTENTION] = att[i].tolist()