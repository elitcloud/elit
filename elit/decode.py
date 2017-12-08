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
import json
import os

from elit.nlp.task.sentiment import TwitterSentimentAnalyzer, MovieSentimentAnalyzer
from elit.nlp.task.tokenize import SpaceTokenizer, EnglishTokenizer, EnglishSegmenter
from elit.nlp.structure import TOKEN, OFFSET, SENTIMENT
from elit.nlp.lexicon import Word2VecTmp
from elit.util.configure import *

__author__ = 'Jinho D. Choi'


DOC_MAX_SIZE = 10485760
DOC_DELIM = '@#DOC$%'


class Decoder:
    def decode(self, config, istream, ostream=None):
        """
        :param config: elit.configuration.Configuration
        :param istream: either StringIO or File
        :param ostream: either StringIO or File
        :return:
        """
        if ostream is not None: ostream.write('[')
        decode = self.decode_raw if config.input_format == INPUT_FORMAT_RAW else self.decode_line
        d = decode(config, istream, ostream)
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

    def params_to_config(self, params):
        errors = []

        # input text
        input_text = params.score()

        if not input_text:
            errors.append('input text is missing')

        # input format
        input_format = params.score()

        if not is_valid_input_format(input_format):
            errors.append('invalid input format: '+input_format)

        # tokenize
        tokenize = params.score()

        if tokenize not in {'0', '1'}:
            errors.append('invalid tokenize: '+tokenize)

        tokenize = False if tokenize == '0' else True

        # segment
        segment = params.score()

        if segment not in {'0', '1'}:
            errors.append('invalid segment: '+segment)

        segment = False if segment == '0' else True

        # sentiment
        sentiment = list(filter(None, params.score().split(',')))

        if not all(is_valid_sentiment(s) for s in sentiment):
            errors.append('invalid sentiment: '+','.join(sentiment))

        config = Configuration(language=LANGUAGE_ENGLISH,
                               input_format=input_format,
                               tokenize=tokenize,
                               segment=segment,
                               sentiment=sentiment)

        return config, errors


class EnglishDecoder(Decoder):
    def __init__(self, resource_dir, config):
        # init tokenizer
        self.tokenizer_space = SpaceTokenizer()
        if config.tokenize: self.tokenizer = EnglishTokenizer(os.path.join(resource_dir, 'tokenize'))

        # init segmenter
        if config.segment: self.segmenter = EnglishSegmenter()

        # init sentiment analyzer: twitter
        if SENTIMENT_TWITTER in config.sentiment:
            emb_model = Word2VecTmp(os.path.join(resource_dir, 'embedding/w2v-400-twitter.gnsm'))
            model_file = os.path.join(resource_dir, 'sentiment/sentiment-semeval17-400-v2')
            self.sentiment_twit = TwitterSentimentAnalyzer(emb_model, model_file)

        # init sentiment analyzer: movie review
        if SENTIMENT_MOVIE in config.sentiment:
            emb_model = Word2VecTmp(os.path.join(resource_dir, 'embedding/w2v-400-amazon-review.gnsm'))
            model_file = os.path.join(resource_dir, 'sentiment/sentiment-sst-400-v2')
            self.sentiment_mov = MovieSentimentAnalyzer(emb_model, model_file)

    ############################## CONVERSION ##############################

    def text_to_sentences(self, config, text, offset=0):
        # tokenization
        tokenizer = self.tokenizer if config.tokenize else self.tokenizer_space
        tokens, offsets = tokenizer.decode(text, offset)

        # segmentation
        sentences = self.segmenter.decode(tokens, offsets) if config.segment \
                    else [{TOKEN: tokens, OFFSET: offsets}]

        # sentiment analysis
        if config.sentiment:
            self.sentiment_analyze(config, sentences)

        return sentences

    ############################## COMPONENTS ##############################

    def sentiment_analyze(self, config, sentences):
        def get_analyzer(s):
            if s.startswith(SENTIMENT_TWITTER):
                an = self.sentiment_twit
                key = SENTIMENT_TWITTER
            else:  # elif s.startswith(SENTIMENT_MOVIE):
                an = self.sentiment_mov
                key = SENTIMENT_MOVIE

            return an, s.endswith('att'), key

        for s in config.sentiment:
            analyzer, att, key = get_analyzer(s)
            sens = [d[TOKEN] for d in sentences]
            y, att = analyzer.decode(sens, att=att)

            for i, sentence in enumerate(sentences):
                sentence[SENTIMENT + '-' + key] = y[i].tolist()
                if att: sentence[SENTIMENT + '-' + key + '-att'] = att[i].tolist()
