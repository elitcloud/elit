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
from enum import Enum

__author__ = 'Jinho D. Choi'

# constants
FLAG_FALSE = '0'

LANGUAGE_ENGLISH = 'en'

INPUT_FORMAT_RAW = 'raw'
INPUT_FORMAT_LINE = 'line'

TOKENIZE_DEFAULT = '1'
SEGMENT_DEFAULT = '1'

SENTIMENT_TWITTER = 'twit'
SENTIMENT_MOVIE = 'mov'
SENTIMENT_TWITTER_ATTENTION = 'twit-att'
SENTIMENT_MOVIE_ATTENTION = 'mov-att'


class Configuration:
    def __init__(self,
                 language=LANGUAGE_ENGLISH,
                 input_format=INPUT_FORMAT_RAW,
                 tokenize=TOKENIZE_DEFAULT,
                 segment=SEGMENT_DEFAULT,
                 sentiment=SENTIMENT_TWITTER):
        self.language = language
        self.input_format = input_format
        self.tokenize = tokenize
        self.segment = segment
        self.sentiment = sentiment

    def is_language(self, language):
        return self.language == language

    def is_input_format(self, format):
        return self.input_format == format

    def is_tokenize(self, flag):
        return self.tokenize == flag

    def is_segment(self, flag):
        return self.sentiment == flag

    def is_sentiment(self, flag):
        return self.sentiment == flag if isinstance(self.sentiment, str) else flag in self.sentiment
