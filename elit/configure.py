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

# flag
FLAG_FALSE = '0'
FLAG_TRUE = '1'

# language
LANGUAGE_ENGLISH = 'en'

# input format
INPUT_FORMAT_RAW = 'raw'
INPUT_FORMAT_LINE = 'line'

# sentiment analysis
SENTIMENT_TWITTER = 'twit'
SENTIMENT_MOVIE = 'mov'
SENTIMENT_TWITTER_ATT = 'twit-att'
SENTIMENT_MOVIE_ATT = 'mov-att'


class Configuration:
    def __init__(self,
                 language=LANGUAGE_ENGLISH,
                 input_format=INPUT_FORMAT_RAW,
                 tokenize=FLAG_TRUE,
                 segment=FLAG_TRUE,
                 sentiment=(SENTIMENT_TWITTER, SENTIMENT_MOVIE)):
        self.language = language
        self.input_format = input_format
        self.tokenize = tokenize
        self.segment = segment
        self.sentiment = sentiment
