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

from elit.api import EnglishDecoder
from io import StringIO
from elit.configuration import *


__author__ = 'Jinho D. Choi'

input_text = 'I watched "The Sound of Music" last night. The ending could have been better. However, it is my favorite movie.'
config = Configuration(tokenize=TOKENIZE_DEFAULT, segment=SEGMENT_DEFAULT, sentiment=(SENTIMENT_TWITTER, SENTIMENT_MOVIE))
nd = EnglishDecoder(resource_dir='../../resources/', config=config)

# returns the output as a list of documents
config = Configuration(language=LANGUAGE_ENGLISH,
                       input_format=INPUT_FORMAT_RAW,
                       tokenize=TOKENIZE_DEFAULT,
                       segment=SEGMENT_DEFAULT,
                       sentiment=SENTIMENT_MOVIE_ATTENTION)
docs = nd.decode(config, StringIO(input_text))
print(docs)

# saves the output as a JSON file: out.json
fout = open('out.json', 'w')
nd.decode(config, StringIO(input_text), fout)

