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

from io import StringIO

from elit.decode import EnglishDecoder, DOC_DELIM
from elit.util.configure import Configuration, SENTIMENT_TWITTER, SENTIMENT_MOVIE, INPUT_FORMAT_RAW, INPUT_FORMAT_LINE

__author__ = 'Jinho D. Choi'

config = Configuration(tokenize=True, segment=True, sentiment=(SENTIMENT_MOVIE, SENTIMENT_TWITTER))
elit = EnglishDecoder(resource_dir='../resources/', config=config)
config.sentiment = ()

# returns the output as a list of documents
input_text = 'I watched "The Sound of Music" last night. The ending could have been better. It is my favorite movie though.'
config.input_format = INPUT_FORMAT_RAW
config.tokenize = False
config.segment = False
docs = elit.decode(config, StringIO(input_text))
print(docs)

config.input_format = INPUT_FORMAT_LINE
docs = elit.decode(config, StringIO(input_text))
print(docs)

config.input_format = INPUT_FORMAT_RAW
config.tokenize = True
docs = elit.decode(config, StringIO(input_text))
print(docs)

config.input_format = INPUT_FORMAT_LINE
docs = elit.decode(config, StringIO(input_text))
print(docs)

config.input_format = INPUT_FORMAT_RAW
config.segment = True
docs = elit.decode(config, StringIO(input_text))
print(docs)

config.input_format = INPUT_FORMAT_LINE
docs = elit.decode(config, StringIO(input_text))
print(docs)

input_text += '\n'+DOC_DELIM+'\nFourth sentence.?! Fifth sentence'
config.input_format = INPUT_FORMAT_RAW
docs = elit.decode(config, StringIO(input_text))
print(docs)

config.sentiment = (SENTIMENT_MOVIE, SENTIMENT_TWITTER)
with open('out.json', 'w') as fout:
    elit.decode(config, StringIO(input_text), fout)
