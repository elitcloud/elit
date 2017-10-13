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

from elit.api import ELITDecoder
from io import StringIO

__author__ = 'Jinho D. Choi'

flag = '0114'
input_text = 'I watched "The Sound of Music" last night. It is my favorite movie.'
nd = ELITDecoder(resource_dir='../../resources/', sentiment_twitter=False)

# returns the output as a list of documents
docs = nd.decode(flag, StringIO(input_text))
print(docs)

# saves the output as a JSON file: out.json
fout = open('out.json', 'w')
nd.decode(flag, StringIO(input_text), fout)

