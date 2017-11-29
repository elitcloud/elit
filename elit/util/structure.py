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

TOKEN = 'word'
LEMMA = 'lemma'
POS = 'pos'
NER = 'ner'
DEP = 'dep'

OFFSET = 'offset'
SENTIMENT = 'sentiment'



# part-of-speech tagging

# named entity recognition


class Sentence(dict):
    def __init__(self, d=None):
        """
        :param d: a dictionary containing fields for the sentence.
        :type d: dict
        """
        super(Sentence, self).__init__()
        if d is not None: self.update(d)

    def __len__(self):
        return len(self[TOKEN])


