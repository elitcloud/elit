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


import unittest
import os
from elit.nlp.lexicon import NamedEntityTree

__author__ = 'Jose Coves'


class EntityVectorTest(unittest.TestCase):
    def setUp(self):
        self.text = "Donald Trump New York City NY Michael Jordan Emory University 07 stanley cup final African American".split(' ')
        self.filepath = "/mnt/ainos-research/emorynlp/lexica/named_entity_gazetteers"
        self.names = []

    def test_entity_vectors(self):
        entityTree = NamedEntityTree(self.filepath)
        for i, filename in enumerate(sorted(os.listdir(self.filepath))):
            self.names.append(filename)
        # vectors = entityTree.get_entity_vectors(self.text)
        vectors = entityTree.get_expanded_entity_vectors(self.text)
        print('text = ', self.text)
        for i, vector in enumerate(vectors):
            print('\n', self.text[i], ":")
            for j, val in enumerate(vector):
                if val > 0.0:
                    print(self.names[j] if len(vector) == len(os.listdir(self.filepath)) else self.names[int(j/3)], "=", val)
        print("\n__Done__")

if __name__ == '__main__':
    unittest.main()
