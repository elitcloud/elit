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


# from elit.dev.template.lexicon import NLPLexiconMapper
# # from fasttext.model import WordVectorModel
# from gensim.models.keyedvectors import KeyedVectors
#
# import unittest
# import os
#
# __author__ = 'Jinho D. Choi'
#
# class LexiconTest(unittest.TestCase):
#
#     def setUp(self):
#         self.w2v_file = os.path.join(os.path.abspath(os.path.dirname(__file__)),
#                                      './../elit/resources/sample/sample.w2v.bin')
#         self.word_embeddings = KeyedVectors.load_word2vec_format(self.w2v_file, binary=True)
#
#
#     def test_lexicon(self):
#         lexicon = NLPLexiconMapper(self.word_embeddings)
#         print(lexicon.w2v)
#
# if __name__ == '__main__':
#     unittest.main()
