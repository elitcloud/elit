# ========================================================================
# Copyright 2018 Emory University
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
from elit.lexicon import LabelMap, Word2Vec, FastText

__author__ = 'Jinho D. Choi'


# def test_label_map():
#     labels = ['A', 'B', 'C']
#     m = LabelMap()
#
#     for label in labels:
#         print(m.add(label))
#
#     for i, label in enumerate(labels):
#         print(m.index(label))
#
#     for label in labels:
#         m.add(label)
#
#     for i, label in enumerate(labels):
#         print(m.index(label))
#
#     for i in range(len(labels)):
#         print(m.get(i))


# def test_word2vec():
#     bin_file = '/Users/jdchoi/Documents/EmoryNLP/word2vec/sample.bin'
#     w2v = Word2Vec(bin_file)
#     print(w2v.dim)
#     print(w2v.get('Choi').dtype)
#     print(w2v.get('Ch').dtype)


# def test_fast_text():
#     bin_file = '/Users/jdchoi/Documents/EmoryNLP/fastText/sample.bin'
#     w2v = FastText(bin_file)
#     print(w2v.dim)
#     print(w2v.get('Choi').dtype)
#     print(w2v.get('Ch').dtype)
#
#
# # test_label_map()
# test_word2vec()
# test_fast_text()
