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


from elit.util.io import tsv_cols, tsv_reader, json_reader

__author__ = "Gary Lai"

from pkg_resources import resource_filename


def test_tsv_cols():
    tsv_heads = [['tok', 0], ['pos', 1]]
    assert tsv_cols(tsv_heads) == {'tok': 0, 'pos': 1}


def test_tsv_reader():
    tsv_heads = [['tok', 0], ['pos', 1]]
    docs, label_map = tsv_reader(
        tsv_directory=resource_filename('elit.tests.resources', 'tsv/pos/trn'),
        cols=tsv_cols(tsv_heads),
        key='pos'
    )
    print(docs)
    assert docs == [{'sen': [
         {'tok': ['John', 'bought', 'a', 'car', '.'], 'pos-gold': ['NNP', 'VBD', 'DT', 'NN', '.'],
          'sen_id': 0},
         {'tok': ['A', 'boy', 'is', 'here', '?'], 'pos-gold': ['DT', 'NN', 'VBZ', 'RB', '.'],
          'sen_id': 1}],
        'doc_id': 0}]
    assert len(label_map) == 7
    docs, label_map = tsv_reader(
        tsv_directory=resource_filename('elit.tests.resources', 'tsv/pos/dev'),
        cols=tsv_cols(tsv_heads),
        key='pos'
    )
    assert docs == [{'sen': [
        {'tok': ['Mary', 'sold', 'a', 'truck', '.'], 'pos-gold': ['NNP', 'VBD', 'DT', 'NN', '.'],
         'sen_id': 0},
        {'tok': ['The', 'girl', 'is', 'there', '!'], 'pos-gold': ['DT', 'NN', 'VBZ', 'RB', '.'],
         'sen_id': 1}],
        'doc_id': 0}]
    assert len(label_map) == 7
    tsv_heads = [['tok', 0], ['ner', 1]]
    docs, label_map = tsv_reader(
        tsv_directory=resource_filename('elit.tests.resources', 'tsv/ner/trn'),
        cols=tsv_cols(tsv_heads),
        key='ner'
    )
    assert docs == [{'sen': [
        {'tok': ['John', 'bought', 'a', 'car', '.'], 'ner-gold': ['U-PER', 'O', 'B-IT', 'L-IT', '.'],
         'sen_id': 0},
        {'tok': ['A', 'boy', 'is', 'here', '?'], 'ner-gold': ['O', 'O', 'B-P', 'I-P', 'L-P'],
         'sen_id': 1}],
        'doc_id': 0}]
    assert len(label_map) == 8
    tsv_heads = [['tok', 0], ['ner', 1]]
    docs, label_map = tsv_reader(
        tsv_directory=resource_filename('elit.tests.resources', 'tsv/ner/dev'),
        cols=tsv_cols(tsv_heads),
        key='ner'
    )
    assert docs == [{'sen': [
        {'tok': ['Mary', 'sold', 'a', 'truck', '.'],
         'ner-gold': ['U-PER', 'O', 'B-IT', 'L-IT', '.'],
         'sen_id': 0},
        {'tok': ['The', 'girl', 'is', 'there', '!'],
         'ner-gold': ['O', 'O', 'B-P', 'I-P', 'L-P'],
         'sen_id': 1}],
        'doc_id': 0}]
    assert len(label_map) == 8


def test_json_reader():
    docs, label_map = json_reader(
        filepath=resource_filename('elit.tests.resources', 'json/pos/trn/sample.json'),
        key='pos'
    )
    assert docs == [{'sen': [
         {'tok': ['John', 'bought', 'a', 'car', '.'], 'pos-gold': ['NNP', 'VBD', 'DT', 'NN', '.'],
          'sen_id': 0},
         {'tok': ['A', 'boy', 'is', 'here', '?'], 'pos-gold': ['DT', 'NN', 'VBZ', 'RB', '.'],
          'sen_id': 1}],
        'doc_id': 0}]
    assert len(label_map) == 7
    docs, label_map = json_reader(
        filepath=resource_filename('elit.tests.resources', 'json/pos/dev/sample.json'),
        key='pos'
    )
    assert docs == [{'sen': [
        {'tok': ['Mary', 'sold', 'a', 'truck', '.'], 'pos-gold': ['NNP', 'VBD', 'DT', 'NN', '.'],
         'sen_id': 0},
        {'tok': ['The', 'girl', 'is', 'there', '!'], 'pos-gold': ['DT', 'NN', 'VBZ', 'RB', '.'],
         'sen_id': 1}],
        'doc_id': 0}]
    assert len(label_map) == 7
    docs, label_map = json_reader(
        filepath=resource_filename('elit.tests.resources', 'json/ner/trn/sample.json'),
        key='ner'
    )
    assert docs == [{'sen': [
        {'tok': ['John', 'bought', 'a', 'car', '.'], 'ner-gold': ['U-PER', 'O', 'B-IT', 'L-IT', '.'],
         'sen_id': 0},
        {'tok': ['A', 'boy', 'is', 'here', '?'], 'ner-gold': ['O', 'O', 'B-P', 'I-P', 'L-P'],
         'sen_id': 1}],
        'doc_id': 0}]
    assert len(label_map) == 8
    docs, label_map = json_reader(
        filepath=resource_filename('elit.tests.resources', 'json/ner/dev/sample.json'),
        key='ner'
    )
    assert docs == [{'sen': [
        {'tok': ['Mary', 'sold', 'a', 'truck', '.'],
         'ner-gold': ['U-PER', 'O', 'B-IT', 'L-IT', '.'],
         'sen_id': 0},
        {'tok': ['The', 'girl', 'is', 'there', '!'],
         'ner-gold': ['O', 'O', 'B-P', 'I-P', 'L-P'],
         'sen_id': 1}],
        'doc_id': 0}]
    assert len(label_map) == 8
