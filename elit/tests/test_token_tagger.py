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
import pytest
import mxnet as mx
from elit.model import NLPModel
from elit.util.io import tsv_reader

__author__ = "Gary Lai"


def test_pos_token_tagger_config(pos_token_tagger_config):
    assert pos_token_tagger_config.reader == tsv_reader
    assert pos_token_tagger_config.log_path is None
    assert pos_token_tagger_config.key == 'pos'
    assert pos_token_tagger_config.sqeuence is False
    assert pos_token_tagger_config.chunking is False
    assert pos_token_tagger_config.feature_windows == [3, 2, 1, 0, -1, -2, -3]
    assert pos_token_tagger_config.position_embedding is False
    assert pos_token_tagger_config.label_embedding is False
    assert pos_token_tagger_config.input_dropout == 0.0
    assert pos_token_tagger_config.fuse_conv_config is None
    assert pos_token_tagger_config.ngrams_conv_config == NLPModel.namespace_ngram_conv_layer(
            ngrams=tuple([1, 2, 3, 4, 5]),
            filters=128,
            activation='relu',
            pool=None,
            dropout=0.2)
    assert pos_token_tagger_config.hidden_configs is None
    assert pos_token_tagger_config.context == mx.cpu(1)
    assert pos_token_tagger_config.core == 1