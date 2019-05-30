# ========================================================================
# Copyright 2018 ELIT
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
import mxnet as mx
from elit.util.mx import mx_loss
from types import SimpleNamespace

from elit.util.reader import tsv_reader, json_reader

__author__ = "Gary Lai"


class TokenTaggerConfig(object):

    def __init__(self, source: dict):
        self.source = source

    def __str__(self):
        return str(self.source)

    @property
    def reader(self):
        return tsv_reader if self.source.get('reader', 'tsv').lower() == 'tsv' else json_reader

    @property
    def log_path(self):
        return self.source.get('log_path', None)

    @property
    def tsv_heads(self):
        return self.source['tsv_heads'] if self.source.get('reader', 'tsv').lower() == 'tsv' else None

    @property
    def chunking(self):
        return self.source.get('chunking', False)

    @property
    def feature_windows(self):
        return self.source.get('feature_windows', [3, 2, 1, 0, -1, -2, -3])

    @property
    def position_embedding(self):
        return self.source.get('position_embedding', False)

    @property
    def label_embedding(self):
        return self.source.get('label_embedding', False)

    @property
    def ctx(self):
        device = self.source.get('device', 'cpu')
        core = self.source.get('core', 0)
        if device.lower() == 'cpu':
            return mx.cpu()
        else:
            return mx.gpu(core)

    @property
    def epoch(self):
        return self.source.get('epoch', 100)

    @property
    def batch_size(self):
        return self.source.get('batch_size', 128)

    @property
    def trn_batch(self):
        return self.source.get('trn_batch', 64)

    @property
    def dev_batch(self):
        return self.source.get('dev_batch', 128)

    @property
    def loss(self):
        return mx_loss(self.source.get('loss', 'softmaxcrossentropyloss'))

    @property
    def optimizer(self):
        return self.source.get('optimizer', 'adagrad')

    @property
    def optimizer_params(self):
        return self.source.get('optimizer_params', {})


    @property
    def input_config(self):
        assert self.source.get('input_config') is not None

        return SimpleNamespace(
            dropout=self.source['input_config'].get('dropout', 0.0)
        )

    @property
    def output_config(self):
        assert self.source.get('output_config') is not None

        return SimpleNamespace(
            dropout=self.source['output_config'].get('dropout', 0.0),
            flatten=self.source['output_config'].get('flatten', True)
        )

    @property
    def fuse_conv_config(self):
        if self.source.get('fuse_conv_config') is None:
            return None
        return SimpleNamespace(
            filters=self.source['fuse_conv_config'].get('filters', 128),
            activation=self.source['fuse_conv_config'].get('activation', 'relu'),
            pool=self.source['fuse_conv_config'].get('pool', None),
            dropout=self.source['fuse_conv_config'].get('dropout', 0.2)
        )

    @property
    def ngram_conv_config(self):
        if self.source.get('ngram_conv_config') is None:
            return None
        return SimpleNamespace(
            ngrams=self.source['ngram_conv_config'].get('ngrams', [1, 2, 3, 4, 5]),
            filters=self.source['ngram_conv_config'].get('filters', 128),
            activation=self.source['ngram_conv_config'].get('activation', 'relu'),
            pool=self.source['ngram_conv_config'].get('pool', None),
            dropout=self.source['ngram_conv_config'].get('dropout', 0.2)
        )

    @property
    def hidden_configs(self):
        if self.source.get('hidden_configs') is None:
            return None

        def hidden_layer(config):
            return SimpleNamespace(**config)

        return [hidden_layer(config) for config in self.source.get('hidden_configs')]

    @property
    def rnn_config(self):
        if self.source.get('rnn_config') is None:
            return None
        return SimpleNamespace(
            mode=self.source['rnn_config'].get('mode', 'rnn'),
            hidden_size=self.source['rnn_config'].get('hidden_size', 200),
            num_layers=self.source['rnn_config'].get('num_layers', 1),
            activation=self.source['rnn_config'].get('activation', 'relu'),
            layout=self.source['rnn_config'].get('layout', 'TNC'),
            dropout=self.source['rnn_config'].get('dropout', 0),
            bidirectional=self.source['rnn_config'].get('bidirectional', False),
            i2h_weight_initializer=self.source['rnn_config'].get('i2h_weight_initializer', None),
            h2h_weight_initializer=self.source['rnn_config'].get('h2h_weight_initializer', None),
            i2h_bias_initializer=self.source['rnn_config'].get('i2h_bias_initializer', 'zeros'),
            h2h_bias_initializer=self.source['rnn_config'].get('h2h_bias_initializer', 'zeros'),
            input_size=self.source['rnn_config'].get('input_size', 0),
            clip=self.source['rnn_config'].get('clip', 0.2),
        )