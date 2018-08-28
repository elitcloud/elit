# -*- coding:utf-8 -*-
# Filename: TaggerConfig.py
# Author：hankcs
# Date: 2018-02-21 21:43
from distutils.util import strtobool

from elit.dep.common.config import Config


class TaggerConfig(Config):
    def __init__(self, config_file, extra_args=None):
        super().__init__(config_file, extra_args)

    @property
    def pretrained_embeddings_file(self):
        return self._config.get('Data', 'pretrained_embeddings_file')

    @property
    def margins(self):
        return self._config.get('Network', 'margins')

    @property
    def word_embedding_dim(self):
        return int(self._config.get('Network', 'word_embedding_dim'))

    @property
    def char_embedding_dim(self):
        return int(self._config.get('Network', 'char_embedding_dim'))

    @property
    def word_lstm_layers(self):
        return int(self._config.get('Network', 'word_lstm_layers'))

    @property
    def word_lstm_dim(self):
        return int(self._config.get('Network', 'word_lstm_dim'))

    @property
    def tie_two_embeddings(self):
        return bool(strtobool(self._config.get('Network', 'tie_two_embeddings')))

    @property
    def lower_case(self):
        return bool(self._config.get('Data', 'lower_case'))

    @property
    def keep_oov(self):
        return bool(self._config.get('Data', 'keep_oov'))

    @keep_oov.setter
    def keep_oov(self, keep_oov):
        self._config.set('Data', 'keep_oov', str(keep_oov).lower())

    @property
    def char_lstm_dim(self):
        return int(self._config.get('Network', 'char_lstm_dim'))

    @property
    def dropout(self):
        return float(self._config.get('Network', 'dropout'))

    @property
    def clip_norm(self):
        return float(self._config.get('Optimizer', 'clip_norm'))

    @property
    def batch_size(self):
        return int(self._config.get('Run', 'batch_size'))


if __name__ == '__main__':
    config = TaggerConfig('data/pku/config.ini')
    config.keep_oov = True
