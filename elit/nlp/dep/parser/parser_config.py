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

from elit.nlp.dep.common.config import Config


class ParserConfig(Config):
    def __init__(self, config_file, extra_args=None):
        """
        Load config
        :param config_file:
        :param extra_args:
        """
        super().__init__(config_file, extra_args)

    @property
    def pretrained_embeddings_file(self):
        return self._config.get('Data', 'pretrained_embeddings_file')

    @property
    def data_dir(self):
        return self._config.get('Data', 'data_dir')

    @property
    def train_file(self):
        return self._config.get('Data', 'train_file')

    @property
    def dev_file(self):
        return self._config.get('Data', 'dev_file')

    @property
    def test_file(self):
        return self._config.get('Data', 'test_file')

    @property
    def min_occur_count(self):
        return self._config.getint('Data', 'min_occur_count')

    @property
    def config_file(self):
        return self._config.get('Save', 'config_file')

    @property
    def save_dir(self) -> str:
        return self._config.get('Save', 'save_dir')

    @property
    def save_model_path(self):
        return self._config.get('Save', 'save_model_path')

    @property
    def save_vocab_path(self):
        return self._config.get('Save', 'save_vocab_path')

    @property
    def lstm_layers(self):
        return self._config.getint('Network', 'lstm_layers')

    @property
    def char_dims(self):
        return self._config.getint('Network', 'char_dims')

    @property
    def word_dims(self):
        return self._config.getint('Network', 'word_dims')

    @property
    def tag_dims(self):
        return self._config.getint('Network', 'tag_dims')

    @property
    def dropout_emb(self):
        return self._config.getfloat('Network', 'dropout_emb')

    @property
    def lstm_hiddens(self):
        return self._config.getint('Network', 'lstm_hiddens')

    @property
    def dropout_lstm_input(self):
        return self._config.getfloat('Network', 'dropout_lstm_input')

    @property
    def dropout_lstm_hidden(self):
        return self._config.getfloat('Network', 'dropout_lstm_hidden')

    @property
    def mlp_arc_size(self):
        return self._config.getint('Network', 'mlp_arc_size')

    @property
    def mlp_rel_size(self):
        return self._config.getint('Network', 'mlp_rel_size')

    @property
    def dropout_mlp(self):
        return self._config.getfloat('Network', 'dropout_mlp')

    @property
    def learning_rate(self):
        return self._config.getfloat('Optimizer', 'learning_rate')

    @property
    def decay(self):
        return self._config.getfloat('Optimizer', 'decay')

    @property
    def decay_steps(self):
        return self._config.getint('Optimizer', 'decay_steps')

    @property
    def beta_1(self):
        return self._config.getfloat('Optimizer', 'beta_1')

    @property
    def beta_2(self):
        return self._config.getfloat('Optimizer', 'beta_2')

    @property
    def epsilon(self):
        return self._config.getfloat('Optimizer', 'epsilon')

    @property
    def num_buckets_train(self):
        return self._config.getint('Run', 'num_buckets_train')

    @property
    def num_buckets_valid(self):
        return self._config.getint('Run', 'num_buckets_valid')

    @property
    def num_buckets_test(self):
        return self._config.getint('Run', 'num_buckets_test')

    @property
    def train_iters(self):
        return self._config.getint('Run', 'train_iters')

    @property
    def train_batch_size(self):
        return self._config.getint('Run', 'train_batch_size')

    @property
    def test_batch_size(self):
        return self._config.getint('Run', 'test_batch_size')

    @property
    def validate_every(self):
        return self._config.getint('Run', 'validate_every')

    @property
    def save_after(self):
        return self._config.getint('Run', 'save_after')

    @property
    def debug(self):
        return self._config.getboolean('Run', 'debug', fallback=False)

    @save_dir.setter
    def save_dir(self, value):
        self._config.set('Save', 'save_dir', value)


import argparse

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='data/ptb/par/config.ini')
    args, extra_args = argparser.parse_known_args()

    config = ParserConfig(args.config_file, extra_args)
    print(config.debug)
