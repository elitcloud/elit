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
# -*- coding:utf-8 -*-
# Filename: config.py
# Author：hankcs
# Date: 2018-02-23 11:15
import configparser
import os
from distutils.util import strtobool


class Config(object):
    def __init__(self, config_file: str, extra_args=None):
        """
        Load config
        :param config_file:
        :param extra_args:
        """
        if not os.path.isfile(config_file):
            raise FileNotFoundError('config file {} not found'.format(config_file))
        _config = configparser.ConfigParser()
        _config.read(config_file)
        if extra_args:
            # extra_args = dict([(k[2:], v) for k, v in zip(extra_args[0::2], extra_args[1::2])])
            for section in _config.sections():
                for k, v in _config.items(section):
                    if k in extra_args:
                        v = type(v)(extra_args[k])
                        _config.set(section, k, v)
        self._config = _config

            # for section in config.sections():
            #     for k, v in config.items(section):
            #         print(k, v)

    @property
    def save_dir(self):
        return self._config.get('Save', 'save_dir')

    @save_dir.setter
    def save_dir(self, save_dir):
        self._config.set('Save', 'save_dir', save_dir)

    @property
    def save_model_path(self):
        return self._config.get('Save', 'save_model_path')

    @property
    def save_vocab_path(self):
        return self._config.get('Save', 'save_vocab_path')

    @property
    def config_file(self):
        return self._config.get('Save', 'config_file')

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
    def num_epochs(self):
        return int(self._config.get('Run', 'num_epochs'))

    @property
    def learning_rate(self):
        return float(self._config.get('Optimizer', 'learning_rate'))

    @property
    def learning_rate_decay(self):
        return float(self._config.get('Optimizer', 'learning_rate_decay'))

    @property
    def debug(self):
        return bool(strtobool(self._config.get('Run', 'debug')))

    @property
    def save_config_path(self):
        return os.path.join(self.save_dir, 'config.ini')

    def save(self):
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        self._config.write(open(self.save_config_path, 'w'))
