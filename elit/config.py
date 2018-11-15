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
import configparser
import os

__author__ = "Han He"


class Config(object):
    def __init__(self, config_file: str, extra_args=None):
        if not os.path.isfile(config_file):
            raise FileNotFoundError('config file {} not found'.format(config_file))
        _config = configparser.ConfigParser(allow_no_value=True)
        _config.read(config_file)
        if extra_args:
            for section in _config.sections():
                for k, v in _config.items(section):
                    if k in extra_args:
                        v = type(v)(extra_args[k])
                        _config.set(section, k, v)
        self._config = _config

    @property
    def config(self):
        return self._config