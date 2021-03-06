# coding: utf-8

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import pickle

from elit.util.io import save_json, load_json


class Savable(object):
    """
    A super class for save/load operations.
    """

    def __init__(self):
        super().__init__()

    def save(self, path):
        """Save to path

        Parameters
        ----------
        path : str
            file path
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        """Load from path

        Parameters
        ----------
        path : str
            file path

        Returns
        -------
        Savable
            An object
        """
        with open(path, 'rb') as f:
            return pickle.load(f)

    def save_json(self, path):
        save_json(self.__dict__, path)

    @staticmethod
    def load_json(path):
        return load_json(path)


def pickle_save(item, path):
    with open(path, 'wb') as f:
        pickle.dump(item, f)


def pickle_load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
