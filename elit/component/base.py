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
import abc
import inspect
from typing import Any

__author__ = 'Jinho D. Choi, Gary Lai'


class Component(abc.ABC):
    """
    :class:`Component` is an abstract class; any component developed in ELIT must inherit this class.

    Abstract methods to be implemented:
      - :meth:`Component.load`
      - :meth:`Component.save`
      - :meth:`Component.train`
      - :meth:`Component.decode`
      - :meth:`Component.evaluate`
    """

    @abc.abstractmethod
    def load(self, model_path: str, **kwargs):
        """
        :param model_path: the filepath where a model can be loaded.

        Loads a model for this component from the filepath.
        """
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))

    @abc.abstractmethod
    def save(self, model_path: str, **kwargs):
        """
        :param model_path: the filepath where the current model can be saved.

        Saves the current model of this component to the filepath.
        """
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))

    @abc.abstractmethod
    def train(self, trn_data: Any, dev_data: Any, model_path: str, **kwargs) -> float:
        """
        :param trn_data: training data.
        :param dev_data: development (validation) data.
        :param model_path: the filepath where the trained model(s) are to be saved.
        :return: the best score form the development data.

        Trains a model for this component and saves the model to the filepath.
        """
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))

    @abc.abstractmethod
    def decode(self, data: Any, **kwargs):
        """
        :param data: input data.

        Processes the input data, make predictions, and saves the predicted labels back to the
        input data.
        """
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))

    @abc.abstractmethod
    def evaluate(self, data: Any, **kwargs):
        """
        :param data: input data.

        Evaluates the current model of this component with the input data.
        """
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))


