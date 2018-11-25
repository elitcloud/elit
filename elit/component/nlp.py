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
from typing import Sequence

from elit.component import Component
from elit.structure import Document

__author__ = "Gary Lai"


class NLPComponent(Component):
    """
    :class:`NLPComponent` is an abstract class; any NLP component developed in ELIT must inherit
    this class.
    It is similar to :class:`Component` except that the type of training and development data is
    specified to :class:`elit.structure.Document`.

    Abstract methods to be implemented:
      - :meth:`Component.init`
      - :meth:`Component.load`
      - :meth:`Component.save`
      - :meth:`NLPComponent.train`
      - :meth:`NLPComponent.decode`
      - :meth:`NLPComponent.evaluate`
    """

    @abc.abstractmethod
    def train(self, trn_docs: Sequence[Document], dev_docs: Sequence[Document], model_path: str, **kwargs) -> float:
        """
        :param trn_docs: the sequence of documents for training.
        :param dev_docs: the sequence of documents for development (validation).
        :param model_path: the filepath where trained model(s) are to be saved.
        :return: the best score form the development data.

        Trains a model for this component and saves the model to the filepath.
        """
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))

    @abc.abstractmethod
    def decode(self, docs: Sequence[Document], **kwargs):
        """
        :param docs: the sequence of input documents.

        Processes the input documents, make predictions, and saves the predicted labels back to the
        input documents.
        """
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))

    @abc.abstractmethod
    def evaluate(self, docs: Sequence[Document], **kwargs):
        """
        :param docs: the sequence of input documents.

        Evaluates the current model of this component with the input documents.
        """
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))
