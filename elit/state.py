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
import abc
import inspect
from typing import Optional, Tuple, Union

import numpy as np

from elit.structure import Document

__author__ = 'Jinho D. Choi'


class NLPState(abc.ABC):
    """
    :class:`NLPState` is an abstract class that takes a document and
    generates states to make predictions for an NLP task defined in its subclass.

    Abstract methods to be implemented:
      - :meth:`NLPState.init`
      - :meth:`NLPState.process`
      - :meth:`NLPState.has_next`
      - :meth:`NLPState.x`
      - :meth:`NLPState.y`
    """

    def __init__(self, document: Document, key: str):
        """
        :param document: the input document.
        :param key: the key to the input document where the predicted labels are to be saved.
        """
        self.document = document
        self.key = key

    def __iter__(self) -> 'NLPState':
        self.init()
        return self

    def __next__(self) -> Tuple[np.ndarray, Optional[int]]:
        if not self.has_next():
            raise StopIteration
        x, y = self.x, self.y
        self.process()
        return x, y

    @abc.abstractmethod
    def init(self, **kwargs):
        """
        Initializes the state of the input document.
        """
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))

    @abc.abstractmethod
    def process(self, *args):
        """
        :param args: parameters to be applied to the current state.
        Applies the parameters to the current state, then processes onto the next state.
        """
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))

    @abc.abstractmethod
    def has_next(self) -> bool:
        """
        :return: ``True`` if there exists the next state to be processed; otherwise, ``False``.
        """
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))

    @property
    def x(self) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """
        :return: the feature vector or the tuple of feature vectors extracted from the current state.
        :rtype: numpy.ndarray or tuple(numpy.ndarray, ....)
        """
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))

    @property
    @abc.abstractmethod
    def y(self) -> Optional[Union[int, Tuple[int, ...]]]:
        """
        :return: the class ID of the gold label for the current state if available; otherwise ``None``.
        :rtype: int or tuple(int, ....) or None
        """
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))
