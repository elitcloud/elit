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
import collections
import inspect
import random
from itertools import islice
from typing import Sequence, Union, List, Tuple

import numpy as np
from mxnet.ndarray import NDArray

from elit.state import NLPState

__author__ = 'Jinho D. Choi'


class NLPIterator(collections.Iterator):
    """
    :class:`NLPIterator` generates batches of instances from the input states that are iterable.
    """

    def __init__(self, states: Sequence[NLPState]):
        """
        :param states: the sequence of input states.
        """
        self.states = states

    @abc.abstractmethod
    def __iter__(self) -> 'NLPIterator':
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))

    @abc.abstractmethod
    def __next__(self):
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))

    def __str__(self):
        return self.__class__.__name__

    @abc.abstractmethod
    def process(self, state_begin: int, *args) -> int:
        """
        :param state_begin: the index of the input state to begin the process with.
        :param args: parameters including the outputs; ``len(args)`` must be at least 1.
        :return: the index of the state to be processed next after this method is called.

        Processes through the states and assigns the parameters accordingly.
        """
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))


class BatchIterator(NLPIterator):
    """
    :class:`BatchIterator` generates all batches of instances in the constructor, and iterates through the generated batches.
    This is useful when all instances are independent; in other words, the prediction for one instance does not affect the prediction of any other instance.
    """

    def __init__(self, states: Sequence[NLPState], batch_size: int, shuffle: bool, label: bool, **kwargs):
        """
        :param states: the input states.
        :param batch_size: the size of batches to process at a time.
        :param shuffle: if ``True``, shuffle instances for every epoch; otherwise, no shuffle.
        :param label: if ``True``, each instance is a tuple of (feature vector, label); otherwise, it is just a feature vector.
        :param kwargs: custom parameters to initialize each state.
        """
        super().__init__(states)
        batches = []
        batch = []
        count = 0

        for state in states:
            for x, y, in state:
                if label: batch.append((x, y))
                else: batch.append(x)
                count += 1

                if count == batch_size:
                    batches.append(batch)
                    batch = []
                    count = 0

        if batch: batches.append(batch)
        self._batches = batches
        self._shuffle = shuffle
        self._kwargs = kwargs
        self._iter = 0

    # override
    def __iter__(self) -> 'BatchIterator':
        if self._shuffle:
            random.shuffle(self._batches)
        else:
            for state in self.states: state.init(**self._kwargs)

        self._iter = 0
        return self

    # override
    def __next__(self) -> List[Union[np.ndarray, Tuple[np.ndarray, int]]]:
        if self._iter >= len(self._batches): raise StopIteration
        batch = self._batches[self._iter]
        self._iter += 1
        return batch

    # override
    def process(self, state_begin: int, *args) -> int:
        if self._shuffle: return -1
        # args = [arg.asnumpy() if isinstance(arg, NDArray) else arg for arg in args]
        o = len(args) == 1
        i = 0

        for state in islice(self.states, state_begin, None):
            while state.has_next():
                if i == len(args[0]): return state_begin
                if o: state.process(args[0][i])
                else: state.process(*[arg[i] for arg in args])
                i += 1

            state_begin += 1

        return state_begin


class SequenceIterator(NLPIterator):
    def __init__(self, states: Sequence[NLPState], batch_size: int, shuffle: bool, label: bool, **kwargs):
        super().__init__(states)
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._label = label
        self._kwargs = kwargs
        self._states = None
        self._iter = 0

    def __iter__(self) -> 'SequenceIterator':
        for state in self.states: state.init(**self._kwargs)
        self._states = list(self.states)
        if self._shuffle: random.shuffle(self._states)
        self._iter = 0
        return self

    def __next__(self) -> List[Union[np.ndarray, Tuple[np.ndarray, int]]]:
        if not self._states: raise StopIteration
        batch = [(state.x, state.y) if self._label else state.x for state in islice(self._states, self._iter, self._iter + self._batch_size)]
        self._iter += len(batch)
        return batch

    def process(self, state_begin: int, *args) -> int:
        # args = [arg.asnumpy() if isinstance(arg, NDArray) else arg for arg in args]
        o = len(args) == 1

        for i, state in enumerate(islice(self._states, state_begin, self._iter)):
            if o: state.process(args[0][i])
            else: state.process(*[arg[i] for arg in args])

        if self._iter >= len(self._states):
            self._states = [state for state in self._states if state.has_next()]
            if self._shuffle: random.shuffle(self._states)
            self._iter = 0

        return self._iter
