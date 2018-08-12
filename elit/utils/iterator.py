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
import collections
import inspect
import random
from typing import Sequence

from mxnet.ndarray import NDArray

from elit.state import NLPState

__author__ = 'Jinho D. Choi'


class NLPIterator(collections.Iterator):
    def __init__(self, states: Sequence[NLPState]):
        self.states = states

    def __next__(self):
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))


class BatchIterator(NLPIterator):
    def __init__(self, states: Sequence[NLPState], batch_size: int, shuffle: bool, label: bool):
        super().__init__(states)
        batches = []
        batch = []
        total = 0
        count = 0

        for state in states:
            for x, y, in state:
                if label: batch.append((x, y))
                else: batch.append(x)
                count += 1

                if count == batch_size:
                    batches.append(batch)
                    total += count
                    batch = []
                    count = 0

        if batch: batches.append(batch)
        self._batches = batches
        self._shuffle = shuffle
        self.total = total

    def __iter__(self) -> 'BatchIterator':
        if self._shuffle: random.shuffle(self._batches)
        self._iter = 0
        return self

    def __next__(self) -> NDArray:
        if self._iter >= len(self._batches): raise StopIteration
        batch = self._batches[self._iter]
        self._iter += 1
        return batch
