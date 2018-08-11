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
import random
from typing import Tuple, Optional, Union, List, Type

import mxnet as mx
import numpy as np
from mxnet import gluon, ndarray

from elit.component import BatchComponent
from elit.eval import EvalMetric

from elit.state import NLPState
from elit.structure import Document

__author__ = 'Jinho D. Choi'


class BatchState(NLPState):
    """
    BatchState provides an abstract class to define a decoding strategy in batch mode.
    Batch mode assumes that predictions made by earlier states do not affect predictions made by later states.
    Thus, all predictions can be made in batch where each prediction is independent from one another.
    BatchState is iterable (see self.__iter__() and self.__next__()).
    Abstract methods to be implemented: init, process, has_next, x, y, assign.
    """

    def __init__(self, document: Document, key: str):
        """
        :param document: an input document.
        :param key: the key to the input document where the predicted labels are to be saved.
        """
        super().__init__(document, key)

    def __iter__(self) -> 'BatchState':
        self.init()
        return self

    def __next__(self) -> Tuple[np.ndarray, Optional[int]]:
        """
        :return: self.x, self.y
        """
        if not self.has_next: raise StopIteration
        x, y = self.x, self.y
        self.process()
        return x, y

    @abc.abstractmethod
    def assign(self, output: np.ndarray, begin: int = 0) -> int:
        """
        Assigns the predicted output in batch to the input document.
        :param output: a matrix where each row contains prediction scores of the corresponding state.
        :param begin: the row index of the output matrix corresponding to the initial state.
        :return: the row index of the output matrix to be assigned by the next document.
        """
        pass


class SequenceState(NLPState):
    """
    SequenceState provides an abstract class to define a decoding strategy in sequence mode.
    Sequence mode assumes that predictions made by earlier states affect predictions made by later states.
    Thus, predictions need to be made in sequence where earlier predictions get passed onto later states.
    Abstract methods to be implemented: init, process, has_next, x, y.
    """

    def __init__(self, document: Document, key: str):
        """
        :param document: an input document
        :param key: the key to the input document where the predicted labels are to be saved.
        """
        super().__init__(document, key)

    @abc.abstractmethod
    def process(self, output: np.ndarray):
        """
        Applies the predicted output to the current state, then processes to the next state.
        :param output: the predicted output of the current state.
        """
        pass


class BatchComponent(BatchComponent):
    """
    BatchComponent provides an abstract class to implement an NLP component in batch mode.
    See the description of :class:`elit.state.BatchState` for more details about batch mode.
    Abstract methods to be implemented: init, load, save, create_states, finalize, eval_metric.
    """

    def __init__(self, ctx: Union[mx.Context, List[mx.Context]]):
        """
        :param ctx: a device context.
        """
        super().__init__(ctx)

    @abc.abstractmethod
    def create_states(self, documents: List[Document]) -> List[BatchState]:
        """
        :param documents: a list of input documents.
        :return: the list of initial states corresponding to the input documents.
        """
        pass

    # override
    def _decode(self,
                states: List[BatchState],
                batch_size: int,
                xs: List[np.ndarray] = None):
        """
        :param states: a list of input states.
        :param batch_size: the batch size .
        :param xs: a list of feature vectors extracted from the input states.
        """
        if xs is None: xs = [x for state in states for x, y in state]

        if isinstance(self.ctx, list):
            device = self.decode_multiple_devices
            batch_size *= len(self.ctx)
        else:
            device = self.decode_single_device

        batches = gluon.data.DataLoader(xs, batch_size=batch_size, shuffle=False)
        outputs = device(states, batches, False)
        begin = 0

        for state in states:
            begin = state.assign(outputs, begin)
            self.finalize(state)

    # override
    def _evaluate(self,
                  states: List[BatchState],
                  batch_size: int,
                  xs: List[np.ndarray] = None) -> Type[EvalMetric]:
        """
        :param states: a list of input states.
        :param batch_size: the batch size.
        :param xs: a list of feature vectors extracted from the input states (optional).
        :return: the metric including evaluation statistics on the input.
        """
        if xs is None: xs = [x for state in states for x, y in state]
        self._decode(states, batch_size, xs)
        metric = self.eval_metric()
        for state in states:
            metric.update(state.document)
        return metric

    # override
    def _train(self,
               states: List[BatchState],
               batch_size: int,
               loss: gluon.loss.Loss,
               trainer: gluon.Trainer,
               xs: List[np.ndarray] = None,
               ys: List[int] = None) -> float:
        """
        :param states: a list of states including documents for training.
        :param batch_size: the batch size.
        :param loss: the loss function.
        :param trainer: the trainer.
        :param xs: a list of feature vectors.
        :param ys: a list of gold-standard class IDs corresponding to the feature vectors.
        :return: the number of correctly classified instances.
        """
        if isinstance(self.ctx, list):
            device = self._train_multiple_devices
            batch_size *= len(self.ctx)
        else:
            device = self._train_single_device

        batches = gluon.data.DataLoader(gluon.data.ArrayDataset(xs, ys), batch_size=batch_size, shuffle=True)
        correct = device(states, batches, loss, trainer, False)
        return 100 * correct / len(ys)


class SequenceComponent(BatchComponent):
    """
    SequenceComponent provides an abstract class to implement an NLP component in sequence mode.
    See the description of :class:`elit.state.SequenceState` for more details about sequence mode.
    Abstract methods to be implemented: init, load, save, create_states, finalize, eval_metric.
    """

    def __init__(self, ctx: Union[mx.Context, List[mx.Context]]):
        """
        :param ctx: a device context.
        """
        super().__init__(ctx)

    @abc.abstractmethod
    def create_states(self, documents: List[Document]) -> List[SequenceState]:
        """
        :param documents: a list of input documents.
        :return: the list of initial states corresponding to the input documents.
        """
        pass

    # override
    def _decode(self,
                states: List[SequenceState],
                batch_size: int,
                xs: List[np.ndarray] = None):
        """
        :param states: a list of input states.
        :param batch_size: the batch size .
        :param xs: not used for sequence mode.
        """
        tmp = list(states)

        if isinstance(self.ctx, list):
            device = self.decode_multiple_devices
            batch_size *= len(self.ctx)
        else:
            device = self.decode_single_device

        while tmp:
            xs = nd.array([state.x for state in tmp])
            batches = gluon.data.DataLoader(xs, batch_size=batch_size, shuffle=False)
            device(states, batches, True)
            tmp = [state for state in tmp if state.has_next]

        for state in states:
            self.finalize(state)

    # override
    def _evaluate(self,
                  states: List[SequenceState],
                  batch_size: int,
                  xs: List[np.ndarray] = None) -> EvalMetric:
        """
        :param states: a list of input states.
        :param batch_size: the batch size.
        :param xs: not used for sequence mode.
        :return: the metric including evaluation statistics on the input.
        """
        self._decode(states, batch_size)
        metric = self.eval_metric()

        for state in states:
            metric.update(state.document)
            state.init()

        return metric

    # override
    def _train(self,
               states: List[SequenceState],
               batch_size: int,
               loss: gluon.loss.Loss,
               trainer: gluon.Trainer,
               xs: List[np.ndarray] = None,
               ys: List[np.ndarray] = None) -> float:
        """
        :param states: a list of states including documents for training.
        :param batch_size: the batch size.
        :param loss: the loss function.
        :param trainer: the trainer.
        :param xs: not used for sequence mode.
        :param ys: not used for sequence mode.
        :return: the number of correctly classified instances.
        """
        tmp = list(states)
        correct = 0
        total = 0

        if isinstance(self.ctx, list):
            device = self._train_multiple_devices
            batch_size *= len(self.ctx)
        else:
            device = self._train_single_device

        while tmp:
            random.shuffle(tmp)
            xs = [state.x for state in tmp]
            ys = [state.y for state in tmp]

            batches = gluon.data.DataLoader(gluon.data.ArrayDataset(xs, ys), batch_size=batch_size)
            correct += device(states, batches, loss, trainer, True)
            total += len(ys)
            tmp = [state for state in tmp if state.has_next]

        for state in states: state.init()
        return 100 * correct / total