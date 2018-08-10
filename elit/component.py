# ========================================================================
# Copyright 2017 Emory University
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
import logging
import random
import time
from typing import List, Union, Sequence, Type

import mxnet as mx
import numpy as np
from mxnet import nd, gluon, autograd
from mxnet.gluon.data import DataLoader
from mxnet.ndarray import NDArray

from elit.eval import EvalMetric
from elit.state import NLPState, BatchState, SequenceState
from elit.structure import Document

__author__ = 'Jinho D. Choi, Gary Lai'


class Component(abc.ABC):
    """
    :class:`Component` provides an abstract class to implement components.
    Any component developed in ELIT must inherit this class.

    Abstract methods to be implemented:
      - :meth:`Component.init`
      - :meth:`Component.load`
      - :meth:`Component.save`
      - :meth:`Component.decode`
      - :meth:`Component.train`
    """

    @abc.abstractmethod
    def init(self, **kwargs):
        """
        :param kwargs: custom parameters.
        Abstract method to initialize this component.
        """
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))

    @abc.abstractmethod
    def load(self, model_path: str, **kwargs):
        """
        :param model_path: the filepath where a pre-trained model can be loaded.
        :param kwargs: custom parameters.
        Abstract method to load a pre-trained model from the filepath.
        """
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))

    @abc.abstractmethod
    def save(self, model_path: str, **kwargs):
        """
        :param model_path: the filepath where the current model can be saved.
        :param kwargs: custom parameters.
        Abstract method to save the current model to the filepath.
        """
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))

    @abc.abstractmethod
    def decode(self, data, **kwargs):
        """
        :param data: input data.
        :param kwargs: custom parameters.
        Abstract method to process the input data and save the processed results to the input data.
        """
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))

    @abc.abstractmethod
    def train(self, trn_data, dev_data, model_path: str, **kwargs):
        """
        :param trn_data: training data.
        :param dev_data: development (validation) data.
        :param model_path: the filepath where trained model(s) are to be saved.
        :param kwargs: custom parameters
        Abstract class to train and save a model for this component.
        """
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))


class NLPComponent(Component):
    """
    :class:`NLPComponent` provides an abstract class to implement NLP components.
    Any NLP component developed in ELIT must inherit this class.
    This is similar to :class:`Component` expect that the type of training and development data is specified to :class:`elit.structure.Document`.

    Abstract methods to be implemented:
      - :meth:`Component.init`
      - :meth:`Component.load`
      - :meth:`Component.save`
      - :meth:`NLPComponent.decode`
      - :meth:`NLPComponent.train`
    """

    @abc.abstractmethod
    def decode(self, docs: Sequence[Document], **kwargs):
        """
        :param docs: the sequence of input documents.
        :param kwargs: custom parameters.
        Abstract method to process the input documents and save the processed results to the input documents.
        """
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))

    @abc.abstractmethod
    def train(self, trn_docs: Sequence[Document], dev_docs: Sequence[Document], model_path: str, **kwargs):
        """
        :param trn_docs: the sequence of documents for training.
        :param dev_docs: the sequence of documents for development (validation).
        :param model_path: the filepath where trained model(s) are to be saved.
        :param kwargs: custom parameters
        Abstract method to train and save a model for this component.
        """
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))


class MXNetComponent(NLPComponent):
    """
    :class:`MXNetComponent` provides an abstract class to implement machine-learning based components using `MXNet <http://mxnet.io>`_.
    The abstract methods :meth:`NLPComponent.decode` and :meth:`NLPComponent.train` are defined here
    although they call other abstract methods newly introduced in this class.

    Abstract methods to be implemented:
      - :meth:`Component.init`
      - :meth:`Component.load`
      - :meth:`Component.save`
      - :meth:`MXNetComponent.create_states`
      - :meth:`MXNetComponent.eval_metric`
      - :meth:`MXNetComponent.finalize`
      - :meth:`MXNetComponent.decode_states`
      - :meth:`MXNetComponent.evaluate_states`
      - :meth:`MXNetComponent.train_states`
    """

    def __init__(self, ctx: Union[mx.Context, Sequence[mx.Context]]):
        """
        :param ctx: the (list of) device context(s) for MXNet blocks.
        """
        self.ctx = ctx
        self.model: gluon.Block = None

    @abc.abstractmethod
    def create_states(self, documents: Sequence[Document]) -> List[NLPState]:
        """
        :param documents: the sequence of input documents.
        :return: the list of states corresponding to the input documents.
        Abstract method to create the list of states corresponding to the input documents.
        """
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))

    @abc.abstractmethod
    def eval_metric(self) -> Type[EvalMetric]:
        """
        :return: the evaluation metric for this component.
        Abstract method to create the evaluation metric for this component.
        """
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))

    @abc.abstractmethod
    def finalize(self, state: NLPState):
        """
        :param state: input state.
        Abstract method to finalize the input state by saving predicted labels to the document associated with the state.
        This method must be called for every state once decoding is done; otherwise, no predicted labels are saved to the document.
        """
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))

    @abc.abstractmethod
    def decode_states(self,
                      states: Sequence[NLPState],
                      batch_size: int,
                      xs: Sequence[np.ndarray] = None):
        """
        :param states: the sequence of input states.
        :param batch_size: the batch size .
        :param xs: the sequence of feature vectors extracted from the input states.
        Abstract method to process the input states and save the processed results to the documents associated with the states.
        """
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))

    @abc.abstractmethod
    def evaluate_states(self,
                        states: List[NLPState],
                        batch_size: int,
                        xs: Sequence[np.ndarray] = None) -> Type[EvalMetric]:
        """
        :param states: the sequence of input states.
        :param batch_size: the batch size .
        :param xs: the sequence of feature vectors extracted from the input states.
        :return: the evaluation metric including evaluation statistics on the input states.
        Abstract method to evaluate the current model of this component on the input states.
        """
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))

    @abc.abstractmethod
    def train_states(self,
                     states: Sequence[NLPState],
                     batch_size: int,
                     loss: gluon.loss.Loss,
                     trainer: gluon.Trainer,
                     xs: Sequence[np.ndarray] = None,
                     ys: Sequence[np.ndarray] = None) -> float:
        """
        :param states: the sequence of input states.
        :param batch_size: the batch size .
        :param loss: the loss function.
        :param trainer: the trainer.
        :param xs: the sequence of feature vectors extracted from the input states.
        :param ys: the sequence of gold-standard class IDs corresponding to the feature vectors.
        :return: the training accuracy.
        Abstract method to train and save a model for this component.
        """
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))

    # override
    def decode(self, docs: Sequence[Document], batch_size: int = 2048):
        """
        :param docs: the sequence of input documents.
        :param batch_size: the batch size.
        Processes the input documents and save the processed results to the input documents.
        """
        log = ('* Decoding',
               '- context(s): %s' % str(self.ctx),
               '- batch size: %d' % batch_size)
        logging.info('\n'.join(log))

        states = self.create_states(docs)
        self.decode_states(states, batch_size)

    def decode_single_device(self,
                             states: Sequence[NLPState],
                             x_batches: DataLoader,
                             sequence: bool = False) -> List[NDArray]:
        """
        :param states: the sequence of input states.
        :param x_batches: the batches of feature vectors extracted from the input states.
        :param sequence: if True, decode in sequence mode; otherwise, in batch mode.
        :return: if sequence is True, an empty list; otherwise, the list of output scores for all instances in the batches.
        Decodes the input states with a single device.
        """
        output_list = []
        begin = 0

        for x_batch in x_batches:
            x_batch = x_batch.as_in_context(self.ctx)
            outputs = self.model(x_batch)

            if sequence:
                for i, output in enumerate(outputs): states[begin + i].process(output.asnumpy())
                begin += len(outputs)
            else:
                output_list.append(outputs)

        return nd.concat(*output_list, dim=0) if output_list else output_list

    def decode_multiple_devices(self,
                                states: Sequence[NLPState],
                                batches: DataLoader,
                                sequence: bool = False) -> List[NDArray]:
        """
        Decodes with multiple devices; in other words, self.ctx is a list of devices.
        :param states: list of states.
        :param batches: decoding batches extracted from the states.
        :param sequence: if True, decode in sequence mode; otherwise, batch mode.
        :return: the list of output scores (batch mode only).
        """
        output_list = []
        begin = 0

        for x_batch in batches:
            ctx = self.ctx[:x_batch.shape[0]] if x_batch.shape[0] < len(self.ctx) else self.ctx
            x_splits = gluon.utils.split_and_load(x_batch, ctx, even_split=False)
            output_splits = [self.model(x_split) for x_split in x_splits]

            if sequence:
                for output_split in output_splits:
                    for i, output in enumerate(output_split): states[begin + i].process(output.asnumpy())
                    begin += len(output_split)
            else:
                output_list.extend(output_splits)

        nd.waitall()
        return output_list

    # override
    def train(self,
              trn_docs: List[Document],
              dev_docs: List[Document],
              model_path: str,
              trn_batch: int = 64,
              dev_batch: int = 2048,
              epoch: int = 50,
              loss: gluon.loss.Loss = None,
              optimizer: str = 'adagrad',
              learning_rate: float = 0.01,
              weight_decay: float = 0.0):
        """
        :param trn_docs: a list of documents for training.
        :param dev_docs: a list of documents for development (validation).
        :param model_path: a filepath to the trained model to be saved.
        :param trn_batch: the batch size applied to the training set.
        :param dev_batch: the batch size applied to the development set.
        :param epoch: the maximum number of epochs.
        :param loss: the loss function (default: :meth:`mxnet.gluon.loss.SoftmaxCrossEntropyLoss`).
        :param optimizer: the type of optimizer for training (:class:`mxnet.optimizer.Optimizer`).
        :param learning_rate: the initial learning rate.
        :param weight_decay: the L2 regularization coefficient.
        """
        # create instances
        trn_states = self.create_states(trn_docs)
        dev_states = self.create_states(dev_docs)

        # generate instances for batch mode
        batch = isinstance(trn_states[0], BatchState)

        if batch:
            trn_xs, trn_ys = zip(*[(x, y) for state in trn_states for x, y in state])
            dev_xs = [x for state in dev_states for x, _ in state]
        else:
            trn_xs, trn_ys, dev_xs = None, None, None

        # create a trainer
        if loss is None:
            loss = gluon.loss.SoftmaxCrossEntropyLoss()
        trainer = gluon.Trainer(self.model.collect_params(), optimizer, {'learning_rate': learning_rate, 'wd': weight_decay})

        # train
        log = ('* Training',
               '- context(s)   : %s' % str(self.ctx),
               '- train batch  : %d' % trn_batch,
               '- epoch        : %d' % epoch,
               '- loss         : %s' % str(loss),
               '- optimizer    : %s' % optimizer,
               '- learning rate: %f' % learning_rate,
               '- weight decay : %f' % weight_decay)
        print('\n'.join(log))

        best_e, best_eval = -1, -1

        for e in range(1, epoch + 1):
            st = time.time()
            trn_acc = self.train_states(trn_states, trn_batch, loss, trainer, trn_xs, trn_ys)
            mt = time.time()
            dev_metric = self.evaluate_states(dev_states, dev_batch, dev_xs)
            et = time.time()
            if best_eval < dev_metric.get():
                best_e, best_eval = e, dev_metric.get()
                self.save(model_path)

            print('%4d: trn-time: %d, dev-time: %d, trn-acc: %5.2f, dev-eval: %5.2f, best-dev: %5.2f @%4d' %
                  (e, mt - st, et - mt, trn_acc, dev_metric.get(), best_eval, best_e))

    def train_single_device(self,
                            states: List[NLPState],
                            batches: DataLoader,
                            loss: gluon.loss.Loss,
                            trainer: gluon.Trainer,
                            sequence: bool = False) -> int:
        """
        Trains with a single device; in other words, self.ctx is one device.
        :param states: list of states.
        :param batches: decoding batches extracted from the states.
        :param loss: the loss function.
        :param trainer: the trainer.
        :param sequence: if True, decode in sequence mode; otherwise, batch mode.
        :return: the number of correctly classified instances.
        """
        correct = 0
        begin = 0

        for x_batch, y_batch in batches:
            x_batch = x_batch.as_in_context(self.ctx)
            y_batch = y_batch.as_in_context(self.ctx)

            with autograd.record():
                outputs = self.model(x_batch)
                l = loss(outputs, y_batch)
                l.backward()

            trainer.step(x_batch.shape[0])
            correct += len([1 for o, y in zip(mx.ndarray.argmax(outputs, axis=1), y_batch) if int(o.asscalar()) == int(y.asscalar())])

            if sequence:
                for i, output in enumerate(outputs): states[begin + i].process(output.asnumpy())
                begin += len(outputs)

        return correct

    def train_multiple_devices(self,
                               states: List[NLPState],
                               batches: DataLoader,
                               loss: gluon.loss.Loss,
                               trainer: gluon.Trainer,
                               sequence: bool = False) -> int:
        """
        Trains with multiple devices; in other words, self.ctx is a list of devices.
        :param states: list of states.
        :param batches: decoding batches extracted from the states.
        :param loss: the loss function.
        :param trainer: the trainer.
        :param sequence: if True, decode in sequence mode; otherwise, batch mode.
        :return: the number of correctly classified instances.
        """
        correct = 0
        begin = 0

        for x_batch, y_batch in batches:
            ctx = self.ctx[:x_batch.shape[0]] if x_batch.shape[0] < len(self.ctx) else self.ctx
            x_splits = gluon.utils.split_and_load(x_batch, ctx, even_split=False)
            y_splits = gluon.utils.split_and_load(y_batch, ctx, even_split=False)

            with autograd.record():
                output_splits = [self.model(x_split) for x_split in x_splits]
                losses = [loss(output_split, y_split) for output_split, y_split in zip(output_splits, y_splits)]
                for l in losses: l.backward()

            trainer.step(x_batch.shape[0])

            for output_split, y_split in zip(output_splits, y_splits):
                correct += len([1 for o, y in zip(mx.ndarray.argmax(output_split, axis=1), y_split) if int(o.asscalar()) == int(y.asscalar())])

                if sequence:
                    for i, output in enumerate(output_split): states[begin + i].process(output.asnumpy())
                    begin += len(output_split)

        nd.waitall()
        return correct


class BatchComponent(MXNetComponent):
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
    def decode_states(self,
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
    def evaluate_states(self,
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
        self.decode_states(states, batch_size, xs)
        metric = self.eval_metric()
        for state in states:
            metric.update(state.document)
        return metric

    # override
    def train_states(self,
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
            device = self.train_multiple_devices
            batch_size *= len(self.ctx)
        else:
            device = self.train_single_device

        batches = gluon.data.DataLoader(gluon.data.ArrayDataset(xs, ys), batch_size=batch_size, shuffle=True)
        correct = device(states, batches, loss, trainer, False)
        return 100 * correct / len(ys)


class SequenceComponent(MXNetComponent):
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
    def decode_states(self,
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
    def evaluate_states(self,
                        states: List[SequenceState],
                        batch_size: int,
                        xs: List[np.ndarray] = None) -> EvalMetric:
        """
        :param states: a list of input states.
        :param batch_size: the batch size.
        :param xs: not used for sequence mode.
        :return: the metric including evaluation statistics on the input.
        """
        self.decode_states(states, batch_size)
        metric = self.eval_metric()

        for state in states:
            metric.update(state.document)
            state.init()

        return metric

    # override
    def train_states(self,
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
            device = self.train_multiple_devices
            batch_size *= len(self.ctx)
        else:
            device = self.train_single_device

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
