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
import time
from itertools import islice
from typing import List, Union, Sequence, Type, Dict

import mxnet as mx
import numpy as np
from mxnet import nd, gluon, autograd
from mxnet.ndarray import NDArray

from elit.eval import EvalMetric
from elit.state import NLPState, process_outputs
from elit.structure import Document
from elit.utils.iterator import BatchIterator

__author__ = 'Jinho D. Choi, Gary Lai'


class Component(abc.ABC):
    """
    :class:`Component` is an abstract class; any component developed in ELIT must inherit this class.

    Abstract methods to be implemented:
      - :meth:`Component.init`
      - :meth:`Component.load`
      - :meth:`Component.save`
      - :meth:`Component.decode`
      - :meth:`Component.evaluate`
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
    def evaluate(self, data, **kwargs):
        """
        :param data: input data.
        :param kwargs: custom parameters.

        Abstract method to evaluate the current model of this component with the input data.
        """
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))

    @abc.abstractmethod
    def train(self, trn_data, dev_data, model_path: str, **kwargs):
        """
        :param trn_data: training data.
        :param dev_data: development (validation) data.
        :param model_path: the filepath where trained model(s) are to be saved.
        :param kwargs: custom parameters.

        Abstract class to train and save a model for this component.
        """
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))


class NLPComponent(Component):
    """
    :class:`NLPComponent` is an abstract class; any NLP component developed in ELIT must inherit this class.
    It is similar to :class:`Component` except that the type of training and development data is specified to :class:`elit.structure.Document`.

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
    def evaluate(self, docs: Sequence[Document], **kwargs):
        """
        :param docs: the sequence of input documents.
        :param kwargs: custom parameters.

        Abstract method to evaluate the current model of this component with the input documents.
        """
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))

    @abc.abstractmethod
    def train(self, trn_docs: Sequence[Document], dev_docs: Sequence[Document], model_path: str, **kwargs):
        """
        :param trn_docs: the sequence of documents for training.
        :param dev_docs: the sequence of documents for development (validation).
        :param model_path: the filepath where trained model(s) are to be saved.
        :param kwargs: custom parameters.

        Abstract method to train and save a model for this component.
        """
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))


class BatchComponent(NLPComponent):
    """
    :class:`BatchComponent` is an abstract class to implement NLP components that process all states in batch
    such that it assumes that every state is independent to one another.
    The abstract methods :meth:`NLPComponent.decode` and :meth:`NLPComponent.train` are defined in this class.

    Abstract methods to be implemented:
      - :meth:`Component.init`
      - :meth:`Component.load`
      - :meth:`Component.save`
      - :meth:`BatchComponent.create_states`
      - :meth:`BatchComponent.eval_metric`
    """

    def __init__(self, ctx: Union[mx.Context, Sequence[mx.Context]]):
        """
        :param ctx: the (list of) device context(s) for :class:`mxnet.gluon.Block`.
        """
        self.ctx = ctx
        self.model: gluon.Block = None

    @abc.abstractmethod
    def create_states(self, documents: Sequence[Document]) -> List[NLPState]:
        """
        :param documents: the sequence of input documents.
        :return: the list of states corresponding to the input documents.

        Abstract method.
        """
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))

    @abc.abstractmethod
    def eval_metric(self) -> Type[EvalMetric]:
        """
        :return: the evaluation metric for this component.

        Abstract method.
        """
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))

    # override
    def decode(self, docs: Sequence[Document], batch_size: int = 2048):
        """
        :param docs: the sequence of input documents.
        :param batch_size: the batch size.

        Processes the input documents and saves the predicted labels to the input documents.
        """
        log = ('* Decoding',
               '- context(s): %s' % str(self.ctx),
               '- batch size: %d' % batch_size)
        logging.info('\n'.join(log) + '\n')

        states = self.create_states(docs)
        iterator = BatchIterator(states, batch_size, shuffle=False, label=False)

        st = time.time()
        self.decode_iter(iterator)
        et = time.time()
        logging.info('- time: %d (sec)\n' % (et - st))

    def decode_iter(self, iterator: BatchIterator):
        """
        :param iterator: the iterator to retrieve batches of feature vectors.

        Decodes all batches in the iterator.
        """
        device = self.decode_multiple_devices if isinstance(self.ctx, list) else self.decode_single_device
        device(iterator)

    def decode_single_device(self, iterator: BatchIterator):
        """
        :param iterator: the iterator to retrieve batches of feature vectors.

        Decodes all batches in the iterator with a single device.
        """
        state_idx = 0

        for batch in iterator:
            xs = nd.array(batch, self.ctx)
            outputs = self.model(xs)
            state_idx = process_outputs(iterator.states, outputs, state_idx)

    def decode_multiple_devices(self, iterator: BatchIterator):
        """
        :param iterator: the iterator to retrieve batches of feature vectors.

        Decodes all batches in the iterator with multiple devices.
        """
        state_idx = 0
        splits = []

        for batch in iterator:
            splits.append(nd.array(batch, self.ctx[len(splits)]))

            if len(splits) == len(self.ctx):
                outputs = [self.model(split) for split in splits]
                outputs = nd.concat(*outputs, dim=0)
                state_idx = process_outputs(iterator.states, outputs, state_idx)
                splits = []

        if splits:
            outputs = [self.model(split) for split in splits]
            outputs = nd.concat(*outputs, dim=0)
            process_outputs(iterator.states, outputs, state_idx)

    # override
    def evaluate(self, docs: Sequence[Document], batch_size: int = 2048):
        """
        :param docs: the sequence of input documents.
        :param batch_size: the batch size.

        Evaluates the current model with the input documents.
        """
        log = ('* Evaluating',
               '- context(s): %s' % str(self.ctx),
               '- batch size: %d' % batch_size)
        logging.info('\n'.join(log) + '\n')

        states = self.create_states(docs)
        iterator = BatchIterator(states, batch_size, shuffle=False, label=True)

        st = time.time()
        metric = self.evaluate_iter(iterator)
        et = time.time()
        logging.info('- time: %d (sec)\n' % (et - st))
        logging.info('%s\n' % str(metric))

    def evaluate_iter(self, iterator: BatchIterator) -> Type[EvalMetric]:
        """
        :param iterator: the iterator to retrieve batches of (feature vectors, labels).
        :return: the evaluation metric including evaluation statistics on the input states.

        Evaluates all batches in the iterator.
        """
        self.decode_iter(iterator)
        metric = self.eval_metric()
        for state in iterator.states: metric.update(state.document)
        return metric

    # override
    def train(self,
              trn_docs: Sequence[Document],
              dev_docs: Sequence[Document],
              model_path: str,
              trn_batch_size: int = 64,
              dev_batch_size: int = 2048,
              epoch: int = 100,
              loss: gluon.loss.Loss = None,
              optimizer: str = 'adagrad',
              optimizer_params: Dict[str, float] = None):
        """
        :param trn_docs: the sequence of documents for training.
        :param dev_docs: the sequence of documents for development (validation).
        :param model_path: the filepath where the trained model to be saved.
        :param trn_batch_size: the batch size applied to the training set.
        :param dev_batch_size: the batch size applied to the development set.
        :param epoch: the maximum number of epochs.
        :param loss: the loss function; if ``None``, it uses :meth:`mxnet.gluon.loss.SoftmaxCrossEntropyLoss`.
        :param optimizer: the type of the optimizer for training (:class:`mxnet.optimizer.Optimizer`).
        :param optimizer_params: the parameters for the optimizer.

        Trains and saves a model for this component.
        """
        # create instances
        trn_states = self.create_states(trn_docs)
        dev_states = self.create_states(dev_docs)

        # create iterators
        trn_iterator = BatchIterator(trn_states, trn_batch_size, shuffle=True, label=True)
        dev_iterator = BatchIterator(dev_states, dev_batch_size, shuffle=False, label=True)

        # create a trainer
        if loss is None: loss = gluon.loss.SoftmaxCrossEntropyLoss()
        trainer = gluon.Trainer(self.model.collect_params(), optimizer, optimizer_params)

        # train
        log = ('* Training',
               '- context(s): %s' % str(self.ctx),
               '- batch size: %d' % trn_batch_size,
               '- max epoch : %d' % epoch,
               '- loss      : %s' % str(loss),
               '- optimizer : %s -> %s' % (optimizer, optimizer_params))
        logging.info('\n'.join(log) + '\n')

        best_e, best_acc = -1, -1

        for e in range(1, epoch + 1):
            st = time.time()
            trn_acc = self._train(trn_iterator, loss, trainer)
            mt = time.time()
            dev_metric = self.evaluate_iter(dev_iterator)
            et = time.time()
            acc = dev_metric.get()

            if best_acc < acc:
                best_e, best_acc = e, acc
                self.save(model_path)

            print('%4d: trn-time: %d, dev-time: %d, trn-acc: %5.2f, dev-eval: %5.2f, best-dev: %5.2f @%4d' %
                  (e, mt - st, et - mt, trn_acc, dev_metric.get(), best_acc, best_e))

    def _train(self,
               iterator: BatchIterator,
               loss: gluon.loss.Loss,
               trainer: gluon.Trainer) -> float:
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
        device = self._train_multiple_devices if isinstance(self.ctx, list) else self._train_single_device
        correct = device(iterator, loss, trainer)
        return 100 * correct / iterator.total

    def _train_single_device(self,
                             iterator: BatchIterator,
                             loss: gluon.loss.Loss,
                             trainer: gluon.Trainer) -> int:
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

        for batch in iterator:
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

    def _train_multiple_devices(self,
                                iterator: BatchIterator,
                                loss: gluon.loss.Loss,
                                trainer: gluon.Trainer) -> int:
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

        return correct
