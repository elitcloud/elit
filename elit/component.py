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
from typing import Union, Sequence, Dict

import mxnet as mx
import numpy as np
from mxnet import nd, gluon, autograd
from mxnet.ndarray import NDArray

from elit.eval import EvalMetric
from elit.state import NLPState
from elit.structure import Document
from elit.utils.iterator import BatchIterator, NLPIterator

__author__ = 'Jinho D. Choi, Gary Lai'


class Component(abc.ABC):
    """
    :class:`Component` is an abstract class; any component developed in ELIT must inherit this class.

    Abstract methods to be implemented:
      - :meth:`Component.init`
      - :meth:`Component.load`
      - :meth:`Component.save`
      - :meth:`Component.train`
      - :meth:`Component.decode`
      - :meth:`Component.evaluate`
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
    def train(self, trn_data, dev_data, model_path: str, **kwargs):
        """
        :param trn_data: training data.
        :param dev_data: development (validation) data.
        :param model_path: the filepath where trained model(s) are to be saved.
        :param kwargs: custom parameters.

        Abstract class to train and save a model for this component.
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


class NLPComponent(Component):
    """
    :class:`NLPComponent` is an abstract class; any NLP component developed in ELIT must inherit this class.
    It is similar to :class:`Component` except that the type of training and development data is specified to :class:`elit.structure.Document`.

    Abstract methods to be implemented:
      - :meth:`Component.init`
      - :meth:`Component.load`
      - :meth:`Component.save`
      - :meth:`NLPComponent.train`
      - :meth:`NLPComponent.decode`
      - :meth:`NLPComponent.evaluate`
    """

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


class MXNetComponent(NLPComponent):
    """
    :class:`MXNetComponent` is an abstract class to implement NLP components using MXNet for machine learning.
    :meth:`MXNetComponent.decode` and :meth:`MXNetComponent.train` are defined in this class.

    Abstract methods to be implemented:
      - :meth:`Component.init`
      - :meth:`Component.load`
      - :meth:`Component.save`
      - :meth:`MXNetComponent.data_iterator`
      - :meth:`MXNetComponent.eval_metric`
      - :meth:`MXNetComponent.train_iter`
      - :meth:`MXNetComponent.decode_iter`
    """

    def __init__(self, ctx: Union[mx.Context, Sequence[mx.Context]]):
        """
        :param ctx: the (list of) device context(s) for :class:`mxnet.gluon.Block`.
        """
        self.ctx = ctx
        self.model: gluon.Block = None

    @abc.abstractmethod
    def data_iterator(self, documents: Sequence[Document], batch_size: int, shuffle: bool, label: bool, **kwargs) -> NLPIterator:
        """
        :param documents: the sequence of input documents.
        :param batch_size: the size of mini-batches.
        :param shuffle: if ``True``, shuffle instances for every epoch; otherwise, no shuffle.
        :param label: if ``True``, each instance is a tuple of (feature vector, label); otherwise, it is just a feature vector.
        :param kwargs: custom parameters.
        :return: the iterator to retrieve batches of training or decoding instances.

        Abstract method.
        """
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))

    @abc.abstractmethod
    def eval_metric(self, **kwargs) -> EvalMetric:
        """
        :param kwargs: custom parameters.
        :return: the evaluation metric for this component.

        Abstract method.
        """
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))

    @abc.abstractmethod
    def train_iter(self,
                   iterator: NLPIterator,
                   loss: gluon.loss.Loss,
                   trainer: gluon.Trainer,
                   **kwargs) -> float:
        """
        :param iterator: the iterator to retrieve batches of (feature vectors, labels).
        :param loss: the loss function.
        :param trainer: the trainer.
        :param kwargs: custom parameters.
        :return: the training accuracy.

        Abstract method to train all batches in the iterator for one epoch.
        """
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))

    @abc.abstractmethod
    def decode_iter(self, iterator: NLPIterator, **kwargs):
        """
        :param iterator: the iterator to retrieve batches of feature vectors.
        :param kwargs: custom parameters.

        Abstract method to decode all batches in the iterator.
        """
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))

    def evaluate_iter(self, iterator: NLPIterator, **kwargs) -> EvalMetric:
        """
        :param iterator: the iterator to retrieve batches of (feature vectors, labels).
        :param kwargs: custom parameters.
        :return: the evaluation metric including evaluation statistics on the input states.

        Evaluates all batches in the iterator.
        """
        self.decode_iter(iterator, kwargs)
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
              optimizer_params: Dict[str, float] = None,
              **kwargs):
        """
        :param trn_docs: the sequence of documents for training.
        :param dev_docs: the sequence of documents for development (validation).
        :param model_path: the filepath where the trained model to be saved.
        :param trn_batch_size: the batch size applied to the training data.
        :param dev_batch_size: the batch size applied to the development data.
        :param epoch: the maximum number of epochs.
        :param loss: the `loss function <https://mxnet.incubator.apache.org/api/python/gluon/loss.html?highlight=softmaxcrossentropyloss#gluon-loss-api>`_; if not specified, ``SoftmaxCrossEntropyLoss`` is used.
        :param optimizer: the type of the `optimizer <https://mxnet.incubator.apache.org/api/python/optimization/optimization.html?highlight=optimiz#the-mxnet-optimizer-package>`_ for training.
        :param optimizer_params: the parameters for the optimizer.
        :param kwargs: custom parameters.

        Trains and saves a model for this component.
        """
        # create iterators
        trn_iterator = self.data_iterator(trn_docs, trn_batch_size, shuffle=True, label=True)
        dev_iterator = self.data_iterator(trn_docs, dev_batch_size, shuffle=False, label=True)

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
            trn_acc = self.train_iter(trn_iterator, loss, trainer)
            mt = time.time()
            dev_metric = self.evaluate_iter(dev_iterator)
            et = time.time()
            acc = dev_metric.get()

            if best_acc < acc:
                best_e, best_acc = e, acc
                self.save(model_path)

            print('%4d: trn-time: %d, dev-time: %d, trn-acc: %5.2f, dev-eval: %5.2f, best-dev: %5.2f @%4d' %
                  (e, mt - st, et - mt, trn_acc, dev_metric.get(), best_acc, best_e))

    # override
    def decode(self, docs: Sequence[Document], batch_size: int = 2048, **kwargs):
        """
        :param docs: the sequence of input documents.
        :param batch_size: the batch size.
        :param kwargs: custom parameters.

        Processes the input documents and saves the predicted labels to the input documents.
        """
        log = ('* Decoding',
               '- context(s): %s' % str(self.ctx),
               '- batch size: %d' % batch_size)
        logging.info('\n'.join(log) + '\n')

        iterator = self.data_iterator(docs, batch_size, shuffle=False, label=False)
        st = time.time()
        self.decode_iter(iterator)
        et = time.time()
        logging.info('- time: %d (sec)\n' % (et - st))

    # override
    def evaluate(self, docs: Sequence[Document], batch_size: int = 2048, **kwargs):
        """
        :param docs: the sequence of input documents.
        :param batch_size: the batch size.
        :param kwargs: custom parameters.

        Evaluates the current model with the input documents.
        """
        log = ('* Evaluating',
               '- context(s): %s' % str(self.ctx),
               '- batch size: %d' % batch_size)
        logging.info('\n'.join(log) + '\n')

        iterator = self.data_iterator(docs, batch_size, shuffle=False, label=True)
        st = time.time()
        metric = self.evaluate_iter(iterator)
        et = time.time()
        logging.info('- time: %d (sec)\n' % (et - st))
        logging.info('%s\n' % str(metric))


class BatchComponent(MXNetComponent):
    """
    :class:`BatchComponent` is an abstract class to implement NLP components that process all states in batch
    such that it assumes every state is independent to one another.
    :meth:`BatchComponent.train_iter` and :meth:`BatchComponent.decode_iter` are defined in this class.

    Abstract methods to be implemented:
      - :meth:`Component.init`
      - :meth:`Component.load`
      - :meth:`Component.save`
      - :meth:`MXNetComponent.eval_metric`
      - :meth:`BatchComponent.data_iterator`
    """

    def __init__(self, ctx: Union[mx.Context, Sequence[mx.Context]]):
        """
        :param ctx: the (list of) device context(s) for :class:`mxnet.gluon.Block`.
        """
        super().__init__(ctx)

    @abc.abstractmethod
    def data_iterator(self, documents: Sequence[Document], batch_size: int, shuffle: bool, label: bool, **kwargs) -> BatchIterator:
        """
        :param documents: the sequence of input documents.
        :param batch_size: the size of mini-batches.
        :param shuffle: if ``True``, shuffle instances for every epoch; otherwise, no shuffle.
        :param label: if ``True``, each instance is a tuple of (feature vector, label); otherwise, it is just a feature vector.
        :param kwargs: custom parameters.
        :return: the iterator to retrieve batches of training or decoding instances.

        Abstract method.
        """
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))

    # override
    def train_iter(self,
                   iterator: BatchIterator,
                   loss: gluon.loss.Loss,
                   trainer: gluon.Trainer,
                   **kwargs) -> float:
        """
        :param iterator: the iterator to retrieve batches of (feature vectors, labels).
        :param loss: the loss function.
        :param trainer: the trainer.
        :param kwargs: custom parameters.
        :return: the training accuracy.

        Trains all batches in the iterator for one epoch.
        """
        device = self.train_multiple_devices if isinstance(self.ctx, list) else self.train_single_device
        correct = device(iterator, loss, trainer)
        return 100 * correct / iterator.total

    def train_single_device(self,
                            iterator: BatchIterator,
                            loss: gluon.loss.Loss,
                            trainer: gluon.Trainer) -> int:
        """
        :param iterator: the iterator to retrieve batches of (feature vectors, labels).
        :param loss: the loss function.
        :param trainer: the trainer.
        :return: the number of correctly classified instances.

        Trains all batches in the iterator with a single device.
        """
        correct = 0

        for batch in iterator:
            xs, ys = zip(*batch)
            xs = nd.array(xs, self.ctx)
            ys = nd.array(ys, self.ctx)

            with autograd.record():
                outputs = self.model(xs)
                l = loss(outputs, ys)
                l.backward()

            trainer.step(xs.shape[0])
            correct += len([1 for o, y in zip(mx.ndarray.argmax(outputs, axis=1), ys) if int(o.asscalar()) == int(y.asscalar())])

        return correct

    def train_multiple_devices(self,
                               iterator: BatchIterator,
                               loss: gluon.loss.Loss,
                               trainer: gluon.Trainer) -> int:
        """
        :param iterator: the iterator to retrieve batches of (feature vectors, labels).
        :param loss: the loss function.
        :param trainer: the trainer.
        :return: the number of correctly classified instances.

        Trains all batches in the iterator with a single device.
        """

        def train():
            with autograd.record():
                outputs = [self.model(x_split) for x_split in x_splits]
                losses = [loss(output, y_split) for output, y_split in zip(outputs, y_splits)]
                for l in losses: l.backward()

            c = 0
            trainer.step(sum(x_split.shape[0] for x_split in x_splits))
            for output, y_split in zip(outputs, y_splits):
                c += len([1 for o, y in zip(mx.ndarray.argmax(output, axis=1), y_split) if int(o.asscalar()) == int(y.asscalar())])
            return c

        x_splits, y_splits = [], []
        correct = 0

        for batch in iterator:
            xs, ys = zip(*batch)
            x_splits.append(nd.array(xs, self.ctx))
            y_splits.append(nd.array(ys, self.ctx))

            if len(x_splits) == len(self.ctx):
                correct += train()
                x_splits, y_splits = [], []

        if x_splits:
            correct += train()

        return correct

    # override
    def decode_iter(self, iterator: BatchIterator, **kwargs):
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
            state_idx = self._process_outputs(iterator.states, outputs, state_idx)

    def decode_multiple_devices(self, iterator: BatchIterator):
        """
        :param iterator: the iterator to retrieve batches of feature vectors.

        Decodes all batches in the iterator with multiple devices.
        """

        def decode():
            outputs = [self.model(split) for split in splits]
            outputs = nd.concat(*outputs, dim=0)
            return self._process_outputs(iterator.states, outputs, state_idx)

        state_idx = 0
        splits = []

        for batch in iterator:
            splits.append(nd.array(batch, self.ctx[len(splits)]))

            if len(splits) == len(self.ctx):
                state_idx = decode()
                splits = []

        if splits:
            decode()

    @classmethod
    def _process_outputs(cls, states: Sequence[NLPState], outputs: Union[NDArray, np.ndarray], state_idx) -> int:
        """
        :param states: the sequence of input states.
        :param outputs: the 2D matrix where each row contains the prediction scores of the corresponding state.
        :param state_idx: the index of the state in the sequence to begin the process with.
        :return: the index of the state to be processed next.

        Processes through the states and assigns the outputs accordingly.
        """
        i = 0

        for state in islice(states, state_idx):
            while state.has_next():
                if i == len(outputs): return state_idx
                state.process(outputs[i])
                i += 1

            state_idx += 1

        return state_idx
