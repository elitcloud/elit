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
from typing import Union, Sequence, Dict

import mxnet as mx
from mxnet import nd, gluon, autograd

from elit.eval import EvalMetric
from elit.iterator import BatchIterator, NLPIterator
from elit.structure import Document

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
        :param batch_size: the size of batches to process at a time.
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

    def train_iter(self,
                   iterator: NLPIterator,
                   loss: gluon.loss.Loss,
                   trainer: gluon.Trainer) -> float:
        """
        :param iterator: the iterator to retrieve batches of (feature vectors, labels).
        :param loss: the loss function.
        :param trainer: the trainer.
        :return: the evaluation metric including evaluation statistics on the training states.

        Trains all batches in the iterator for one epoch.
        """
        device = self.train_multiple_devices if isinstance(self.ctx, list) else self.train_single_device
        return device(iterator, loss, trainer)

    def train_single_device(self,
                            iterator: NLPIterator,
                            loss: gluon.loss.Loss,
                            trainer: gluon.Trainer) -> float:
        """
        :param iterator: the iterator to retrieve batches of (feature vectors, labels).
        :param loss: the loss function.
        :param trainer: the trainer.
        :return: the training accuracy.

        Trains all batches in the iterator with a single device.
        """
        state_begin, total, correct = 0, 0, 0

        for batch in iterator:
            xs, ys = zip(*batch)
            total += len(ys)
            xs = nd.array(xs, self.ctx)
            ys = nd.array(ys, self.ctx)

            with autograd.record():
                outputs = self.model(xs)
                l = loss(outputs, ys)
                l.backward()

            trainer.step(xs.shape[0])
            state_begin = iterator.process(outputs.asnumpy(), state_begin)
            correct += len([1 for o, y in zip(mx.ndarray.argmax(outputs, axis=1), ys) if int(o.asscalar()) == int(y.asscalar())])

        return 100.0 * correct / total

    def train_multiple_devices(self,
                               iterator: NLPIterator,
                               loss: gluon.loss.Loss,
                               trainer: gluon.Trainer) -> float:
        """
        :param iterator: the iterator to retrieve batches of (feature vectors, labels).
        :param loss: the loss function.
        :param trainer: the trainer.
        :return: the training accuracy.

        Trains all batches in the iterator with a single device.
        """

        def train():
            with autograd.record():
                outputs = [self.model(x_split) for x_split in x_splits]
                losses = [loss(output, y_split) for output, y_split in zip(outputs, y_splits)]
                for l in losses: l.backward()

            trainer.step(sum(x_split.shape[0] for x_split in x_splits))
            begin, c = state_begin, 0
            for output, y_split in zip(outputs, y_splits):
                begin = iterator.process(output.asnumpy(), begin)
                c += len([1 for o, y in zip(mx.ndarray.argmax(output, axis=1), y_split) if int(o.asscalar()) == int(y.asscalar())])
            return begin, c

        state_begin, total, correct = 0, 0, 0
        x_splits, y_splits = [], []

        for batch in iterator:
            xs, ys = zip(*batch)
            total += len(ys)
            x_splits.append(nd.array(xs, self.ctx[len(x_splits)]))
            y_splits.append(nd.array(ys, self.ctx[len(y_splits)]))

            if len(x_splits) == len(self.ctx):
                t = train()
                state_begin = t[0]
                correct += t[1]
                x_splits, y_splits = [], []

        if x_splits: correct += train()[1]
        return 100.0 * correct / total

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

    def decode_iter(self, iterator: NLPIterator):
        """
        :param iterator: the iterator to retrieve batches of feature vectors.

        Decodes all batches in the iterator.
        """
        device = self.decode_multiple_devices if isinstance(self.ctx, list) else self.decode_single_device
        device(iterator)

    def decode_single_device(self, iterator: NLPIterator):
        """
        :param iterator: the iterator to retrieve batches of feature vectors.

        Decodes all batches in the iterator with a single device.
        """
        state_begin = 0

        for batch in iterator:
            xs = nd.array(batch, self.ctx)
            outputs = self.model(xs)
            state_begin = iterator.process(outputs.asnumpy(), state_begin)

    def decode_multiple_devices(self, iterator: BatchIterator):
        """
        :param iterator: the iterator to retrieve batches of feature vectors.

        Decodes all batches in the iterator with multiple devices.
        """

        def decode():
            outputs = [self.model(split) for split in splits]
            begin = state_begin
            for output in outputs: begin = iterator.process(output.asnumpy(), begin)
            return begin

        state_begin = 0
        splits = []

        for batch in iterator:
            splits.append(nd.array(batch, self.ctx[len(splits)]))

            if len(splits) == len(self.ctx):
                state_begin = decode()
                splits = []

        if splits: decode()

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

    def evaluate_iter(self, iterator: NLPIterator, decode=True) -> EvalMetric:
        """
        :param iterator: the iterator to retrieve batches of (feature vectors, labels).
        :param decode: if ``True``, decodes all the states in the iterator before evaluation.
        :return: the evaluation metric including evaluation statistics on the input states.

        Evaluates all batches in the iterator.
        """
        if decode: self.decode_iter(iterator)
        metric = self.eval_metric()
        for state in iterator.states: metric.update(state.document)
        return metric
