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
from typing import Union, Sequence, Dict, Any

import mxnet as mx
from mxnet import nd, gluon, autograd
from mxnet.ndarray import NDArray

from elit.eval import EvalMetric
from elit.util.iterator import BatchIterator, NLPIterator
from elit.util.structure import Document

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

        Initializes this component.
        """
        raise NotImplementedError(
            '%s.%s()' %
            (self.__class__.__name__, inspect.stack()[0][3]))

    @abc.abstractmethod
    def load(self, model_path: str, **kwargs):
        """
        :param model_path: the filepath where a model can be loaded.

        Loads a model for this component from the filepath.
        """
        raise NotImplementedError(
            '%s.%s()' %
            (self.__class__.__name__, inspect.stack()[0][3]))

    @abc.abstractmethod
    def save(self, model_path: str, **kwargs):
        """
        :param model_path: the filepath where the current model can be saved.

        Saves the current model of this component to the filepath.
        """
        raise NotImplementedError(
            '%s.%s()' %
            (self.__class__.__name__, inspect.stack()[0][3]))

    @abc.abstractmethod
    def train(
            self,
            trn_data: Any,
            dev_data: Any,
            model_path: str,
            **kwargs) -> float:
        """
        :param trn_data: training data.
        :param dev_data: development (validation) data.
        :param model_path: the filepath where the trained model(s) are to be saved.
        :return: the best score form the development data.

        Trains a model for this component and saves the model to the filepath.
        """
        raise NotImplementedError(
            '%s.%s()' %
            (self.__class__.__name__, inspect.stack()[0][3]))

    @abc.abstractmethod
    def decode(self, data: Any, **kwargs):
        """
        :param data: input data.

        Processes the input data, make predictions, and saves the predicted labels back to the input data.
        """
        raise NotImplementedError(
            '%s.%s()' %
            (self.__class__.__name__, inspect.stack()[0][3]))

    @abc.abstractmethod
    def evaluate(self, data: Any, **kwargs):
        """
        :param data: input data.

        Evaluates the current model of this component with the input data.
        """
        raise NotImplementedError(
            '%s.%s()' %
            (self.__class__.__name__, inspect.stack()[0][3]))


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
    def train(
            self,
            trn_docs: Sequence[Document],
            dev_docs: Sequence[Document],
            model_path: str,
            **kwargs) -> float:
        """
        :param trn_docs: the sequence of documents for training.
        :param dev_docs: the sequence of documents for development (validation).
        :param model_path: the filepath where trained model(s) are to be saved.
        :return: the best score form the development data.

        Trains a model for this component and saves the model to the filepath.
        """
        raise NotImplementedError(
            '%s.%s()' %
            (self.__class__.__name__, inspect.stack()[0][3]))

    @abc.abstractmethod
    def decode(self, docs: Sequence[Document], **kwargs):
        """
        :param docs: the sequence of input documents.

        Processes the input documents, make predictions, and saves the predicted labels back to the input documents.
        """
        raise NotImplementedError(
            '%s.%s()' %
            (self.__class__.__name__, inspect.stack()[0][3]))

    @abc.abstractmethod
    def evaluate(self, docs: Sequence[Document], **kwargs):
        """
        :param docs: the sequence of input documents.

        Evaluates the current model of this component with the input documents.
        """
        raise NotImplementedError(
            '%s.%s()' %
            (self.__class__.__name__, inspect.stack()[0][3]))


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
        self.model = None

    @abc.abstractmethod
    def data_iterator(
            self,
            documents: Sequence[Document],
            batch_size: int,
            shuffle: bool,
            label: bool,
            **kwargs) -> NLPIterator:
        """
        :param documents: the sequence of input documents.
        :param batch_size: the size of batches to process at a time.
        :param shuffle: if ``True``, shuffle instances for every epoch; otherwise, no shuffle.
        :param label: if ``True``, each instance is a tuple of (feature vector, label); otherwise, it is just a feature vector.
        :return: the iterator to retrieve batches of training or decoding instances.
        """
        raise NotImplementedError(
            '%s.%s()' %
            (self.__class__.__name__, inspect.stack()[0][3]))

    @abc.abstractmethod
    def eval_metric(self, **kwargs) -> EvalMetric:
        """
        :return: the evaluation metric for this component.
        """
        raise NotImplementedError(
            '%s.%s()' %
            (self.__class__.__name__, inspect.stack()[0][3]))

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
              **kwargs) -> float:
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

        Trains and saves a model for this component.
        """
        log = ('Configuration',
               '- context(s): %s' % str(self.ctx),
               '- batch size: %d' % trn_batch_size,
               '- max epoch : %d' % epoch,
               '- loss func : %s' % str(loss),
               '- optimizer : %s <- %s' % (optimizer, optimizer_params))
        logging.info('\n'.join(log))

        # create iterators
        logging.info('Iterators')

        st = time.time()
        trn_iterator = self.data_iterator(
            trn_docs, trn_batch_size, shuffle=True, label=True, **kwargs)
        et = time.time()
        logging.info('- trn: %s (%d sec)' % (str(trn_iterator), et - st))

        st = time.time()
        dev_iterator = self.data_iterator(
            dev_docs,
            dev_batch_size,
            shuffle=False,
            label=True,
            **kwargs)
        et = time.time()
        logging.info('- dev: %s (%d sec)' % (str(dev_iterator), et - st))

        # create a trainer
        if loss is None:
            loss = gluon.loss.SoftmaxCrossEntropyLoss()
        trainer = gluon.Trainer(
            self.model.collect_params(),
            optimizer,
            optimizer_params)

        # train
        logging.info('Training')
        best_e, best_eval = -1, -1

        for e in range(1, epoch + 1):
            st = time.time()
            trn_acc = self.train_iter(trn_iterator, loss, trainer)
            mt = time.time()
            dev_metric = self.evaluate_iter(dev_iterator)
            et = time.time()

            ev = dev_metric.get()
            if best_eval < ev:
                best_e, best_eval = e, ev
                self.save(model_path)

            logging.info(
                '%4d: trn-time: %d, dev-time: %d, trn-acc: %6.2f, dev-eval: %5.2f, best-dev: %5.2f @%4d' %
                (e, mt - st, et - mt, trn_acc, dev_metric.get(), best_eval, best_e))

        return best_eval

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
        device = self.train_multiple_devices if isinstance(
            self.ctx, list) else self.train_single_device
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
        others = []

        for batch in iterator:
            xs, ys = zip(*batch)
            total += len(ys)
            xs = nd.array(
                xs, self.ctx) if iterator.x1 else [
                nd.array(
                    x, self.ctx) for x in zip(
                    *xs)]
            ys = nd.array(ys, self.ctx)

            with autograd.record():
                t = self.model(xs) if iterator.x1 else self.model(*xs)
                if isinstance(t, NDArray):
                    output = t
                else:
                    output = t[0]
                    others = t[1:]

                l = loss(output, ys)
                l.backward()

            trainer.step(xs.shape[0])
            iterator.process(output, *others)
            correct += len([1 for o,
                            y in zip(mx.ndarray.argmax(output,
                                                       axis=1),
                                     ys) if int(o.asscalar()) == int(y.asscalar())])

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
                ts = [
                    self.model(x_split) if iterator.x1 else self.model(
                        *x_split) for x_split in x_splits]
                if isinstance(ts[0], NDArray):
                    outputs = ts
                    others = empty
                else:
                    outputs = [t[0] for t in ts]
                    others = [t[1:] for t in ts]

                losses = [loss(output, y_split)
                          for output, y_split in zip(outputs, y_splits)]
                for l in losses:
                    l.backward()

            trainer.step(
                sum(x_split.shape[0] for x_split in x_splits), ignore_stale_grad=True)

            c, i = 0, 0
            for output, y_split in zip(outputs, y_splits):
                if others:
                    iterator.process(output, *others[i])
                else:
                    iterator.process(output)
                c += len([1 for o, y in zip(mx.ndarray.argmax(output, axis=1),
                                            y_split) if int(o.asscalar()) == int(y.asscalar())])
                i += 1
            return c

        x_splits, y_splits, empty = [], [], []
        total, correct = 0, 0

        for batch in iterator:
            if batch:
                xs, ys = zip(*batch)
                total += len(ys)
                ctx = self.ctx[len(x_splits)]
                x_splits.append(
                    nd.array(
                        xs, ctx) if iterator.x1 else [
                        nd.array(
                            x, ctx) for x in zip(
                            *xs)])
                y_splits.append(nd.array(ys, ctx))

            if len(x_splits) == len(self.ctx) or not batch:
                correct += train()
                x_splits, y_splits = [], []

        if x_splits:
            correct += train()
        return 100.0 * correct / total

    # override
    def decode(
            self,
            docs: Sequence[Document],
            batch_size: int = 2048,
            **kwargs):
        """
        :param docs: the sequence of input documents.
        :param batch_size: the batch size.

        Processes the input documents and saves the predicted labels to the input documents.
        """
        log = ('Decoding',
               '- context(s): %s' % str(self.ctx),
               '- batch size: %d' % batch_size)
        logging.info('\n'.join(log))

        iterator = self.data_iterator(
            docs, batch_size, shuffle=False, label=False)
        st = time.time()
        self.decode_iter(iterator)
        et = time.time()
        logging.info('- time: %d (sec)' % (et - st))

    def decode_iter(self, iterator: NLPIterator):
        """
        :param iterator: the iterator to retrieve batches of feature vectors.

        Decodes all batches in the iterator.
        """
        device = self.decode_multiple_devices if isinstance(
            self.ctx, list) else self.decode_single_device
        device(iterator)

    def decode_single_device(self, iterator: NLPIterator):
        """
        :param iterator: the iterator to retrieve batches of feature vectors.

        Decodes all batches in the iterator with a single device.
        """
        others = []

        for batch in iterator:
            if not isinstance(batch[0], NDArray):
                batch, _ = zip(*batch)
            xs = nd.array(batch, self.ctx)
            t = self.model(xs)
            if isinstance(t, NDArray):
                output = t
            else:
                output = t[0]
                others = t[1:]

            iterator.process(output, *others)

    def decode_multiple_devices(self, iterator: BatchIterator):
        """
        :param iterator: the iterator to retrieve batches of feature vectors.

        Decodes all batches in the iterator with multiple devices.
        """

        def decode():
            ts = [
                self.model(split) if iterator.x1 else self.model(
                    *split) for split in splits]
            if isinstance(ts[0], NDArray):
                outputs = ts
                others = empty
            else:
                outputs = [t[0] for t in ts]
                others = [t[1:] for t in ts]

            for i, output in enumerate(outputs):
                if others:
                    iterator.process(output, *others[i])
                else:
                    iterator.process(output)

        splits, empty = [], []

        for batch in iterator:
            if batch:
                if iterator.label:
                    batch, _ = zip(*batch)
                ctx = self.ctx[len(splits)]
                splits.append(
                    nd.array(
                        batch,
                        ctx) if iterator.x1 else [
                        nd.array(
                            x,
                            ctx) for x in zip(
                            *batch)])

            if len(splits) == len(self.ctx) or not batch:
                decode()
                splits = []

        if splits:
            decode()

    # override
    def evaluate(
            self,
            docs: Sequence[Document],
            batch_size: int = 2048,
            **kwargs):
        """
        :param docs: the sequence of input documents.
        :param batch_size: the batch size.
        :param kwargs: custom parameters.

        Evaluates the current model with the input documents.
        """
        log = ('Evaluating',
               '- context(s): %s' % str(self.ctx),
               '- batch size: %d' % batch_size)
        logging.info('\n'.join(log))

        iterator = self.data_iterator(
            docs, batch_size, shuffle=False, label=True)
        st = time.time()
        metric = self.evaluate_iter(iterator)
        et = time.time()
        logging.info('- time: %d (sec)' % (et - st))
        logging.info('%s' % str(metric))

    def evaluate_iter(self, iterator: NLPIterator, decode=True) -> EvalMetric:
        """
        :param iterator: the iterator to retrieve batches of (feature vectors, labels).
        :param decode: if ``True``, decodes all the states in the iterator before evaluation.
        :return: the evaluation metric including evaluation statistics on the input states.

        Evaluates all batches in the iterator.
        """
        if decode:
            self.decode_iter(iterator)
        metric = self.eval_metric()
        for state in iterator.states:
            metric.update(state.document)
        return metric
