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
import random
import time
from itertools import islice
from typing import List


import mxnet as mx
import numpy as np
from mxnet import nd, gluon, autograd

from elit.eval import EvalMetric
from elit.state import NLPState, BatchState, SequenceState
from elit.structure import Document

__author__ = 'Jinho D. Choi, Gary Lai'


class Component(abc.ABC):
    """
    Component provides an abstract class to implement a component.
    Any component deployed to ELIT must inherit this class.
    Abstract methods to be implemented: init, load, save, decode, train.
    """

    @abc.abstractmethod
    def init(self, **kwargs):
        """
        Initializes this component.
        :param kwargs: custom parameters.
        """
        raise NotImplementedError("Not implemented")

    @abc.abstractmethod
    def load(self, model_path: str, **kwargs):
        """
        Loads the pre-trained model to this component.
        :param model_path: a filepath to the model.
        :param kwargs: custom parameters.
        """
        raise NotImplementedError("Not implemented")

    @abc.abstractmethod
    def save(self, model_path: str, **kwargs):
        """
        Saves the trained model.
        :param model_path: a filepath to the model to be saved.
        :param kwargs: custom parameters.
        """
        raise NotImplementedError("Not implemented")

    @abc.abstractmethod
    def decode(self, data, **kwargs):
        """
        Processes the input data and saves the decoding results to the input data.
        :param data: input data to be decoded.
        :param kwargs: custom parameters.
        """
        raise NotImplementedError("Not implemented")

    @abc.abstractmethod
    def train(self, trn_data, dev_data, model_path: str, **kwargs):
        """
        Trains a model for this component.
        :param trn_data: training data.
        :param dev_data: development (validation) data.
        :param model_path: a filepath to the model to be saved.
        :param kwargs: custom parameters
        """
        raise NotImplementedError("Not implemented")


class NLPComponent(Component):
    """
    NLPComponent provides an abstract class to implement an NLP component.
    Any NLP component deployed to ELIT must inherit this class.
    Abstract methods to be implemented: init, load, save, decode, train.
    """

    @abc.abstractmethod
    def decode(self, docs: List[Document], **kwargs):
        """
        Processes the input documents and saves the decoding results to the input documents.
        :param docs: a list of input documents to be decoded.
        :param kwargs: custom parameters.
        """
        pass

    @abc.abstractmethod
    def train(self, trn_docs: List[Document], dev_docs: List[Document], model_path: str, **kwargs):
        """
        Trains a model for this component.
        :param trn_docs: a list of documents for training.
        :param dev_docs: a list of documents for development (validation).
        :param model_path: a filepath to the model to be saved.
        :param kwargs: custom parameters
        """
        pass


class MXNetComponent(NLPComponent):
    """
    MXNetNLPComponent provides an abstract class to implement a machine-learning based NLP component using MXNet.
    Abstract methods to be implemented: init, load, save, create_states, finalize, eval_metric, _decode, _evaluate, _train.
    """

    def __init__(self, ctx: mx.Context):
        """
        :param ctx: a device context.
        """
        self.ctx = ctx
        self.model = None

    @abc.abstractmethod
    def create_states(self, documents: List[Document]) -> List[NLPState]:
        """
        :param documents: a list of input documents.
        :return: the list of initial states corresponding to the input documents.
        """
        pass

    @abc.abstractmethod
    def finalize(self, state: NLPState):
        """
        Finalizes by saving the predicted labels to the input document once decoding is done.
        """
        pass

    @abc.abstractmethod
    def eval_metric(self) -> EvalMetric:
        """
        :return: the evaluation metric for this component.
        """
        pass

    @abc.abstractmethod
    def _decode(self,
                states: List[NLPState],
                batch_size: int,
                xs: List[np.ndarray] = None):
        """
        :param states: a list of input states.
        :param batch_size: the batch size .
        :param xs: a list of feature vectors extracted from the input states (batch mode only).
        """
        pass

    @abc.abstractmethod
    def _evaluate(self,
                  states: List[NLPState],
                  batch_size: int,
                  xs: List[np.ndarray] = None) -> EvalMetric:
        """
        :param states: a list of input states.
        :param batch_size: the batch size.
        :param xs: a list of feature vectors extracted from the input states (batch mode only).
        :return: the metric including evaluation statistics on the input.
        """
        pass

    @abc.abstractmethod
    def _train(self,
               states: List[NLPState],
               batch_size: int,
               loss: gluon.loss.Loss,
               trainer: gluon.Trainer,
               xs: List[np.ndarray] = None,
               ys: List[np.ndarray] = None) -> int:
        """
        :param states: a list of states including documents for training.
        :param batch_size: the batch size.
        :param loss: the loss function.
        :param trainer: the trainer.
        :param xs: a list of feature vectors (batch mode only).
        :param ys: a list of gold-standard class IDs corresponding to the feature vectors (batch mode only).
        :return: the number of correctly classified instances.
        """
        pass

    # override
    def decode(self, docs: List[Document], batch_size: int = 2048):
        """
        :param docs: a list of input documents.
        :param batch_size: the batch size.
        """
        states = self.create_states(docs)
        self._decode(states, batch_size)

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
            trn_acc = self._train(trn_states, trn_batch, loss, trainer, trn_xs, trn_ys)
            mt = time.time()
            dev_metric = self._evaluate(dev_states, dev_batch, dev_xs)
            et = time.time()
            if best_eval < dev_metric.get():
                best_e, best_eval = e, dev_metric.get()
                self.save(model_path)

            print('%4d: trn-time: %d, dev-time: %d, trn-acc: %5.2f, dev-eval: %5.2f, best-dev: %5.2f @%4d' %
                  (e, mt - st, et - mt, trn_acc, dev_metric.get(), best_eval, best_e))


class BatchComponent(MXNetComponent):
    """
    BatchComponent provides an abstract class to implement an NLP component in batch mode.
    See the description of :class:`elit.state.BatchState` for more details about batch mode.
    Abstract methods to be implemented: init, load, save, create_states, finalize, eval_metric.
    """

    def __init__(self, ctx: mx.Context):
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
        batches = gluon.data.DataLoader(xs, batch_size=batch_size, shuffle=False)
        begin = 0

        for x_batch in batches:
            x_batch = x_batch.as_in_context(self.ctx)
            outputs = self.model(x_batch).asnumpy()
            o_begin = 0

            for state in islice(states, begin, None):
                o_begin = state.assign(outputs, o_begin)
                begin += 1
                if o_begin >= len(outputs): break

        for state in states:
            self.finalize(state)

    # override
    def _evaluate(self,
                  states: List[BatchState],
                  batch_size: int,
                  xs: List[np.ndarray] = None) -> EvalMetric:
        """
        :param states: a list of input states.
        :param batch_size: the batch size.
        :param xs: a list of feature vectors extracted from the input states (optional).
        :return: the metric including evaluation statistics on the input.
        """
        if xs is None: xs = [x for state in states for x, y in state]
        self._decode(states, batch_size, xs)
        metric = self.eval_metric()
        for state in states: metric.update(state.document)
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
        batches = gluon.data.DataLoader(gluon.data.ArrayDataset(xs, ys), batch_size=batch_size, shuffle=True)
        correct = 0

        for x_batch, y_batch in batches:
            x_batch = x_batch.as_in_context(self.ctx)
            y_batch = y_batch.as_in_context(self.ctx)

            with autograd.record():
                outputs = self.model(x_batch)
                l = loss(outputs, y_batch)
                l.backward()

            trainer.step(x_batch.shape[0])
            correct += len([1 for o, y in zip(mx.ndarray.argmax(outputs, axis=1), y_batch) if int(o.asscalar()) == int(y.asscalar())])

        return 100 * correct / len(ys)


class SequenceComponent(MXNetComponent):
    """
    SequenceComponent provides an abstract class to implement an NLP component in sequence mode.
    See the description of :class:`elit.state.SequenceState` for more details about sequence mode.
    Abstract methods to be implemented: init, load, save, create_states, finalize, eval_metric.
    """

    def __init__(self, ctx: mx.Context):
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

        while tmp:
            xs = nd.array([state.x for state in tmp])
            batches = gluon.data.DataLoader(xs, batch_size=batch_size, shuffle=False)
            begin = 0

            for x_batch in batches:
                x_batch = x_batch.as_in_context(self.ctx)
                outputs = self.model(x_batch).asnumpy()

                for i, output in enumerate(outputs):
                    states[begin + i].process(output)

                begin += len(outputs)

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
               states: List[BatchState],
               batch_size: int,
               loss: gluon.loss.Loss,
               trainer: gluon.Trainer,
               xs: List[np.ndarray] = None,
               ys: List[np.ndarray] = None) -> int:
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

        while tmp:
            random.shuffle(tmp)
            xs = [state.x for state in tmp]
            ys = [state.y for state in tmp]

            batches = gluon.data.DataLoader(gluon.data.ArrayDataset(xs, ys), batch_size=batch_size)
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
                for i, output in enumerate(outputs): states[begin + i].process(output.asnumpy())
                begin += len(outputs)

            tmp = [state for state in tmp if state.has_next]

        for state in states: state.init()
        return correct
