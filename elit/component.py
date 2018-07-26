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
import pickle
import random
import time
from itertools import islice
from types import SimpleNamespace
from typing import List


import mxnet as mx
import numpy as np
from mxnet import nd, gluon, autograd

from elit.model import FFNNModel, output_namespace, input_namespace
from elit.state import NLPState, BatchState, SequenceState
from elit.structure import Document
from elit.util import pkl, gln, EvalMetric
from elit.vsm import LabelMap, X_ANY

__author__ = 'Jinho D. Choi, Gary Lai'


class Component(abc.ABC):
    """
    Component provides an abstract class to implement a component.
    Any component deployed to ELIT must inherit this class.
    """

    @abc.abstractmethod
    def init(self, **kwargs):
        """
        Initializes this component.
        :param kwargs: custom parameters.
        """
        pass

    @abc.abstractmethod
    def load(self, model_path: str, **kwargs):
        """
        Loads the pre-trained model to this component.
        :param model_path: a filepath to the model.
        :param kwargs: custom parameters.
        """
        pass

    @abc.abstractmethod
    def save(self, model_path: str, **kwargs):
        """
        Saves the trained model.
        :param model_path: a filepath to the model to be saved.
        :param kwargs: custom parameters.
        """
        pass

    @abc.abstractmethod
    def decode(self, data, **kwargs):
        """
        Processes the input data and saves the decoding results to the input data.
        :param data: input data to be decoded.
        :param kwargs: custom parameters.
        """
        pass

    @abc.abstractmethod
    def train(self, trn_data, dev_data, model_path: str, **kwargs):
        """
        Trains a model for this component.
        :param trn_data: training data.
        :param dev_data: development (validation) data.
        :param model_path: a filepath to the model to be saved.
        :param kwargs: custom parameters
        """
        pass


class NLPComponent(Component):
    """
    NLPComponent provides an abstract class to implement an NLP component.
    Any NLP component deployed to ELIT must inherit this class.
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
    """

    def __init__(self, ctx: mx.Context = None):
        """
        :param ctx: a device context (default: mxnet.cpu()).
        """
        self.ctx = mx.cpu() if ctx is None else ctx
        self.label_map = None
        self.model = None

    @abc.abstractmethod
    def create_states(self, documents: List[Document]) -> List[NLPState]:
        """
        :param documents: a list of input documents.
        :return: the list of initial states corresponding to the input documents.
        """
        pass

    @abc.abstractmethod
    def finalize(self, document: Document):
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
        if isinstance(trn_states[0], BatchState):
            trn_xs, trn_ys = zip(*[(x, y) for state in trn_states for x, y in state])
            dev_xs = [x for state in dev_states for x, _ in state]
        else:
            trn_xs, trn_ys, dev_xs = None, None, None

        # create a trainer
        if loss is None: loss = gluon.loss.SoftmaxCrossEntropyLoss()
        trainer = gluon.Trainer(self.model.collect_params(), optimizer, {'learning_rate': learning_rate, 'wd': weight_decay})

        # train
        log = ('* Training',
               '- train batch   : %d' % trn_batch,
               '- epoch         : %d' % epoch,
               '- loss          : %d' % str(loss),
               '- optimizer     : %s' % optimizer,
               '- learning rate : %f' % learning_rate,
               '- weight decay  : %f' % weight_decay)
        print('\n'.join(log))

        best_e, best_eval = -1, -1

        for e in range(1, epoch + 1):
            st = time.time()
            trn_correct = self._train(trn_states, trn_batch, loss, trainer, trn_xs, trn_ys)
            mt = time.time()
            dev_metric = self._evaluate(dev_states, dev_batch, dev_xs)
            et = time.time()
            if best_eval < dev_metric.get():
                best_e, best_eval = e, dev_metric.get()
                self.save(model_path)

            print('%4d: trn-time: %d, dev-time: %d, trn-acc: %5.2f, dev-eval: %5.2f, num-class: %d, best-dev: %5.2f @%4d' %
                  (e, mt - st, et - mt, 100.0 * trn_correct / len(trn_ys), dev_metric.get(), len(self.label_map), best_eval, best_e))


class BatchComponent(MXNetComponent):
    """
    BatchComponent provides an abstract class to implement an NLP component in batch mode.
    See the description of :class:`elit.state.BatchState` for more details about batch mode.
    """

    def __init__(self, ctx: mx.Context = None):
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
            self.finalize(state.document)

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
               ys: List[np.ndarray] = None) -> int:
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
            correct += int(sum(mx.ndarray.argmax(outputs, axis=0) == y_batch).asscalar())

        return correct


class SequenceComponent(MXNetComponent):
    """
    SequenceComponent provides an abstract class to implement an NLP component in sequence mode.
    See the description of :class:`elit.state.SequenceState` for more details about sequence mode.
    """

    def __init__(self, ctx: mx.Context = None):
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
            self.finalize(state.document)

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
            xs = nd.array([state.x for state in tmp])
            ys = nd.array([state.y for state in tmp])
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
                correct += int(sum(mx.ndarray.argmax(outputs, axis=0) == y_batch).asscalar())
                for i, output in enumerate(outputs): states[begin + i].process(output)
                begin += len(outputs)

            tmp = [state for state in tmp if state.has_next]

        for state in states: state.init()
        return correct


class TokenTagger:
    def __init__(self, ctx, vsm_list):
        """
        TokenTagger provides a generic template to implement a component that predicts a tag for every token.
        :param ctx: "[cg]\\d*"; the context (e.g., CPU or GPU) to process.
        :type ctx: str
        :param vsm_list: a list of vector space models (must include at least one).
        :type vsm_list: list of elit.vsm.VectorSpaceModel
        """
        super().__init__(ctx)
        self.vsm_list = vsm_list
        self.padout = None

        # to be initialized
        self.label_map = None
        self.label_embedding = None
        self.feature_windows = None
        self.input_config = None
        self.output_config = None
        self.conv2d_config = None
        self.hidden_config = None

    def __str__(self):
        s = ('Configuration',
             '- label embedding: %r' % self.label_embedding,
             '- feature windows: %s' % str(self.feature_windows),
             '- input layer    : %s' % str(self.input_config).replace('namespace', ''),
             '- output layer   : %s' % str(self.output_config).replace('namespace', ''),
             '- conv2d layer   : %s' % str(self.conv2d_config).replace('namespace', ''),
             '- hidden layer   : %s' % str(self.hidden_config).replace('namespace', ''))
        return '\n'.join(s)

    @abc.abstractmethod
    def create_state(self, document):
        return

    @abc.abstractmethod
    def eval_metric(self):
        return

    # override
    def init(self,
             label_embedding=True,
             feature_windows=tuple(range(-3, 4)),
             num_class=50,
             input_dropout=0.0,
             conv2d_config=None,
             hidden_config=None,
             **kwargs):
        """
        :param label_embedding: True if label embeddings are used as features; otherwise, False.
        :type label_embedding: bool
        :param feature_windows: contextual windows for feature extraction.
        :type feature_windows: tuple of int
        :param num_class: the number of classes (part-of-speech tags).
        :type num_class: int
        :param input_dropout: a dropout rate to be applied to the input layer.
        :type input_dropout: float
        :param conv2d_config: configuration for n-gram 2D convolutions.
        :type conv2d_config: list of SimpleNamespace
        :param hidden_config: configuration for hidden layers
        :type hidden_config: list of SimpleNamespace
        :param kwargs: parameters for the initialization of gluon.Block.
        :type kwargs: dict
        :return: self
        :rtype: NLPComponent
        """
        self.label_map = LabelMap()
        self.label_embedding = label_embedding
        self.feature_windows = feature_windows

        # input dimension
        input_dim = len(X_ANY) + sum([vsm.dim for vsm in self.vsm_list])
        if self.label_embedding: input_dim += num_class

        # network configuration
        self.input_config = input_namespace(input_dim, maxlen=len(feature_windows), dropout=input_dropout)
        self.output_config = output_namespace(num_class)
        self.conv2d_config = conv2d_config
        self.hidden_config = hidden_config

        # model
        self.model = FFNNModel(self.input_config, self.output_config, self.conv2d_config, self.hidden_config, **kwargs)
        self.model.collect_params().initialize(mx.init.Xavier(rnd_type='gaussian', magnitude=2.24), ctx=self.ctx)
        self.padout = np.zeros(self.output_config.dim).astype('float32') if self.label_embedding else None
        print(self.__str__())
        return self

    # override
    def load(self, model_path, **kwargs):
        """
        :param model_path: the path to a pre-trained model to be loaded.
        :type model_path: str
        :param kwargs: parameters for the initialization of gluon.Block.
        :type kwargs: dict
        :return: self
        :rtype: NLPComponent
        """
        with open(pkl(model_path), 'rb') as f:
            self.label_map = pickle.load(f)
            self.label_embedding = pickle.load(f)
            self.feature_windows = pickle.load(f)
            self.input_config = pickle.load(f)
            self.output_config = pickle.load(f)
            self.conv2d_config = pickle.load(f)
            self.hidden_config = pickle.load(f)

        self.model = FFNNModel(self.input_config, self.output_config, self.conv2d_config, self.hidden_config, **kwargs)
        self.model.load_parameters(gln(model_path), ctx=self.ctx)
        self.padout = np.zeros(self.output_config.dim).astype('float32') if self.label_embedding else None
        print(self.__str__())
        return self

    # override
    def save(self, model_path, **kwargs):
        """
        :param model_path: the filepath where the model is saved.
        :type model_path: str
        """
        with open(pkl(model_path), 'wb') as f:
            pickle.dump(self.label_map, f)
            pickle.dump(self.label_embedding, f)
            pickle.dump(self.feature_windows, f)
            pickle.dump(self.input_config, f)
            pickle.dump(self.output_config, f)
            pickle.dump(self.conv2d_config, f)
            pickle.dump(self.hidden_config, f)

            # self.model.save_parameters(gln(model_path))
