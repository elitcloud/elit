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
import pickle
from types import SimpleNamespace

import mxnet as mx
import numpy as np
import time
from elitsdk import Component
from mxnet import nd, gluon, autograd

from elit.lexicon import LabelMap, X_ANY
from elit.util import pkl, gln

__author__ = 'Jinho D. Choi'


# ======================================== Models ========================================

class FFNNModel(gluon.Block):
    def __init__(self, input_config, output_config, conv2d_config=None, hidden_config=None, **kwargs):
        """
        Feed-Forward Neural Network including either convolution layer or hidden layers or both.
        :param input_config: (dim, dropout); configuration for the input layer.
        :type input_config: SimpleNamespace(int, float)
        :param output_config: (dim); configuration for the output layer.
        :type output_config: SimpleNamespace(int)
        :param conv2d_config: (ngram, filters, activation, dropout); configuration for the 2D convolution layer.
        :type conv2d_config: list of SimpleNamespace(int, int, str, float)
        :param hidden_config: (dim, activation, dropout); configuration for the hidden layers.
        :type hidden_config: list of SimpleNamespace(int, str, float)
        :param kwargs: parameters for the initialization of mxnet.gluon.Block.
        :type kwargs: dict
        """
        super().__init__(**kwargs)

        if conv2d_config:
            self.conv2d = [SimpleNamespace(
                conv=mx.gluon.nn.Conv2D(channels=c.filters, kernel_size=(c.ngram, input_config.dim), strides=(1, input_config.dim), activation=c.activation),
                dropout=mx.gluon.nn.Dropout(c.dropout)) for c in conv2d_config] if conv2d_config else None
        else:
            self.conv2d = None

        if hidden_config:
            self.hidden = [SimpleNamespace(
                dense=mx.gluon.nn.Dense(units=h.dim, activation=h.activation),
                dropout=mx.gluon.nn.Dropout(h.dropout)) for h in hidden_config] if hidden_config else None
        else:
            self.hidden = None

        with self.name_scope():
            self.input_dropout = mx.gluon.nn.Dropout(input_config.dropout)
            self.output = mx.gluon.nn.Dense(output_config.dim)

            if self.conv2d:
                for i, c in enumerate(self.conv2d, 1):
                    setattr(self, 'conv_'+str(i), c.conv)
                    setattr(self, 'conv_dropout_'+str(i), c.dropout)

            if self.hidden:
                for i, h in enumerate(self.hidden, 1):
                    setattr(self, 'hidden_' + str(i), h.dense)
                    setattr(self, 'hidden_dropout_' + str(i), h.dropout)

    def forward(self, x):
        # input layer
        x = self.input_dropout(x)

        # convolution layer
        if self.conv2d:
            x = x.reshape((0, 1, x.shape[1], x.shape[2]))
            t = [c.dropout(c.conv(x).reshape((0, -1))) for c in self.conv2d]
            x = nd.concat(*t, dim=1)

        # hidden layers
        if self.hidden:
            for h in self.hidden:
                x = h.dense(x)
                x = h.dropout(x)

        # output layer
        x = self.output(x)
        return x


# TODO: LSTMModel needs to be reimplemented
class LSTMModel(gluon.Block):
    def __init__(self, input_col, num_class, n_hidden, dropout, **kwargs):
        """
        :param kwargs: parameters to initialize gluon.Block.
        :type kwargs: dict
        """
        bi = True
        super().__init__(**kwargs)
        with self.name_scope():
            self.model = gluon.rnn.LSTM(n_hidden, input_size=input_col, bidirectional=bi)
            self.dropout = gluon.nn.Dropout(dropout)
            self.out = gluon.nn.Dense(num_class)
        print('Init Model: LSTM, bidirectional = %r' % bi)

    def forward(self, x):
        x = self.model(x)
        x = self.dropout(x)
        # output layer
        x = self.out(x)
        return x


# ======================================== States ========================================

class NLPState(abc.ABC):
    def __init__(self, document):
        """
        NLPState defines a decoding strategy to process the input document.
        :param document: the input document.
        :type document: elit.util.Document
        """
        self.document = document
        self.output = None

    @abc.abstractmethod
    def reset(self):
        """
        Resets to the initial state.
        """
        pass

    @abc.abstractmethod
    def process(self, output):
        """
        Applies the prediction output to the current state, and moves onto the next state.
        :param output: the prediction output of the current state, that is the output_config layer of the network.
        :type output: numpy.array
        """
        pass

    @abc.abstractmethod
    def has_next(self):
        """
        :return: True if there exists the next state to be processed; otherwise, False.
        :rtype: bool
        """
        return

    @abc.abstractmethod
    def finalize(self):
        """
        Finalizes the predictions for the input document.
        """
        pass

    @abc.abstractmethod
    def eval(self, metric):
        """
        Updates the evaluation metric by comparing the gold-standard labels and the predicted labels from `self.labels`.
        :param metric: the evaluation metric.
        :type metric: elit.util.EvalMetric
        """
        pass

    @property
    @abc.abstractmethod
    def labels(self):
        """
        :return: the predicted labels for the input document inferred by `self.output`.
        """
        return

    @property
    @abc.abstractmethod
    def x(self):
        """
        :return: the feature vector (or matrix) extracted from the current state.
        :rtype: numpy.array
        """
        pass

    @property
    @abc.abstractmethod
    def y(self):
        """
        :return: the class ID of the gold-standard label for the current state (used for training only).
        :rtype: int
        """
        pass


class ForwardState(NLPState):
    def __init__(self, document, label_map, zero_output, key, key_out=None):
        """
        ForwardState defines the one-pass, left-to-right decoding strategy.
        :param document: the input document.
        :type document: elit.util.Document
        :param label_map: the mapping between class labels and their unique IDs.
        :type label_map: elit.lexicon.LabelMap
        :param zero_output: a vector of size `num_class` where all values are 0; used to zero-pad label embeddings.
        :type zero_output: numpy.array
        :param key: the key in the sentence dictionary where the predicated labels are stored in.
        :type key: str
        :param key_out: the key in the sentence dictionary where the predicted outputs (output layers) are stored in.
        :type key_out: str
        """
        super().__init__(document)
        self.label_map = label_map
        self.zero_output = zero_output

        self.key = key
        self.key_out = key_out if key_out else key + '-out'

        self.sen_id = 0
        self.tok_id = 0
        self.reset()

    def reset(self):
        self.output = [[self.zero_output] * len(s) for s in self.document]
        self.sen_id = 0
        self.tok_id = 0

    def process(self, output):
        # apply the output to the current state
        self.output[self.sen_id][self.tok_id] = output

        # move onto the next state
        self.tok_id += 1
        if self.tok_id == len(self.document[self.sen_id]):
            self.sen_id += 1
            self.tok_id = 0

    def has_next(self):
        return 0 <= self.sen_id < len(self.document)

    def finalize(self):
        """
        Finalizes the predicted labels and the prediction scores for all tokens in the input document.
        """
        for i, labels in enumerate(self.labels):
            d = self.document[i]
            d[self.key] = labels
            d[self.key_out] = self.output[i]

    @abc.abstractmethod
    def eval(self, metric):
        pass

    @property
    def labels(self):
        """
        :rtype: list of (list of str)
        """
        def aux(scores):
            if size < len(scores): scores = scores[:size]
            return self.label_map.get(np.argmax(scores))

        size = len(self.label_map)
        return [[aux(o) for o in output] for output in self.output]

    @property
    @abc.abstractmethod
    def x(self):
        pass

    @property
    def y(self):
        label = self.document[self.sen_id][self.key][self.tok_id]
        return self.label_map.add(label)


# ======================================== Component ========================================

class NLPComponent(Component):
    def __init__(self, ctx=None):
        """
        This class provides a generic template to implement an NLP component.
        :param ctx: "[cg]\d*"; the context (e.g., CPU or GPU) to process.
        :type ctx: str
        """
        if ctx:
            d = int(ctx[1:]) if len(ctx) > 1 else 0
            self.ctx = mx.gpu(d) if ctx[0] == 'g' else mx.cpu(d)
        else:
            self.ctx = None

        self.model = None

    @abc.abstractmethod
    def create_state(self, document):
        """
        :param document: the input document.
        :type document: elit.util.Document
        :return: the state containing the input document for this component.
        :rtype: NLPState
        """
        return

    # override
    def decode(self, input_data, reader, batch_size=2048):
        """
        Makes predictions for all states and saves them to the corresponding documents.
        """
        states = reader(input_data, self.create_state)
        self._decode(states, batch_size)
        for state in states: state.finalize()

        return {state: state.labels for state in states}

    def _evaluate(self, states, batch_size, metric, reset=True):
        """
        Makes predictions for all states and evaluates the results using the metric.
        :param states: input states.
        :type states: list of NLPState
        :param metric: the evaluation metric.
        :type metric: elit.util.EvalMetric
        :param batch_size: the maximum size of each batch.
        :type batch_size: int
        :param reset: if True, reset all states to their initial stages.
        :type reset: bool
        :return: the overall evaluation score.
        :rtype: float or tuple of float
        """
        self._decode(states, batch_size)

        for state in states:
            state.eval(metric)
            if reset: state.reset()

        return metric.get()

    def _train(self, states, batch_size, trainer, loss_func, metric=None):
        """
        :param states: input states.
        :type states: list of NLPState
        :param batch_size: the maximum size of each batch.
        :type batch_size: int
        :param trainer: the trainer including the optimizer.
        :type trainer: mxnet.gluon.Trainer
        :param loss_func: the loss function for the optimizer.
        :type loss_func: mxnet.gluon.loss.Loss
        :param metric: the evaluation metric
        :type metric: elit.util.EvalMetric
        :param reset: if True, reset all states to their initial stages.
        :type reset: bool
        """
        tmp = list(states)

        while tmp:
            begin = 0
            random.shuffle(tmp)
            xs = nd.array([state.x for state in tmp])
            ys = nd.array([state.y for state in tmp])
            batches = gluon.data.DataLoader(gluon.data.ArrayDataset(xs, ys), batch_size=batch_size)

            for x, y in batches:
                x = x.as_in_context(self.ctx)
                y = y.as_in_context(self.ctx)

                with autograd.record():
                    output = self.model(x)
                    loss = loss_func(output, y)
                    loss.backward()

                trainer.step(x.shape[0])
                begin += self._process(tmp, output, begin)

            tmp = [state for state in tmp if state.has_next()]

        for state in states:
            if metric: state.eval(metric)
            state.reset()

        return metric.get() if metric else 0

    def _decode(self, states, batch_size):
        """
        :param states: input states.
        :type states: list of NLPState
        :param batch_size: the maximum size of each batch.
        :type batch_size: int
        """
        tmp = list(states)

        while tmp:
            begin = 0
            xs = nd.array([state.x for state in tmp])
            batches = gluon.data.DataLoader(xs, batch_size=batch_size)

            for x in batches:
                x = x.as_in_context(self.ctx)
                outputs = self.model(x)
                begin += self._process(tmp, outputs, begin)

            tmp = [state for state in tmp if state.has_next()]

    @staticmethod
    def _process(states, outputs, begin):
        """
        :param states: input states.
        :type states: list of NLPState
        :param outputs: the prediction outputs (output layers).
        :param begin: list of numpy.array
        :return: the number of processed states.
        :rtype: int
        """
        size = len(outputs)

        for i in range(size):
            states[begin + i].process(outputs[i].asnumpy())

        return size


class TokenTagger(NLPComponent):
    def __init__(self, ctx, vsm_list):
        """
        This class provides a generic template to implement a token-based tagger using FFNNModel.
        :param ctx: "[cg]\d*"; the context (e.g., CPU or GPU) to process.
        :type ctx: str
        :param vsm_list: a list of vector space models (must include at least one).
        :type vsm_list: list of elit.lexicon.VectorSpaceModel
        """
        super().__init__(ctx)
        self.vsm_list = vsm_list

        # to be initialized
        self.label_map = None
        self.label_emb = None
        self.feature_windows = None
        self.input_config = None
        self.output_config = None
        self.conv2d_config = None
        self.hidden_config = None

    def __str__(self):
        s = ('Configuration',
             '- label embedding  : %r' % self.label_emb,
             '- feature windows  : %s' % str(self.feature_windows),
             '- input configure  : %s' % str(self.input_config).replace('namespace', ''),
             '- output configure : %s' % str(self.output_config).replace('namespace', ''),
             '- conv2d configure : %s' % str(self.conv2d_config).replace('namespace', ''),
             '- hidden configure : %s' % str(self.hidden_config).replace('namespace', ''))
        return '\n'.join(s)

    # override
    def init(self,
             label_emb=True,
             feature_windows=tuple(range(-3, 4)),
             num_class=50,
             input_dropout=0.0,
             conv2d_config=(SimpleNamespace(ngram=i, filters=128, activation='relu', dropout=0.2) for i in range(1, 5)),
             hidden_config=None,
             **kwargs):
        """
        :param label_emb: True if label embeddings are used as features; otherwise, False.
        :type label_emb: bool
        :param feature_windows: contextual windows for feature extraction.
        :type feature_windows: tuple of int
        :param num_class: number of classes (part-of-speech tags).
        :type num_class: int
        :param input_dropout: dropout rate to be applied to the input layer.
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
        self.label_emb = label_emb
        self.feature_windows = feature_windows

        # input dimension
        input_dim = len(X_ANY) + sum([vsm.dim for vsm in self.vsm_list])
        if self.label_emb: input_dim += num_class

        # network configuration
        self.input_config = SimpleNamespace(dim=input_dim, dropout=input_dropout)
        self.output_config = SimpleNamespace(dim=num_class)
        self.conv2d_config = conv2d_config
        self.hidden_config = hidden_config

        # model
        self.model = FFNNModel(self.input_config, self.output_config, self.conv2d_config, self.hidden_config, **kwargs)
        return self

    # override
    def load(self, model_path, **kwargs):
        """
        :param model_path: path to a pre-trained model.
        :type model_path: str
        :param kwargs: parameters for the initialization of gluon.Block.
        :type kwargs: dict
        :return: self
        :rtype: NLPComponent
        """
        with open(pkl(model_path), 'rb') as f:
            self.label_map = pickle.load(f)
            self.label_emb = pickle.load(f)
            self.feature_windows = pickle.load(f)
            self.input_config = pickle.load(f)
            self.output_config = pickle.load(f)
            self.conv2d_config = pickle.load(f)
            self.hidden_config = pickle.load(f)

        self.model = FFNNModel(self.input_config, self.output_config, self.conv2d_config, self.hidden_config, **kwargs)
        self.model.load_parameters(gln(model_path), ctx=self.ctx)
        return self

    # override
    def save(self, model_path, **kwargs):
        with open(pkl(model_path), 'wb') as f:
            pickle.dump(self.label_map, f)
            pickle.dump(self.label_emb, f)
            pickle.dump(self.feature_windows, f)
            pickle.dump(self.input_config, f)
            pickle.dump(self.output_config, f)
            pickle.dump(self.conv2d_config, f)
            pickle.dump(self.hidden_config, f)

        self.model.save_parameters(gln(model_path))

    # override
    def train(self, trn_data, dev_data, model_path, reader,
              epoch=50, trn_batch=64, dev_batch=2048, optimizer='adagrad', learning_rate=0.01, weight_decay=0.0):
        log = ('Training',
               '- epoch         : %f' % epoch,
               '- train batch   : %d' % trn_batch,
               '- develop batch : %d' % dev_batch,
               '- optimizer     : %s' % optimizer,
               '- learning rate : %f' % learning_rate,
               '- weight decay  : %f' % weight_decay)
        print('\n'.join(log))

        trn_states = reader(trn_data, self.create_state)
        dev_states = reader(dev_data, self.create_state)

        # optimizer
        loss_func = gluon.loss.SoftmaxCrossEntropyLoss()
        trainer = gluon.Trainer(self.model.collect_params(), optimizer, {'learning_rate': learning_rate, 'wd': weight_decay})

        # train
        best_e, best_eval = -1, -1
        trn_metric = self.eval_metric()
        dev_metric = self.eval_metric()

        for e in range(epoch):
            trn_metric.reset()
            dev_metric.reset()

            st = time.time()
            trn_eval = self._train(trn_states, trn_batch, trainer, loss_func, trn_metric)
            mt = time.time()
            dev_eval = self._evaluate(dev_states, dev_batch, dev_metric)
            et = time.time()
            if best_eval < dev_eval:
                best_e, best_eval = e, dev_eval
                self.save(model_path=model_path)

            print('%4d: trn-time: %d, dev-time: %d, trn-acc: %5.2f (%d), dev-acc: %5.2f (%d), num-class: %d, best-acc: %5.2f @%4d' %
                  (e, mt - st, et - mt, trn_eval, trn_metric.total, dev_eval, dev_metric.total, len(self.label_map), best_eval, best_e))

    @abc.abstractmethod
    def eval_metric(self):
        """
        :return: the evaluation metric for this component.
        :rtype: elit.util.Accuracy
        """
        return
