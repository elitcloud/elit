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
from types import SimpleNamespace

import mxnet as mx
import numpy as np
from elitsdk.sdk import Component
from mxnet import nd, gluon, autograd

from elit.lexicon import LabelMap, X_ANY
from elit.util import pkl, gln, group_states

__author__ = 'Jinho D. Choi'


# ======================================== States ========================================

class NLPState(abc.ABC):
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
        return

    @property
    @abc.abstractmethod
    def y(self):
        """
        :return: the class ID of the gold-standard label for the current state (used for training only).
        :rtype: int
        """
        return


class ForwardState(NLPState):
    def __init__(self, document, label_map, zero_output, key, key_out=None):
        """
        ForwardState defines the one-pass, left-to-right decoding strategy.
        :param document: the input document.
        :type document: elit.util.Document
        :param label_map: the mapping between class labels and their unique IDs.
        :type label_map: elit.lexicon.LabelMap
        :param zero_output: a zero vector of size `num_class`; used to zero-pad label embeddings.
        :type zero_output: numpy.array
        :param key: the key in the sentence dictionary where the predicated labels are stored in.
        :type key: str
        :param key_out: the key in the sentence dictionary where the predicted outputs (output layers) are stored in.
        :type key_out: str
        """
        self.document = document
        self.output = None

        self.label_map = label_map
        self.zero_output = zero_output

        self.key = key
        self.key_out = key_out if key_out else key + '-out'

        self.sen_id = 0
        self.tok_id = 0
        self.reset()

    @abc.abstractmethod
    def eval(self, metric):
        pass

    @property
    @abc.abstractmethod
    def x(self):
        return

    def reset(self):
        if self.output is None:
            self.output = [[self.zero_output] * len(s) for s in self.document]
        else:
            for out in self.output:
                for o in out: o.fill(0.0)

        self.sen_id = 0
        self.tok_id = 0

    def process(self, output):
        # apply the output to the current state
        self.output[self.sen_id][self.tok_id] = output

        # move onto the next state
        self.tok_id += 1
        if self.tok_id == len(self.document.get_sentence(self.sen_id)):
            self.sen_id += 1
            self.tok_id = 0

    def has_next(self):
        return 0 <= self.sen_id < len(self.document)

    def finalize(self):
        """
        Finalizes the predicted labels and the prediction scores for all tokens in the input document.
        """
        for i, labels in enumerate(self.labels):
            d = self.document.get_sentence(i)
            d[self.key] = labels
            d[self.key_out] = self.output[i]

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
    def y(self):
        label = self.document.get_sentence(self.sen_id)[self.key][self.tok_id]
        return self.label_map.add(label)

    def _get_label_embeddings(self):
        """
        :return: (self.output, self.zero_output)
        """
        return self.output, self.zero_output


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

        self.conv2d = [SimpleNamespace(
            conv=mx.gluon.nn.Conv2D(channels=c.filters, kernel_size=(c.ngram, input_config.dim), strides=(1, input_config.dim), activation=c.activation),
            dropout=mx.gluon.nn.Dropout(c.dropout)) for c in conv2d_config] if conv2d_config else None

        self.hidden = [SimpleNamespace(
            dense=mx.gluon.nn.Dense(units=h.dim, activation=h.activation),
            dropout=mx.gluon.nn.Dropout(h.dropout)) for h in hidden_config] if hidden_config else None

        with self.name_scope():
            self.input_dropout = mx.gluon.nn.Dropout(input_config.dropout)
            self.output = mx.gluon.nn.Dense(output_config.dim)

            if self.conv2d:
                for i, c in enumerate(self.conv2d, 1):
                    setattr(self, 'conv_'+str(i), c.conv)
                    setattr(self, 'conv_dropout_' + str(i), c.dropout)

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


# ======================================== Component ========================================

class NLPComponent(Component):
    def __init__(self, ctx=None):
        """
        NLPComponent provides a generic template to implement an NLP component.
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

    @abc.abstractmethod
    def eval_metric(self):
        """
        :return: the evaluation metric for this component.
        :rtype: elit.util.EvalMetric
        """
        return

    def _decode(self, states, batch_size=2048):
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

    def _evaluate(self, states, batch_size=2048, reset=True):
        """
        :param states: input states.
        :type states: list of NLPState
        :param batch_size: the maximum size of each batch.
        :type batch_size: int
        :param reset: if True, reset all states to their initial stages.
        :type reset: bool
        :return: the evaluation metric.
        :rtype: elit.util.EvalMetric
        """
        self._decode(states, batch_size)
        metric = self.eval_metric()

        for state in states:
            state.eval(metric)
            if reset: state.reset()

        return metric

    def _train(self, states, trainer, loss_func, batch_size=64):
        """
        :param states: input states.
        :type states: list of NLPState
        :param trainer: the trainer including the optimizer.
        :type trainer: mxnet.gluon.Trainer
        :param loss_func: the loss function for the optimizer.
        :type loss_func: mxnet.gluon.loss.Loss
        :param batch_size: the maximum size of each batch.
        :type batch_size: int
        :return: the evaluation metric.
        :rtype: elit.util.EvalMetric
        """
        metric = self.eval_metric()
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
            state.eval(metric)
            state.reset()

        return metric

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
        TokenTagger provides a generic template to implement a component that predicts a tag for every token.
        :param ctx: "[cg]\d*"; the context (e.g., CPU or GPU) to process.
        :type ctx: str
        :param vsm_list: a list of vector space models (must include at least one).
        :type vsm_list: list of elit.lexicon.VectorSpaceModel
        """
        super().__init__(ctx)
        self.vsm_list = vsm_list
        self.zero_output = None

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
    def create_state(self, document): return

    @abc.abstractmethod
    def eval_metric(self): return

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
        self.input_config = SimpleNamespace(dim=input_dim, dropout=input_dropout)
        self.output_config = SimpleNamespace(dim=num_class)
        self.conv2d_config = conv2d_config
        self.hidden_config = hidden_config

        # model
        self.model = FFNNModel(self.input_config, self.output_config, self.conv2d_config, self.hidden_config, **kwargs)
        self.model.collect_params().initialize(mx.init.Xavier(rnd_type='gaussian', magnitude=2.24), ctx=self.ctx)
        self.zero_output = np.zeros(self.output_config.dim).astype('float32')
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
        self.zero_output = np.zeros(self.output_config.dim).astype('float32')
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

        self.model.save_parameters(gln(model_path))

    # override
    def decode(self, input_data, batch_size=2048, **kwargs):
        """
        :param input_data: a list of documents or sentences.
        :type input_data: list of elit.util.Document or list of elit.util.Sentence
        :param batch_size: the maximum size of each batch.
        :type batch_size: int
        """
        states = group_states(input_data, self.create_state)
        self._decode(states, batch_size)
        for state in states: state.finalize()

    # override
    def train(self, trn_data, dev_data, model_path,
              trn_batch=64, dev_batch=2048, epoch=50, optimizer='adagrad', learning_rate=0.01, weight_decay=0.0, **kwargs):
        log = ('Training',
               '- train batch   : %d' % trn_batch,
               '- develop batch : %d' % dev_batch,
               '- epoch         : %d' % epoch,
               '- optimizer     : %s' % optimizer,
               '- learning rate : %f' % learning_rate,
               '- weight decay  : %f' % weight_decay)
        print('\n'.join(log))

        trn_states = group_states(trn_data, self.create_state)
        dev_states = group_states(dev_data, self.create_state)

        # optimizer
        loss_func = gluon.loss.SoftmaxCrossEntropyLoss()
        trainer = gluon.Trainer(self.model.collect_params(), optimizer, {'learning_rate': learning_rate, 'wd': weight_decay})

        # train
        best_e, best_eval = -1, -1

        for e in range(1, epoch+1):
            st = time.time()
            trn_metric = self._train(trn_states, trainer, loss_func, trn_batch)
            mt = time.time()
            dev_metric = self._evaluate(dev_states, dev_batch)
            et = time.time()
            if best_eval < dev_metric.get():
                best_e, best_eval = e, dev_metric.get()
                self.save(model_path)

            print('%4d: trn-time: %d, dev-time: %d, trn-acc: %5.2f, dev-acc: %5.2f, num-class: %d, best-acc: %5.2f @%4d' %
                  (e, mt - st, et - mt, trn_metric.get(), dev_metric.get(), len(self.label_map), best_eval, best_e))

        return trn_states, dev_states
