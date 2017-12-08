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

import numpy as np
from mxnet import nd, gluon, autograd

from elit.nlp.structure import OUT

__author__ = 'Jinho D. Choi'


# ======================================== State ========================================

class NLPState(metaclass=abc.ABCMeta):
    def __init__(self, document):
        """
        NLPState defines a decoding strategy to process the input document.
        :param document: the input document.
        :type document: elit.nlp.structure.Document
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
        Applies the output to the current state, and moves onto the next state.
        :param output: the prediction output of the current state.
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

    @property
    @abc.abstractmethod
    def labels(self):
        """
        :return: the predicted labels for the input document inferred from the self.output.
        """
        return

    @abc.abstractmethod
    def supply(self):
        """
        Supplies the predicted labels as well as other information (e.g., self.output) to the input document.
        """
        pass

    @abc.abstractmethod
    def eval(self, metric):
        """
        Updates the evaluation metric using the gold-standard labels in document and predicted labels from self.labels.
        :param metric: the evaluation metric.
        :type metric: elit.nlp.metric.Metric
        """
        pass

    @property
    @abc.abstractmethod
    def x(self):
        """
        :return: the feature vector (or matrix) for the current state.
        :rtype: numpy.array
        """
        return

    @property
    @abc.abstractmethod
    def y(self):
        """
        :return: the ID of the gold-standard label for the current state.
        :rtype: int
        """
        return


class ForwardState(NLPState, metaclass=abc.ABCMeta):
    def __init__(self, document, label_map, zero_output, key):
        """
        ForwardState defines the one-pass, left-to-right decoding strategy.
        :param document: the input document.
        :type document: elit.nlp.structure.Document
        :param label_map: the mapping between class labels and their unique IDs.
        :type label_map: elit.nlp.lexicon.LabelMap
        :param zero_output: a vector of size `num_class` where all values are 0; used to zero pad label embeddings.
        :type zero_output: numpy.array
        :param key: the key to the sentence where the predicated labels are stored in.
        :type key: str
        """
        super().__init__(document)
        self.label_map = label_map
        self.zero_output = zero_output

        self.key = key
        self.key_out = key+OUT

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

    def supply(self):
        """
        Supplies the predicted labels and the prediction scores for all tokens in the input document.
        """
        for i, labels in enumerate(self.labels):
            d = self.document[i]
            d[self.key] = labels
            d[self.key_out] = self.output[i]

    @property
    def y(self):
        label = self.document[self.sen_id][self.key][self.tok_id]
        return self.label_map.add(label)


# ======================================== Model ========================================

class CNN2DModel(gluon.Block):
    def __init__(self, input_col, num_class, ngram_conv, dropout, **kwargs):
        """
        :param kwargs: parameters to initialize gluon.Block.
        :type kwargs: dict
        """
        super().__init__(**kwargs)
        self.ngram_conv = []

        with self.name_scope():
            for i, n in enumerate(ngram_conv):
                conv = gluon.nn.Conv2D(channels=n.filters, kernel_size=(n.kernel_row, input_col), strides=(1, input_col), activation=n.activation)
                name = 'conv_' + str(i)
                self.ngram_conv.append(conv)
                setattr(self, name, conv)

            self.dropout = gluon.nn.Dropout(dropout)
            self.out = gluon.nn.Dense(num_class)

    def forward(self, x):
        # prepare for 2D convolution
        x = x.reshape((0, 1, x.shape[1], x.shape[2]))

        # n-gram convolutions
        t = [conv(x).reshape((0, -1)) for conv in self.ngram_conv]
        x = nd.concat(*t, dim=1)
        x = self.dropout(x)

        # output layer
        x = self.out(x)
        return x


# ======================================== Component ========================================

def pkl(filepath): return filepath+'.pkl'
def gln(filepath): return filepath+'.gln'


class NLPComponent(metaclass=abc.ABCMeta):
    def __init__(self, ctx, model):
        """
        NLPComponent gives a template to implement a machine learning-based component.
        :param ctx: the context (e.g., CPU or GPU) to process this component.
        :type ctx: mxnet.context.Context
        :param model: a machine learning model.
        :type model: mxnet.gluon.Block
        """
        self.ctx = ctx
        self.model = model

    @abc.abstractmethod
    def save(self, filepath):
        """
        Saves this component to the filepath.
        :param filepath: the path to the file to be saved.
        :type filepath: str
        """
        pass

    @abc.abstractmethod
    def create_state(self, document):
        """
        :param document: the input document.
        :type document: elit.nlp.structure.Document
        :return: the state containing the input document for this component.
        :rtype: NLPState
        """
        return

    def decode(self, states, batch_size, reset=False):
        """
        Makes predictions for all states and saves them to the corresponding documents.
        :param states: the input states.
        :type states: list of NLPState
        :param batch_size: the maximum size of each batch.
        :type batch_size: int
        :param reset: if True, reset all states to their initial stages.
        :type reset: bool
        """
        self._decode(states, batch_size)

        for state in states:
            state.supply()
            if reset: state.reset()

    def evaluate(self, states, batch_size, metric, reset=True):
        """
        Makes predictions for all states and evaluates the results using the metric.
        :param states: the input states.
        :type states: list of elit.nlp.component.NLPState
        :param metric: the evaluation metric.
        :type metric: elit.nlp.metric.Metric
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

    def train(self, states, batch_size, trainer, loss_func, metric=None, reset=True):
        """
        :param states: the input states.
        :type states: list of elit.nlp.component.NLPState
        :param batch_size: the maximum size of each batch.
        :type batch_size: int
        :param trainer: the trainer including the optimizer.
        :type trainer: mxnet.gluon.Trainer
        :param loss_func: the loss function for the optimizer.
        :type loss_func: mxnet.gluon.loss.Loss
        :param metric: the evaluation metric
        :type metric: elit.nlp.metric.Metric
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
            if reset: state.reset()

        return metric.get() if metric else 0

    @staticmethod
    def _process(states, output, begin):
        size = len(output)

        for i in range(size):
            states[begin+i].process(output[i].asnumpy())

        return size

    def _decode(self, states, batch_size):
        tmp = list(states)

        while tmp:
            begin = 0
            xs = nd.array([state.x for state in tmp])
            batches = gluon.data.DataLoader(xs, batch_size=batch_size)

            for x in batches:
                x = x.as_in_context(self.ctx)
                output = self.model(x)
                begin += self._process(tmp, output, begin)

            tmp = [state for state in tmp if state.has_next()]
