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

import time
from random import shuffle

import numpy as np
import mxnet as mx
from mxnet import nd, gluon, autograd

__author__ = 'Jinho D. Choi'


class NLPState(metaclass=abc.ABCMeta):
    x_fst = np.array([1, 0]).astype('float32')  # representing the first word
    x_lst = np.array([0, 1]).astype('float32')  # representing the last word
    x_any = np.array([0, 0]).astype('float32')  # representing any other word

    def __init__(self, document):
        """
        NLPState implements the decoding algorithm and processes through the input document.
        :param document: the input document.
        :type document: list of elit.util.structure.Sentence
        """
        self.document = document

    @abc.abstractmethod
    def reset(self):
        """
        Reset to the beginning state.
        """
        pass

    @abc.abstractmethod
    def process(self, output):
        """
        Apply the output to the current state, and move onto the next state.
        :param output: the prediction output of the current state.
        :type output: numpy.array
        """
        pass

    @property
    @abc.abstractmethod
    def has_next(self):
        """
        :return: True if there exists the next state that can be processed; otherwise, False.
        :rtype: bool
        """
        return

    @property
    @abc.abstractmethod
    def x(self):
        """
        :return: the feature vector given the current state.
        :rtype: numpy.array
        """
        return

    @property
    @abc.abstractmethod
    def y(self):
        """
        :return: the index of the gold-standard label.
        :rtype: int
        """
        return


class NLPEval(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def update(self, state):
        """
        Update the evaluation scores with the state.
        :param state: the NLP state to evaluate.
        :type state: NLPState
        """
        pass

    @abc.abstractmethod
    def get(self):
        """
        :return: the evaluated score.
        """
        return

    @abc.abstractmethod
    def reset(self):
        """
        Reset all scores to 0.
        """
        pass


def data_loader(states, batch_size, shuffle=False):
    """
    :param states: the list of NLP states.
    :type states: list of NLPState
    :param batch_size: the batch size.
    :type batch_size: int
    :param shuffle: if True, shuffle the instances.
    :type shuffle: bool
    :return: the data loader containing pairs of feature vectors and their labels.
    :rtype: gluon.data.DataLoader
    """
    x = nd.array([state.x for state in states])
    y = nd.array([state.y for state in states])
    batch_size = min(batch_size, len(x))
    return gluon.data.DataLoader(gluon.data.ArrayDataset(x, y), batch_size=batch_size, shuffle=shuffle)


def epoch(model, states, batch_size, ctx, reshape_x=None, trainer=None, loss_func=None, metric=None, reset=True):
    st = time.time()
    tmp = list(states)

    while tmp:
        begin = 0
        if trainer: shuffle(tmp)
        batches = data_loader(tmp, batch_size)
        for x, y in batches:
            x = x.as_in_context(ctx)
            y = y.as_in_context(ctx)
            if reshape_x: x = reshape_x(x)

            if trainer:
                with autograd.record():
                    output = model(x)
                    loss = loss_func(output, y)
                    loss.backward()
                trainer.step(x.shape[0])
            else:
                output = model(x)

            for i in range(len(y)): tmp[begin+i].process(output[i].asnumpy())
            begin += len(output)

        tmp = [state for state in tmp if state.has_next]

    for state in states:
        if metric: metric.update(state)
        if reset: state.reset()

    return time.time() - st
