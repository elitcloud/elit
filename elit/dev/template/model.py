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
import logging
import time
from abc import ABCMeta, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait
from itertools import islice
from random import shuffle

import mxnet as mx
import numpy as np

__author__ = 'Jinho D. Choi'


class NLPModel(metaclass=ABCMeta):
    def __init__(self, state, batch_size):
        # y
        """

        :param state:
        :type state: Callable[[NLPGraph, NLPLexiconMapper, bool], NLPState]
        :param batch_size:
        :type batch_size: int
        """
        self.index_map = {}
        # Dict[str, int]
        self.labels = []
        # List[str]

        # init
        self.mxmod = None
        self.state = state
        self.batch_size = batch_size
        self.pool_feature_vector = None

    # ============================== Label ==============================

    def get_label_index(self, label):
        """

        :param label:
        :type label: str
        :return: the index of the y.
        :rtype: int
        """
        return self.index_map.get(label, -1)

    def get_label(self, index):
        """

        :param index: the index of the y to be returned.
        :type index: Union[int, np.int]
        :return: the index'th y.
        :rtype: str
        """
        return self.labels[index]

    def add_label(self, label):
        """

        :param label:
        :type label: str
        :return: the index of the y. Add a y to this map if not exist already.
        :rtype int
        """
        idx = self.get_label_index(label)
        if idx < 0:
            idx = len(self.labels)
            self.index_map[label] = idx
            self.labels.append(label)
        return idx

    @property
    def num_label(self):

        """

        :return:
        :rtype: int
        """
        return len(self.labels)

    # ============================== Feature ==============================

    @abstractmethod
    def x(self, state):
        """

        :param state:
        :type state: NLPState
        :return: the feature vector for the current state.
        :rtype: np.array
        """

    def feature_vectors(self, states):
        """

        :param states:
        :type states: List[NLPState]
        :return:
        :rtype: np.array
        """
        xs_0 = [[self.x(state)[0]] for state in states]
        xs_1 = [[self.x(state)[1]] for state in states]
        out_0 = np.stack(xs_0, axis=0)
        out_1 = np.stack(xs_1, axis=0)
        out_0 = np.squeeze(out_0, axis=1)
        out_1 = np.squeeze(out_1, axis=1)
        return [out_0, out_1]

    def train_instances(self, states, num_threads=1):
        """

        :param states:
        :type states: List[NLPState]
        :param num_threads:
        :type num_threads: int
        :return:
        :rtype: Tuple[np.array, np.array]
        """

        def instances(thread_id=0, batch_size=len(states)):
            bidx = thread_id * batch_size
            eidx = min((thread_id + 1) * batch_size, len(states))
            return zip(*[(self.x(state), state.gold) for state in islice(states, bidx, eidx)])

        def xys(future):
            """

            :param future:
            :type future: Future
            :return:
            """
            xs, ys = future.result()
            return xs, np.array([self.add_label(y) for y in ys])

        if num_threads == 1:
            xs, ys = instances()
            xs_1, xs_2 = zip(*xs)
            x1 = np.stack(xs_1, axis=0)
            x2 = np.stack(xs_2, axis=0)
            data = [x1, x2]
            label = np.array([self.add_label(y) for y in ys])
            return data, label
        pool = ThreadPoolExecutor(num_threads)
        size = np.math.ceil(len(states) / num_threads)
        futures = [pool.submit(instances, i, size) for i in range(num_threads)]
        while wait(futures)[1]:
            pass
        xxs, yys = zip(*[xys(future) for future in futures])
        return np.vstack(xxs), np.hstack(yys)

    # ============================== Module ==============================

    def bind(self, data, label=None, batch_size=32, for_training=True, force_rebind=True):
        """

        :param data:
        :type data: np.array
        :param label:
        :type label: np.array
        :param batch_size:
        :type batch_size: int
        :param for_training:
        :type for_training: bool
        :param force_rebind:
        :type force_rebind: bool
        :return:
        :rtype: mx.io.DataIter
        """
        batches = self.data_iter(data, label, batch_size)
        # mx.io.NDArrayIter
        label_shapes = None if label is None else batches.provide_label
        self.mxmod.bind(data_shapes=batches.provide_data,
                        label_shapes=label_shapes,
                        for_training=for_training,
                        force_rebind=force_rebind)
        return batches

    def fit(self, batches, num_epoch=1):
        """

        :param batches:
        :type batches: mx.io.DataIter
        :param num_epoch:
        :type num_epoch: int
        :return:
        :rtype: np.array
        """
        for _ in range(num_epoch):
            for batch in batches:
                self.mxmod.forward(batch)

                # extract pooled layer feature vector.
                # [0] index output is softmax layer output
                temp_symbol_1 = self.mxmod.get_outputs()[1]
                fea = temp_symbol_1.asnumpy()

                # create combined feature vector
                if self.pool_feature_vector is None:
                    self.pool_feature_vector = fea
                else:
                    self.pool_feature_vector = np.concatenate((self.pool_feature_vector, fea),
                                                              axis=0)

                self.mxmod.backward()
                self.mxmod.update()

            # sync aux params across devices
            arg_params, aux_params = self.mxmod.get_params()
            self.mxmod.set_params(arg_params, aux_params)

            # end of 1 epoch, reset the x-iter for another epoch
            batches.reset()

        return self.predict(batches)

    def predict(self, batches):
        """

        :param batches:
        :type batches: mx.io.DataIter
        :return:
        :rtype: np.array
        """
        return self.mxmod.predict(batches)[0].asnumpy()

    def train(self, trn_graphs, dev_graphs, lexicon, num_steps=2000, bagging_ratio=0.63,
              initializer=mx.initializer.Normal(0.01), arg_params=None, aux_params=None,
              allow_missing=False, force_init=False, kvstore='local', optimizer='Adam',
              optimizer_params=(('learning_rate', 0.01),)):

        """

        :param trn_graphs:
        :type trn_graphs: List[NLPGraph]
        :param dev_graphs:
        :type dev_graphs: List[NLPGraph]
        :param lexicon:
        :type lexicon: NLPLexiconMapper
        :param num_steps:
        :type num_steps: int
        :param bagging_ratio:
        :type bagging_ratio: int
        :param initializer:
        :type initializer:  mx.initializer.Initializer
        :param arg_params:
        :type arg_params:
        :param aux_params:
        :type aux_params:
        :param allow_missing:
        :type allow_missing: bool
        :param force_init:
        :type force_init: bool
        :param kvstore:
        :type kvstore: Union[str, mx.kvstore.KVStore]
        :param optimizer:
        :type optimizer: Union[str, mx.optimizer.Optimizer]
        :param optimizer_params:
        :type optimizer_params:
        """
        trn_states = [self.state(graph, lexicon, save_gold=True) for graph in trn_graphs]
        # List[NLPState]
        dev_states = [self.state(graph, lexicon, save_gold=True) for graph in dev_graphs]
        # List[NLPState]

        bag_size = int(len(trn_states) * bagging_ratio)

        best_eval = 0
        for step in range(1, num_steps + 1):
            st = time.time()
            shuffle(trn_states)
            trn_states.sort(key=lambda x: x.reset_count)
            xs, ys = self.train_instances(trn_states[:bag_size])
            batches = self.bind(xs, ys, batch_size=self.batch_size)

            if step == 1:
                self.mxmod.init_params(
                    initializer=initializer, arg_params=arg_params, aux_params=aux_params,
                    allow_missing=allow_missing, force_init=force_init)
                self.mxmod.init_optimizer(
                    kvstore=kvstore, optimizer=optimizer, optimizer_params=optimizer_params)

            predictions = self.fit(batches)
            correct = 0

            for state, y, yhats in zip(trn_states, ys, predictions):
                yh = np.argmax(yhats if len(yhats) == self.num_label else yhats[:self.num_label])
                state.process(self.get_label(yh), yhats)
                if y == yh:
                    correct += 1
                if state.terminate:
                    state.reset()

            trn_acc = correct / len(ys)
            dev_eval = self.evaluate(dev_states, batch_size=self.batch_size)
            tt = time.time() - st
            logging.info('%6d: trn-acc = %6.4f, dev-eval = %6.4f, time = %d',
                         step, trn_acc, dev_eval, tt)
            best_eval = max(dev_eval, best_eval)

        logging.info('best: %6.4f', best_eval)

    def evaluate(self, states, batch_size=32):
        """

        :param states:
        :type states:  List[NLPState]
        :param batch_size:
        :type batch_size: int
        :return:
        """
        for state in states:
            state.reset()
        backup = states

        while states:
            xs = self.feature_vectors(states)
            batches = self.bind(xs, batch_size=batch_size, for_training=False)
            # mx.io.NDArrayIter
            predictions = self.predict(batches)

            for state, yhats in zip(states, predictions):
                yh = np.argmax(yhats if len(yhats) == self.num_label else yhats[:self.num_label])
                state.process(self.get_label(yh), yhats)

            states = [state for state in states if not state.terminate]

        stats = np.array([0, 0])
        acc = 0

        for state in backup:
            acc = state.score()

        return acc

    # ============================== Helper ==============================

    @classmethod
    def data_iter(cls, data, label=None, batch_size=32):
        """

        :param data:
        :type data: np.array
        :param label:
        :type label: np.array
        :param batch_size:
        :type batch_size: int
        :return:
        :rtype: mx.io.DataIter
        """
        batch_size = len(data[0]) if len(data[0]) < batch_size else batch_size
        return mx.io.NDArrayIter(data={'data_f2v': data[0], 'data_a2v': data[1]}, label=label,
                                 batch_size=batch_size, shuffle=False)
