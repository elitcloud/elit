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
from mxnet import nd, gluon, autograd
from mxnet.gluon.loss import _apply_weighting, Loss

from elit.util.structure import TOKENS

__author__ = 'Jinho D. Choi'


LABEL_MAP = 'label_map'
WORD_VSM = 'word_vsm'
ZERO_OUTPUT = 'zero_output'
CONTEXT_WINDOWS = 'context_windows'
NGRAM_CONV = 'ngram_conv'
DROPOUT = 'dropout'


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
    def set_labels(self, key):
        """
        Assign the string labels to each sentence[key] by inferring self.scores.
        :param key: the key to the sentence where the labels are stored.
        :type key: str
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


class ForwardState(NLPState, metaclass=abc.ABCMeta):
    def __init__(self, document, params, key_gold):
        """
        ForwardState implements a one-pass, left-to-right tagging algorithm.
        :param document: the input document.
        :type document: list of elit.util.structure.Sentence
        :param params: a dictionary containing [label_map, word_vsm, zero_output]
        :type params: dict
        :param key_gold: the key where the gold-standard tags are stored in (e.g., POS_GOLD)
        :type key_gold: str
        """
        super().__init__(document)
        self.KEY_GOLD = key_gold

        # parameters
        self.label_map = params[LABEL_MAP]        # elit.util.lexicon.LabelMap
        self.word_vsm = params[WORD_VSM]          # elit.util.lexicon.VectorSpaceModel
        self.zero_output = params[ZERO_OUTPUT]    # numpy.array

        # embeddings
        self.word_emb = [self.word_vsm.get_list(s[TOKENS]) for s in document]
        self.scores = [[self.zero_output] * len(s) for s in self.document]

        # state trackers
        self.sen_id = 0
        self.tok_id = 0

    def reset(self):
        self.scores = [[self.zero_output] * len(s) for s in self.document]
        self.sen_id = 0
        self.tok_id = 0

    def set_labels(self, key):
        for i, scores in enumerate(self.scores):
            sentence = self.document[i]
            sentence[key] = [self.label_map.get(np.argmax(s)) for s in scores]

    def process(self, output):
        # apply the output to the current state
        self.scores[self.sen_id][self.tok_id] = output

        # move onto the next state
        self.tok_id += 1
        if self.tok_id == self._sen_len:
            self.sen_id += 1
            self.tok_id = 0

    @property
    def has_next(self):
        return 0 <= self.sen_id < len(self.document)

    @property
    def y(self):
        return self.document[self.sen_id][self.KEY_GOLD][self.tok_id]

    @property
    def _sen_len(self):
        return len(self.document[self.sen_id])

    def _x_position(self, window):
        i = self.tok_id + window
        return self.x_fst if i == 0 else self.x_lst if i + 1 == self._sen_len else self.x_any

    def _extract_x(self, window, emb, zero):
        i = self.tok_id + window
        return emb[i] if 0 <= i < self._sen_len else zero


class BidirectionalState(ForwardState, metaclass=abc.ABCMeta):
    def __init__(self, document, params, key_gold):
        """
        BidirectionalState implements the two-passes, left-to-right and right-to-left tagging algorithm.
        :param document: the input document.
        :type document: list of elit.util.structure.Sentence
        :param params: a dictionary containing [label_map, word_vsm, zero_output]
        :type params: dict
        :param key_gold: the key where the gold-standard tags are stored in (e.g., POS_GOLD)
        :type key_gold: str
        """
        super().__init__(document, params, key_gold)
        self.dir = 1    # direction: 1 or -1

    def reset(self):
        super(BidirectionalState, self).reset()
        self.dir = 1

    def process(self, output):
        # apply the output to the current state
        self.scores[self.sen_id][self.tok_id] = output

        # move onto the next state
        self.tok_id += self.dir

        if self.tok_id == self._sen_len:
            if self.sen_id+1 < len(self.document):
                self.sen_id += self.dir
                self.tok_id = 0
            else:  # switch to right-to-left
                self.dir = -1
                self.tok_id += self.dir
        elif self.tok_id < 0:
            self.sen_id += self.dir
            self.tok_id = self._sen_len - 1


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


class BILOU:
    B = 'B'     # beginning
    I = 'I'     # inside
    L = 'L'     # last
    O = 'O'     # outside
    U = 'U'     # unit

    def collect(self, tags):
        """
        :param tags: a list of tags encoded by the BILOU format.
        :type tags: list of str
        :return: a dictionary where the keys represent the chunk spans and the values represent the tags
        """
        entities = {}
        begin = -1

        for i, tag in enumerate(tags):
            c = tag[0]

            if tag == self.B:
                begin = i
            elif tag == self.I:
                pass
            elif tag == self.L:
                if begin >= 0: entities[(begin, i+1)] = tags[2:]
                begin = -1
            elif tag == self.O:
                begin = -1
            elif tag == self.U:
                entities[(i, i+1)] = tags[2:]
                begin = -1

        return entities

    def quick_fix(self, tags):
        def fix(i, pt, ct, t1, t2):
            if pt == ct: tags[i][0] = t1
            else: tags[i-1][0] = t2

        def aux(i):
            p = tags[i-1][0]
            c = tags[i][0]
            pt = tags[i-1][1:]
            ct = tags[i][1:]

            if p == self.B:
                if   c == self.B: fix(i, pt, ct, self.I, self.U)  # BB -> BI or UB
                elif c == self.U: fix(i, pt, ct, self.L, self.U)  # BU -> BL or UU
                elif c == self.O: tags[i-1][0] = self.U           # BO -> UO
            elif p == self.I:
                if   c == self.B: fix(i, pt, ct, self.I, self.L)  # IB -> II or LB
                elif c == self.U: fix(i, pt, ct, self.I, self.L)  # IU -> II or LU
                elif c == self.O: tags[i-1][0] = self.L           # IO -> LO
            elif p == self.L:
                if   c == self.I: fix(i, pt, ct, self.I, self.B)  # LI -> II or LB
                elif c == self.L: fix(i, pt, ct, self.I, self.B)  # LL -> IL or LB
            elif p == self.O:
                if   c == self.I: tags[i][0] = self.B             # OI -> OB
                elif c == self.L: tags[i][0] = self.B             # OL -> OB
            elif p == self.U:
                if   c == self.I: fix(i, pt, ct, self.B, self.B)  # UI -> BI or UB
                elif c == self.L: fix(i, pt, ct, self.B, self.B)  # UL -> BL or UB

        for i in range(1, len(tags)): aux(i)
        p = tags[-1][0]
        if   p == self.B: tags[-1][0] = self.U
        elif p == self.I: tags[-1][0] = self.L


class MultiLabelSoftmaxCrossEntropyLoss(Loss):
    def __init__(self, axis=-1, sparse_label=True, from_logits=False, weight=None,
                 batch_axis=0, **kwargs):
        super(MultiLabelSoftmaxCrossEntropyLoss, self).__init__(weight, batch_axis, **kwargs)
        self._axis = axis
        self._sparse_label = sparse_label
        self._from_logits = from_logits

    def hybrid_forward(self, F, output, label, sample_weight=None):
        if not self._from_logits:
            output = F.log_softmax(output)
        if self._sparse_label:
            # loss = -F.pick(output, label, axis=self._axis, keepdims=True)
            l = -F.pick(output, label, axis=self._axis, keepdims=True)
            d = nd.array([0 if i.asscalar() < 0 else 1 for i in label]).reshape((-1, 1))
            loss = l * d
        else:
            loss = -F.sum(output*label, axis=self._axis, keepdims=True)
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        return F.mean(loss, axis=self._batch_axis, exclude=True)


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
    xs = nd.array([state.x for state in states])
    ys = nd.array([state.y for state in states])
    batch_size = min(batch_size, len(xs))
    return gluon.data.DataLoader(gluon.data.ArrayDataset(xs, ys), batch_size=batch_size, shuffle=shuffle), xs, ys


def process_online(model, states, batch_size, ctx, trainer=None, loss_func=None, metric=None, reshape_x=None, xs=None, ys=None, reset=True):
    st = time.time()
    tmp = list(states)

    while tmp:
        begin = 0
        if trainer: shuffle(tmp)
        batches, txs, tys = data_loader(tmp, batch_size)
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
        if xs is not None: xs.append(txs)
        if ys is not None: ys.extend(tys)

    for state in states:
        if metric: metric.update(state)
        if reset: state.reset()

    return time.time() - st


def train_batch(model, batches, ctx, trainer=None, loss_func=None, reshape_x=None):
    st = time.time()

    for x, y in batches:
        x = x.as_in_context(ctx)
        y = y.as_in_context(ctx)
        if reshape_x: x = reshape_x(x)

        with autograd.record():
            output = model(x)
            loss = loss_func(output, y)
            loss.backward()
        trainer.step(x.shape[0])

    return time.time() - st
