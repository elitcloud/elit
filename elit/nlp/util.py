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
import bisect
import glob
import logging
import time
from random import shuffle

from mxnet import gluon, autograd

from elit.nlp.structure import TOKEN, DEP, Sentence

__author__ = 'Jinho D. Choi'


def read_tsv(filepath, create_state, cols):
    """
    Reads data from TSV files specified by the filepath.
    :param filepath: the path to a file (e.g., train.tsv) or multiple files (e.g., folder/*.tsv).
    :type filepath: str
    :param create_state: a function that takes a document and returns a state.
    :type create_state: (list of elit.util.structure.Sentence) -> elit.nlp.NLPState
    :param kwargs: a ditionary containing the column index of each field.
    :type kwargs: dict
    :return: a list of states containing documents, where each document is a list of sentences.
    :rtype: list of elit.nlp.NLPState
    """
    def create_dict():
        return {k: [] for k in cols.keys()}

    def aux(filename):
        fin = open(filename)
        d = create_dict()
        wc = 0

        for line in fin:
            l = line.split()
            if l:
                for k, v in cols.items():
                    if k == DEP:  # (head ID, deprel)
                        f = (int(l[v[0]]) - 1, l[v[1]])
                    else:
                        f = l[v]

                    d[k].append(f)
            elif d[TOKEN]:
                sentences.append(Sentence(d))
                wc += len(sentences[-1])
                d = create_dict()

        return wc

    sentences = []
    word_count = 0
    for file in glob.glob(filepath): word_count += aux(file)
    states = group_states(sentences, create_state)
    logging.info('Read: %s (sc = %d, wc = %d, grp = %d)' % (filepath, len(sentences), word_count, len(states)))
    return states


def group_states(sentences, create_state, max_len=-1):
    """
    Groups sentences into documents such that each document consists of multiple sentences and the total number of words
    across all sentences within a document is close to the specified maximum length.
    :param sentences: list of sentences.
    :type sentences: list of elit.util.structure.Sentence
    :param create_state: a function that takes a document and returns a state.
    :type create_state: (list of elit.util.structure.Sentence) -> elit.nlp.NLPState
    :param max_len: the maximum number of words; if max_len < 0, it is inferred by the length of the longest sentence.
    :type max_len: int
    :return: list of states, where each state roughly consists of the max_len number of words.
    :rtype: list of elit.nlp.NLPState
    """
    def aux(i):
        ls = d[keys[i]]
        t = ls.pop()
        document.append(t)
        if not ls: del keys[i]
        return len(t)

    # key = length, value = list of sentences with the key length
    d = {}
    for s in sentences: d.setdefault(len(s), []).append(s)
    keys = sorted(list(d.keys()))
    if max_len < 0: max_len = keys[-1]

    states = []
    document = []
    wc = max_len - aux(-1)

    while keys:
        idx = bisect.bisect_left(keys, wc)
        if idx >= len(keys) or keys[idx] > wc:
            idx -= 1
        if idx < 0:
            states.append(create_state(document))
            document = []
            wc = max_len - aux(-1)
        else:
            wc -= aux(idx)

    if document: states.append(create_state(document))
    return states


def data_loader(states, batch_size, shuffle=False):
    """
    :param states: the list of NLP states.
    :type states: list of elit.nlp.NLPState
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


def process_online(model, states, batch_size, ctx, trainer=None, loss_func=None, eval_counts=None, reshape_x=None, xs=None, ys=None, reset=True):
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
        if eval_counts: model.eval(state, eval_counts)
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


def argparse_train(title):


    return parser


def reshape_conv2d(x):
    return x.reshape((0, 1, x.shape[1], x.shape[2]))


class LabelMap:
    """
    LabelMap gives the mapping between class labels and their unique IDs.
    """
    def __init__(self):
        self.index_map = {}
        self.labels = []

    def __len__(self):
        return len(self.labels)

    def index(self, label):
        """
        :param label: the class label.
        :type label: str
        :return: the ID of the class label if exists; otherwise, -1.
        :rtype: int
        """
        return self.index_map[label] if label in self.index_map else -1

    def get(self, index):
        """
        :param index: the ID of the class label.
        :type index: int
        :return: the index'th class label.
        :rtype: str
        """
        return self.labels[index]

    def add(self, label):
        """
        Adds the class label to this map if not already exist.
        :param label: the class label.
        :type label: str
        :return: the ID of the class label.
        :rtype int
        """
        idx = self.index(label)
        if idx < 0:
            idx = len(self.labels)
            self.index_map[label] = idx
            self.labels.append(label)
        return idx