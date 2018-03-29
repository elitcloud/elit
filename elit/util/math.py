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
import numpy as np


__author__ = 'Jinho D. Choi'


def softmax(array):
    """
    :param array: the input 1-dimensional array.
    :type array: numpy.array
    :return: softmax output of the input array.
    :rtype: numpy.array
    """
    n = np.exp(array)
    d = np.sum(n, axis=0)
    return n/d


def transition_probs(sentences, num_class, getter):
    """
    :param sentences: the input sentences.
    :type sentences: list of elit.util.structure.Sentence
    :param num_class: the total number of classes.
    :type num_class: int
    :return: the transition probability matrix.
    :rtype: numpy.array
    """
    count = np.full((num_class, num_class), 1)  # Laplace smoothing

    for sentence in sentences:
        tags = getter(sentence)
        for j in range(1, len(tags)):
            count[tags[j-1], tags[j]] += 1

    total = count.sum(axis=1).reshape(num_class, 1)
    return np.transpose(count / total)


def viterbi(bigram_probs, scores, num_class):
    probs = np.zeros((len(scores), num_class))
    trace = np.zeros((len(scores), num_class))
    r = range(num_class)

    for i, s in enumerate(scores):
        s = softmax(s)
        if i == 0:
            probs[i] = s
        else:
            p = probs[i-1]
            for j in r:
                t = p * bigram_probs[j]
                m = np.argmax(t)
                probs[i][j] = s[j] + t[m]
                trace[i][j] = m

    pred = [np.argmax(probs[-1])]

    for i in range(len(scores)-1, 0, -1):
        pred.append(int(trace[i][pred[-1]]))

    pred.reverse()
    return np.array(pred)