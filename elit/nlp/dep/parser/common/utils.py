# ========================================================================
# Copyright 2018 ELIT
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
# -*- coding: UTF-8 -*-
import sys

import mxnet as mx
import numpy as np
import mxnet.ndarray as nd
from mxnet.gluon import rnn
from mxnet.gluon.contrib.rnn import VariationalDropoutCell
from mxnet.gluon.rnn import BidirectionalCell

from .data import ParserVocabulary
from .tarjan import Tarjan


def orthonormal_VanillaLSTMBuilder(lstm_layers, input_dims, lstm_hiddens, dropout_x=0., dropout_h=0., debug=False):
    assert lstm_layers == 1, 'only accept one layer lstm'
    W = orthonormal_initializer(lstm_hiddens, lstm_hiddens + input_dims, debug)
    W_h, W_x = W[:, :lstm_hiddens], W[:, lstm_hiddens:]
    b = nd.zeros((4 * lstm_hiddens,))
    b[lstm_hiddens:2 * lstm_hiddens] = -1.0
    lstm_cell = rnn.LSTMCell(input_size=input_dims, hidden_size=lstm_hiddens,
                             i2h_weight_initializer=mx.init.Constant(np.concatenate([W_x] * 4, 0)),
                             h2h_weight_initializer=mx.init.Constant(np.concatenate([W_h] * 4, 0)),
                             h2h_bias_initializer=mx.init.Constant(b))
    wrapper = VariationalDropoutCell(lstm_cell, drop_states=dropout_h)
    return wrapper


def orthonormal_VanillaBiLSTMBuilder(lstm_layers, input_dims, lstm_hiddens, dropout_x=0., dropout_h=0., debug=False):
    return BidirectionalCell(
        orthonormal_VanillaLSTMBuilder(lstm_layers, input_dims, lstm_hiddens, dropout_x, dropout_h, debug),
        orthonormal_VanillaLSTMBuilder(lstm_layers, input_dims, lstm_hiddens, dropout_x, dropout_h, debug),
    )


def biLSTM(f_lstm, b_lstm, inputs, batch_size=None, dropout_x=0., dropout_h=0.):
    """
    Feature extraction
    :param inputs: # seq_len x batch_size
    :param batch_size:
    :return: Outputs of BiLSTM layers, seq_len x 2 hidden_dims x batch_size
    """
    for f, b in zip(f_lstm, b_lstm):
        inputs = nd.Dropout(inputs, dropout_x, axes=[0])
        fo, fs = f.unroll(length=inputs.shape[0], inputs=inputs, layout='TNC', merge_outputs=True)
        bo, bs = b.unroll(length=inputs.shape[0], inputs=inputs.flip(axis=0), layout='TNC', merge_outputs=True)
        f.reset(), b.reset()
        inputs = nd.concat(fo, bo.flip(axis=0), dim=2)
    return inputs


def leaky_relu(x):
    return nd.LeakyReLU(x, slope=.1)


def bilinear(x, W, y, input_size, seq_len, batch_size, num_outputs=1, bias_x=False, bias_y=False):
    """
    Do xWy

    :param x: (input_size x seq_len) x batch_size
    :param W:
    :param y: (input_size x seq_len) x batch_size
    :param input_size:
    :param seq_len:
    :param batch_size:
    :param num_outputs:
    :param bias_x:
    :param bias_y:
    :return: [seq_len_y x seq_len_x if output_size == 1 else seq_len_y x num_outputs x seq_len_x] x batch_size
    """
    if bias_x:
        x = nd.concat(x, nd.ones((1, seq_len, batch_size)), dim=0)
    if bias_y:
        y = nd.concat(y, nd.ones((1, seq_len, batch_size)), dim=0)

    nx, ny = input_size + bias_x, input_size + bias_y
    # W: (num_outputs x ny) x nx
    lin = nd.dot(W, x)
    if num_outputs > 1:
        lin = reshape_fortran(lin, (ny, num_outputs * seq_len, batch_size))
    y = y.transpose([2, 1, 0])
    lin = lin.transpose([2, 1, 0])
    blin = nd.batch_dot(lin, y, transpose_b=True)
    blin = blin.transpose([2, 1, 0])
    if num_outputs > 1:
        blin = reshape_fortran(blin, (seq_len, num_outputs, seq_len, batch_size))
    return blin


def dynet_bilinear(x, W, y, input_size, seq_len, batch_size, num_outputs=1, bias_x=False, bias_y=False):
    """
    Do xWy

    :param x: (input_size x seq_len) x batch_size
    :param W:
    :param y: (input_size x seq_len) x batch_size
    :param input_size:
    :param seq_len:
    :param batch_size:
    :param num_outputs:
    :param bias_x:
    :param bias_y:
    :return: [seq_len_y x seq_len_x if output_size == 1 else seq_len_y x num_outputs x seq_len_x] x batch_size
    """
    import dynet as dy
    if isinstance(x, np.ndarray):
        x = dy.inputTensor(x, batched=True)
    if isinstance(y, np.ndarray):
        y = dy.inputTensor(y, batched=True)
    if isinstance(W, np.ndarray):
        W = dy.inputTensor(W)

    if bias_x:
        x = dy.concatenate([x, dy.inputTensor(np.ones((1, seq_len), dtype=np.float32))])
    if bias_y:
        y = dy.concatenate([y, dy.inputTensor(np.ones((1, seq_len), dtype=np.float32))])

    nx, ny = input_size + bias_x, input_size + bias_y
    # W: (num_outputs x ny) x nx
    lin = W * x
    if num_outputs > 1:
        lin = dy.reshape(lin, (ny, num_outputs * seq_len), batch_size=batch_size)
    blin = dy.transpose(y) * lin
    if num_outputs > 1:
        blin = dy.reshape(blin, (seq_len, num_outputs, seq_len), batch_size=batch_size)
    return blin


def debug_bilinear(x, W, y, input_size, seq_len, batch_size, num_outputs=1, bias_x=False, bias_y=False):
    """
    Do xWy

    :param x: (input_size x seq_len) x batch_size
    :param W:
    :param y: (input_size x seq_len) x batch_size
    :param input_size:
    :param seq_len:
    :param batch_size:
    :param num_outputs:
    :param bias_x:
    :param bias_y:
    :return: [seq_len_y x seq_len_x if output_size == 1 else seq_len_y x num_outputs x seq_len_x] x batch_size
    """
    import dynet as dy
    xd = dy.inputTensor(x, batched=True)
    xm = nd.array(x)
    yd = dy.inputTensor(y, batched=True)
    ym = nd.array(y)
    Wd = dy.inputTensor(W)
    Wm = nd.array(W)

    def allclose(dyarray, mxarray):
        a = dyarray.npvalue()
        b = mxarray.asnumpy()
        return np.allclose(a, b)

    if bias_x:
        xd = dy.concatenate([xd, dy.inputTensor(np.ones((1, seq_len), dtype=np.float32))])
        xm = nd.concat(xm, nd.ones((1, seq_len, batch_size)), dim=0)
        # print(allclose(xd, xm))
    if bias_y:
        yd = dy.concatenate([yd, dy.inputTensor(np.ones((1, seq_len), dtype=np.float32))])
        ym = nd.concat(ym, nd.ones((1, seq_len, batch_size)), dim=0)
        # print(allclose(yd, ym))

    nx, ny = input_size + bias_x, input_size + bias_y
    # W: (num_outputs x ny) x nx
    lind = Wd * xd
    linm = nd.dot(Wm, xm)
    # print(allclose(lind, linm))
    if num_outputs > 1:
        lind = dy.reshape(lind, (ny, num_outputs * seq_len), batch_size=batch_size)
        # linm = nd.reshape(linm, (ny, num_outputs * seq_len, batch_size))
        linm = reshape_fortran(linm, (ny, num_outputs * seq_len, batch_size))
        # print(allclose(lind, linm))

    blind = dy.transpose(yd) * lind
    ym = ym.transpose([2, 1, 0])
    linm = linm.transpose([2, 1, 0])
    blinm = nd.batch_dot(linm, ym, transpose_b=True)
    blinm = blinm.transpose([2, 1, 0])

    print(np.allclose(blind.npvalue(), blinm.asnumpy()))

    if num_outputs > 1:
        blind = dy.reshape(blind, (seq_len, num_outputs, seq_len), batch_size=batch_size)
        blinm = reshape_fortran(blinm, (seq_len, num_outputs, seq_len, batch_size))
        print(allclose(blind, blinm))
    return blind


def orthonormal_initializer(output_size, input_size, debug=False):
    """
    adopted from Timothy Dozat https://github.com/tdozat/Parser/blob/master/lib/linalg.py
    """
    print((output_size, input_size))
    if debug:
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
        return np.transpose(Q.astype(np.float32))
    I = np.eye(output_size)
    lr = .1
    eps = .05 / (output_size + input_size)
    success = False
    tries = 0
    while not success and tries < 10:
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
        for i in range(100):
            QTQmI = Q.T.dot(Q) - I
            loss = np.sum(QTQmI ** 2 / 2)
            Q2 = Q ** 2
            Q -= lr * Q.dot(QTQmI) / (
                    np.abs(Q2 + Q2.sum(axis=0, keepdims=True) + Q2.sum(axis=1, keepdims=True) - 1) + eps)
            if np.max(Q) > 1e6 or loss > 1e6 or not np.isfinite(loss):
                tries += 1
                lr /= 2
                break
        success = True
    if success:
        print(('Orthogonal pretrainer loss: %.2e' % loss))
    else:
        print('Orthogonal pretrainer failed, using non-orthogonal random matrix')
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
    return np.transpose(Q.astype(np.float32))


def arc_argmax(parse_probs, length, tokens_to_keep, ensure_tree=True):
    """
    MST
    adopted from Timothy Dozat https://github.com/tdozat/Parser/blob/master/lib/models/nn.py

    :param parse_probs: seq_len x seq_len, the probability of arcs
    :param length: true sentence length
    :param tokens_to_keep: seq_len, legal tokens
    :param ensure_tree:
    :return: seq_len, prediction of arc
    """
    if ensure_tree:
        I = np.eye(len(tokens_to_keep))
        # block loops and pad heads
        parse_probs = parse_probs * tokens_to_keep * (1 - I)
        parse_preds = np.argmax(parse_probs, axis=1)
        tokens = np.arange(1, length)
        roots = np.where(parse_preds[tokens] == 0)[0] + 1
        # ensure at least one root
        if len(roots) < 1:
            # The current root probabilities
            root_probs = parse_probs[tokens, 0]
            # The current head probabilities
            old_head_probs = parse_probs[tokens, parse_preds[tokens]]
            # Get new potential root probabilities
            new_root_probs = root_probs / old_head_probs
            # Select the most probable root
            new_root = tokens[np.argmax(new_root_probs)]
            # Make the change
            parse_preds[new_root] = 0
        # ensure at most one root
        elif len(roots) > 1:
            # The probabilities of the current heads
            root_probs = parse_probs[roots, 0]
            # Set the probability of depending on the root zero
            parse_probs[roots, 0] = 0
            # Get new potential heads and their probabilities
            new_heads = np.argmax(parse_probs[roots][:, tokens], axis=1) + 1
            new_head_probs = parse_probs[roots, new_heads] / root_probs
            # Select the most probable root
            new_root = roots[np.argmin(new_head_probs)]
            # Make the change
            parse_preds[roots] = new_heads
            parse_preds[new_root] = 0
        # remove cycles
        tarjan = Tarjan(parse_preds, tokens)
        cycles = tarjan.SCCs
        for SCC in tarjan.SCCs:
            if len(SCC) > 1:
                dependents = set()
                to_visit = set(SCC)
                while len(to_visit) > 0:
                    node = to_visit.pop()
                    if not node in dependents:
                        dependents.add(node)
                        to_visit.update(tarjan.edges[node])
                # The indices of the nodes that participate in the cycle
                cycle = np.array(list(SCC))
                # The probabilities of the current heads
                old_heads = parse_preds[cycle]
                old_head_probs = parse_probs[cycle, old_heads]
                # Set the probability of depending on a non-head to zero
                non_heads = np.array(list(dependents))
                parse_probs[np.repeat(cycle, len(non_heads)), np.repeat([non_heads], len(cycle), axis=0).flatten()] = 0
                # Get new potential heads and their probabilities
                new_heads = np.argmax(parse_probs[cycle][:, tokens], axis=1) + 1
                new_head_probs = parse_probs[cycle, new_heads] / old_head_probs
                # Select the most probable change
                change = np.argmax(new_head_probs)
                changed_cycle = cycle[change]
                old_head = old_heads[change]
                new_head = new_heads[change]
                # Make the change
                parse_preds[changed_cycle] = new_head
                tarjan.edges[new_head].add(changed_cycle)
                tarjan.edges[old_head].remove(changed_cycle)
        return parse_preds
    else:
        # block and pad heads
        parse_probs = parse_probs * tokens_to_keep
        parse_preds = np.argmax(parse_probs, axis=1)
        return parse_preds


def rel_argmax(rel_probs, length, ensure_tree=True):
    """
    adopted from Timothy Dozat https://github.com/tdozat/Parser/blob/master/lib/models/nn.py

    :param rel_probs: seq_len x #rel
    :param length:
    :param ensure_tree:
    :return:
    """
    if ensure_tree:
        rel_probs[:, ParserVocabulary.PAD] = 0
        root = ParserVocabulary.ROOT
        tokens = np.arange(1, length)
        rel_preds = np.argmax(rel_probs, axis=1)
        roots = np.where(rel_preds[tokens] == root)[0] + 1
        if len(roots) < 1:
            rel_preds[1 + np.argmax(rel_probs[tokens, root])] = root
        elif len(roots) > 1:
            root_probs = rel_probs[roots, root]
            rel_probs[roots, root] = 0
            new_rel_preds = np.argmax(rel_probs[roots], axis=1)
            new_rel_probs = rel_probs[roots, new_rel_preds] / root_probs
            new_root = roots[np.argmin(new_rel_probs)]
            rel_preds[roots] = new_rel_preds
            rel_preds[new_root] = root
        return rel_preds
    else:
        rel_probs[:, ParserVocabulary.PAD] = 0
        rel_preds = np.argmax(rel_probs, axis=1)
        return rel_preds


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def reshape_fortran(tensor, shape):
    return tensor.T.reshape(tuple(reversed(shape))).T


if __name__ == '__main__':
    C = 2
    T = 3
    N = 4
    O = 5

    x = np.random.normal(0, 1, (C, T, N))
    W = np.random.normal(0, 1, (C, C + 1))
    U = np.random.normal(0, 1, ((C + 1) * O, C + 1))
    y = np.random.normal(0, 1, (C, T, N))
    # single output
    # zmxnet = bilinear(nd.array(x), nd.array(W), nd.array(y), C, IN, NN, 1, True, False).asnumpy()
    # zdynet = dynet_bilinear(x, W, y, C, IN, NN, 1, True, False).npvalue()
    # print(zmxnet.shape)
    # print(zdynet.shape)
    # print(np.allclose(zmxnet, zdynet))

    # multiple output
    zmxnet = bilinear(nd.array(x), nd.array(U), nd.array(y), C, T, N, O, True, True).asnumpy()
    zdynet = dynet_bilinear(x, U, y, C, T, N, O, True, True).npvalue()
    print(zmxnet.shape)
    print(zdynet.shape)
    print(np.allclose(zmxnet, zdynet))

    # debug_bilinear(x, U, y, C, IN, NN, O, True, True)
