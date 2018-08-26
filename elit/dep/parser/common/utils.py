# -*- coding: UTF-8 -*-
from elit.dep.common.utils import stdchannel_redirected
import numpy as np
import sys
import os

with stdchannel_redirected(sys.stderr, os.devnull):
    import dynet as dy

from .data import ParserVocabulary
from .tarjan import Tarjan


def orthonormal_VanillaLSTMBuilder(lstm_layers, input_dims, lstm_hiddens, pc, debug=False):
    builder = dy.VanillaLSTMBuilder(lstm_layers, input_dims, lstm_hiddens, pc)
    for layer, params in enumerate(builder.get_parameters()):
        W = orthonormal_initializer(lstm_hiddens, lstm_hiddens + (lstm_hiddens if layer > 0 else input_dims), debug)
        W_h, W_x = W[:, :lstm_hiddens], W[:, lstm_hiddens:]
        params[0].set_value(np.concatenate([W_x] * 4, 0))
        params[1].set_value(np.concatenate([W_h] * 4, 0))
        b = np.zeros(4 * lstm_hiddens, dtype=np.float32)
        b[lstm_hiddens:2 * lstm_hiddens] = -1.0
        params[2].set_value(b)
    return builder


def biLSTM(builders, inputs, batch_size=None, dropout_x=0., dropout_h=0.):
    """
    Feature extraction
    :param builders: BiLSTM layers
    :param inputs: # seq_len x batch_size
    :param batch_size:
    :param dropout_x:
    :param dropout_h:
    :return: Outputs of BiLSTM layers, seq_len x 2 hidden_dims x batch_size
    """
    for fb, bb in builders:
        f, b = fb.initial_state(), bb.initial_state()
        fb.set_dropouts(dropout_x, dropout_h)
        bb.set_dropouts(dropout_x, dropout_h)
        if batch_size is not None:
            fb.set_dropout_masks(batch_size)
            bb.set_dropout_masks(batch_size)
        fs, bs = f.transduce(inputs), b.transduce(reversed(inputs))
        inputs = [dy.concatenate([f, b]) for f, b in zip(fs, reversed(bs))]
    return inputs


def LSTM(lstm, inputs, batch_size=None, dropout_x=0., dropout_h=0.):
    """
    unidirectional LSTM
    :param lstm: one LSTM layer
    :param inputs: # seq_len x batch_size
    :param batch_size:
    :param dropout_x:
    :param dropout_h:
    :return: Output of LSTM layer, seq_len x hidden_dim x batch_size
    """
    s = lstm.initial_state()
    lstm.set_dropouts(dropout_x, dropout_h)
    if batch_size is not None:
        lstm.set_dropout_masks(batch_size)
    # hs = s.transduce(inputs)
    hs = s.add_inputs(inputs)
    return hs


def attention(hs, w):
    H = dy.concatenate_cols(hs)
    a = dy.softmax(dy.transpose(w) * H)
    h = H * dy.transpose(a)
    return h


def leaky_relu(x):
    return dy.bmax(.1 * x, x)


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


def orthonormal_initializer(output_size, input_size, debug=False):
    """
    adopted from Timothy Dozat https://github.com/tdozat/Parser/blob/master/lib/linalg.py
    """
    # print((output_size, input_size))
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
    # if success:
    #     print(('Orthogonal pretrainer loss: %.2e' % loss))
    # else:
    #     print('Orthogonal pretrainer failed, using non-orthogonal random matrix')
    if not success:
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


def one_hot(id_vector, vocabulary_size):
    """
    Generate one hot vector for every id in id_vector

    :param id_vector: A vector of ids
    :param vocabulary_size: How big the vocabulary is
    :return: A matrix of (len(id_vector), vocabulary_size)
    """
    b = np.zeros((len(id_vector), vocabulary_size), dtype=np.int32)
    b[np.arange(len(id_vector)), id_vector] = 1
    return b
