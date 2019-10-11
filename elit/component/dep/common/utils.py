# coding: utf-8

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import heapq
from typing import Sequence

import mxnet as mx
import mxnet.ndarray as nd
import numpy as np
from mxnet.gluon import rnn, nn
from mxnet.gluon.contrib.rnn import VariationalDropoutCell

from elit.component.dep.common.data import ParserVocabulary
from elit.component.dep.common.tarjan import Tarjan
from elit.structure import Document, DEP, Sentence, POS, SENS, ConllWord, ConllSentence


def orthonormal_VanillaLSTMBuilder(lstm_layers, input_dims, lstm_hiddens, dropout_x=0., dropout_h=0., debug=False):
    """Build a standard LSTM cell, with variational dropout,
    with weights initialized to be orthonormal (https://arxiv.org/abs/1312.6120)

    Parameters
    ----------
    lstm_layers : int
        Currently only support one layer
    input_dims : int
        word vector dimensions
    lstm_hiddens : int
        hidden size
    dropout_x : float
        dropout on inputs, not used in this implementation, see `biLSTM` below
    dropout_h : float
        dropout on hidden states
    debug : bool
        set to True to skip orthonormal initialization

    Returns
    -------
    lstm_cell : VariationalDropoutCell
        A LSTM cell
    """
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


def biLSTM(f_lstm, b_lstm, inputs, batch_size=None, dropout_x=0., dropout_h=0.):
    """Feature extraction through BiLSTM

    Parameters
    ----------
    f_lstm : VariationalDropoutCell
        Forward cell
    b_lstm : VariationalDropoutCell
        Backward cell
    inputs : NDArray
        seq_len x batch_size
    dropout_x : float
        Variational dropout on inputs
    dropout_h :
        Not used

    Returns
    -------
    outputs : NDArray
        Outputs of BiLSTM layers, seq_len x 2 hidden_dims x batch_size
    """
    for f, b in zip(f_lstm, b_lstm):
        inputs = nd.Dropout(inputs, dropout_x, axes=[0])  # important for variational dropout
        fo, fs = f.unroll(length=inputs.shape[0], inputs=inputs, layout='TNC', merge_outputs=True)
        bo, bs = b.unroll(length=inputs.shape[0], inputs=inputs.flip(axis=0), layout='TNC', merge_outputs=True)
        f.reset(), b.reset()
        inputs = nd.concat(fo, bo.flip(axis=0), dim=2)
    return inputs


def leaky_relu(x):
    """slope=0.1 leaky ReLu

    Parameters
    ----------
    x : NDArray
        Input

    Returns
    -------
    y : NDArray
        y = x > 0 ? x : 0.1 * x
    """
    return nd.LeakyReLU(x, slope=.1)


def bilinear(x, W, y, input_size, seq_len, batch_size, num_outputs=1, bias_x=False, bias_y=False):
    """Do xWy

    Parameters
    ----------
    x : NDArray
        (input_size x seq_len) x batch_size
    W : NDArray
        (num_outputs x ny) x nx
    y : NDArray
        (input_size x seq_len) x batch_size
    input_size : int
        input dimension
    seq_len : int
        sequence length
    batch_size : int
        batch size
    num_outputs : int
        number of outputs
    bias_x : bool
        whether concat bias vector to input x
    bias_y : bool
        whether concat bias vector to input y

    Returns
    -------
    output : NDArray
        [seq_len_y x seq_len_x if output_size == 1 else seq_len_y x num_outputs x seq_len_x] x batch_size
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
    y = y.transpose([2, 1, 0])  # May cause performance issues
    lin = lin.transpose([2, 1, 0])
    blin = nd.batch_dot(lin, y, transpose_b=True)
    blin = blin.transpose([2, 1, 0])
    if num_outputs > 1:
        blin = reshape_fortran(blin, (seq_len, num_outputs, seq_len, batch_size))
    return blin


def orthonormal_initializer(output_size, input_size, debug=False):
    """adopted from Timothy Dozat https://github.com/tdozat/Parser/blob/master/lib/linalg.py

    Parameters
    ----------
    output_size : int
    input_size : int
    debug : bool
        Whether to skip this initializer
    Returns
    -------
    Q : np.ndarray
        The orthonormal weight matrix of input_size x output_size
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
    if success:
        # print(('Orthogonal pretrainer loss: %.2e' % loss))
        pass
    else:
        # print('Orthogonal pretrainer failed, using non-orthogonal random matrix')
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
    return np.transpose(Q.astype(np.float32))


def arc_argmax(parse_probs, length, tokens_to_keep, ensure_tree=True):
    """
    Adopted from Timothy Dozat https://github.com/tdozat/Parser/blob/master/lib/models/nn.py

    Parameters
    ----------
    parse_probs : NDArray
        seq_len x seq_len, the probability of arcs
    length : NDArray
        real sentence length
    tokens_to_keep : NDArray
        mask matrix
    ensure_tree :
        whether to ensure tree structure of output (apply MST)
    Returns
    -------
    parse_preds : np.ndarray
        prediction of arc parsing with size of (seq_len,)
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


def arc_mst(parse_probs, length, tokens_to_keep, want_max=True):
    # block and pad heads
    parse_probs[0] = 1. / length
    np.fill_diagonal(parse_probs, 0)
    parse_probs = parse_probs * tokens_to_keep
    parse_probs = parse_probs.T + 1e-20
    if want_max:
        parse_probs = -np.log(parse_probs)
    mincost = [1e20] * length
    mincost[0] = 0
    used = [False] * length
    que = []
    heads = [-1] * length
    heapq.heappush(que, (0, 0, 0))  # cost, to, from
    total_cost = 0
    while que:
        cost, v, fr = heapq.heappop(que)
        if used[v] or cost > mincost[v]:
            continue
        used[v] = True
        total_cost += mincost[v]
        heads[v] = fr
        for i in range(0, length):
            if mincost[i] > parse_probs[v][i]:
                mincost[i] = parse_probs[v][i]
                heapq.heappush(que, (mincost[i], i, v))
    return heads


def rel_argmax(rel_probs, length, ensure_tree=True):
    """Fix the relation prediction by heuristic rules

    Parameters
    ----------
    rel_probs : NDArray
        seq_len x rel_size
    length :
        real sentence length
    ensure_tree :
        whether to apply rules
    Returns
    -------
    rel_preds : np.ndarray
        prediction of relations of size (seq_len,)
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


def reshape_fortran(tensor, shape):
    """The missing Fortran reshape for mx.NDArray

    Parameters
    ----------
    tensor : NDArray
        source tensor
    shape : NDArray
        desired shape

    Returns
    -------
    output : NDArray
        reordered result
    """
    return tensor.T.reshape(tuple(reversed(shape))).T


def _save_conll(documents: Sequence[Document], out):
    with open(out, 'w') as out:
        for doc in documents:
            for sent in doc.sentences:
                words = []
                for word, tag, (head, rel) in zip(sent.tokens, sent.part_of_speech_tags, sent[DEP]):
                    words.append(ConllWord(id=len(words) + 1, form=word, pos=tag, head=head, relation=rel))
                out.write(str(ConllSentence(words)))
                out.write('\n\n')


def _load_conll(path) -> Document:
    """
    Load whole conll file as a document
    :param path: .conll or .conllx file
    :return: single document
    """

    def create_sentence() -> Sentence:
        sent = Sentence()
        sent[POS] = []
        sent[DEP] = []
        return sent

    sents = []
    with open(path) as src:
        sent = create_sentence()
        for line in src:
            info = line.strip().split()
            if info:
                assert (len(info) == 10), 'Illegal line: %s' % line
                word, tag, head, rel = info[1], info[3], int(info[6]), info[7]
                sent.tokens.append(word)
                sent.part_of_speech_tags.append(tag)
                sent[DEP].append((head, rel))
            else:
                sents.append(sent)
                sent = create_sentence()
    return Document({SENS: sents})


def flatten_numpy(ndarray):
    """Flatten nd-array to 1-d column vector

    Parameters
    ----------
    ndarray : numpy.ndarray
        input tensor

    Returns
    -------
    numpy.ndarray
        A column vector

    """
    return np.reshape(ndarray, (-1,), 'F')


def flatten_numpy(ndarray):
    """Flatten nd-array to 1-d column vector

    Parameters
    ----------
    ndarray : numpy.ndarray
        input tensor

    Returns
    -------
    numpy.ndarray
        A column vector

    """
    return np.reshape(ndarray, (-1,), 'F')


def embedding_from_numpy(we, trainable=True):
    word_embs = nn.Embedding(we.shape[0], we.shape[1], weight_initializer=mx.init.Constant(we))
    if not trainable:
        word_embs.collect_params().setattr('grad_req', 'null')
    return word_embs


def parameter_from_numpy(model, name, array):
    """ Create parameter with its value initialized according to a numpy tensor

    Parameters
    ----------
    name : str
        parameter name
    array : np.ndarray
        initiation value

    Returns
    -------
    mxnet.gluon.parameter
        a parameter object
    """
    p = model.params.get(name, shape=array.shape, init=mx.init.Constant(array))
    return p


def parameter_init(model, name, shape, init):
    """Create parameter given name, shape and initiator

    Parameters
    ----------
    name : str
        parameter name
    shape : tuple
        parameter shape
    init : mxnet.initializer
        an initializer

    Returns
    -------
    mxnet.gluon.parameter
        a parameter object
    """
    p = model.params.get(name, shape=shape, init=init)
    return p


def freeze(model):
    model.collect_params().setattr('grad_req', 'null')


def _test_mst():
    length = 7
    score = np.zeros((length, length))
    score[0, 1] = score[1, 0] = 10
    score[0, 2] = score[2, 0] = 2
    score[1, 3] = score[3, 1] = 5
    score[2, 3] = score[3, 2] = 7
    score[2, 4] = score[4, 2] = 1
    score[2, 5] = score[5, 2] = 3
    score[3, 5] = score[5, 3] = 1
    score[3, 6] = score[6, 3] = 8
    score[5, 6] = score[6, 5] = 5
    tree = arc_mst(score, length, None)
    print(tree)


if __name__ == '__main__':
    # print(fetch_resource(EN_LM_FLAIR_FW_WMT11))
    _test_mst()
