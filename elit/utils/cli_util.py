# ========================================================================
# Copyright 2018 Emory University
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
import argparse
import re
from types import SimpleNamespace
from typing import Tuple, Type, Dict, Union, Optional

import mxnet as mx

from elit.utils.io_util import tsv_reader, json_reader
from elit.vsm import VectorSpaceModel, Word2Vec, FastText

__author__ = "Gary Lai, Jinho D. Choi"


# ======================================== Namespaces ========================================

def namespace_reader(reader_type: Union[json_reader, tsv_reader], params: Optional[Dict] = None) -> SimpleNamespace:
    return SimpleNamespace(type=reader_type, params=params)


def namespace_vsm(vsm_type: Type[VectorSpaceModel], key: str, filepath: str) -> SimpleNamespace:
    return SimpleNamespace(type=vsm_type, key=key, filepath=filepath)


def namespace_input(col: int, row: int, dropout: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(col=col, row=row, dropout=dropout)


def namespace_output(dim: int) -> SimpleNamespace:
    return SimpleNamespace(dim=dim)


def namespace_conv2d(ngram: int, filters: int, activation: str, pool: str = None, dropout: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(ngram=ngram, filters=filters, activation=activation, pool=pool, dropout=dropout)


def namespace_hidden(dim: int, activation: str, dropout: float) -> SimpleNamespace:
    return SimpleNamespace(dim=dim, activation=activation, dropout=dropout)


# ======================================== Arguments ========================================

def args_dict_str_int(s: str) -> Dict[str, int]:
    """
    :param s: key:int(,key:int)*
    :return: a dictionary including key-value pairs from the input string.
    """
    r = re.compile(r'([A-Za-z0-9_-]+):(\d+)')
    d = {}

    for t in s.split(','):
        m = r.match(t)
        if m:
            d[m.group(1)] = int(m.group(2))
        else:
            raise argparse.ArgumentTypeError

    if not d: raise argparse.ArgumentTypeError
    return d


def args_tuple_int(s: str) -> Tuple[int, ...]:
    """
    :param s: int(, int)*
    :return: tuple of integers
    """
    r = re.compile(r'-?\d+(,-?\d+)*$')
    if r.match(s):
        return tuple(map(int, s.split(',')))
    else:
        raise argparse.ArgumentTypeError


def args_reader(s: str) -> SimpleNamespace:
    """
    :param s: json or tsv=(str:int)(,#1)*
    :return: the output of namespace_reader().
    """
    if s == 'json': return namespace_reader(reader_type=json_reader)
    if s.startswith('tsv='): return namespace_reader(reader_type=tsv_reader, params=args_dict_str_int(s[4:]))
    raise argparse.ArgumentTypeError


def args_vsm(s: str) -> SimpleNamespace:
    """
    :param s: (fasttext|word2vec):_key:filepath
    :return: the output of namespace_vsm().
    """
    i = s.find(':')
    if i < 1: raise argparse.ArgumentTypeError

    v = s[:i]
    if v == 'fasttext':
        vsm = FastText
    elif v == 'word2vec':
        vsm = Word2Vec
    else:
        raise argparse.ArgumentTypeError("Unsupported vector space model: " + v)

    s = s[i + 1:]
    i = s.find(':')
    if i < 1: raise argparse.ArgumentTypeError

    key = s[:i]
    filepath = s[i + 1:]
    if not filepath: raise argparse.ArgumentTypeError

    return namespace_vsm(vsm_type=vsm, key=key, filepath=filepath)


def args_conv2d(s: str) -> Optional[SimpleNamespace]:
    """
    :param s: ngram:filters:activation:pool:dropout
    :return: the output of namespace_conf2d().
    """
    if s.lower() == 'none': return None
    c = s.split(':')
    pool = c[3] if c[3].lower() != 'none' else None
    return namespace_conv2d(ngram=int(c[0]), filters=int(c[1]), activation=c[2], pool=pool, dropout=float(c[4]))


def args_hidden(s: str) -> Optional[SimpleNamespace]:
    """
    :param s: dim:activation:dropout
    :return: the output of namespace_hidden()
    """
    if s.lower() == 'none': return None
    c = s.split(':')
    return SimpleNamespace(dim=int(c[0]), activation=c[1], dropout=float(c[2]))


def args_context(s: str) -> mx.Context:
    """
    :param s: [cg]int
    :return: a device context
    """
    m = re.match(r'([cg])(\d*)', s)
    if m:
        d = int(m.group(2))
        return mx.cpu(d) if m.group(1) == 'c' else mx.gpu(d)
    else:
        raise argparse.ArgumentTypeError


def args_loss(s: str) -> mx.gluon.loss.Loss:
    s = s.lower()

    if s == 'softmaxcrossentropyloss':
        return mx.gluon.loss.SoftmaxCrossEntropyLoss()
    if s == 'sigmoidbinarycrossentropyloss':
        return mx.gluon.loss.SigmoidBinaryCrossEntropyLoss()
    if s == 'l2loss':
        return mx.gluon.loss.L2Loss()
    if s == 'l2loss':
        return mx.gluon.loss.L1Loss()
    if s == 'kldivloss':
        return mx.gluon.loss.KLDivLoss()
    if s == 'huberloss':
        return mx.gluon.loss.HuberLoss()
    if s == 'hingeloss':
        return mx.gluon.loss.HingeLoss()
    if s == 'squaredhingeloss':
        return mx.gluon.loss.SquaredHingeLoss()
    if s == 'logisticloss':
        return mx.gluon.loss.LogisticLoss()
    if s == 'tripletloss':
        return mx.gluon.loss.TripletLoss()
    if s == 'ctcloss':
        return mx.gluon.loss.CTCLoss()

    raise argparse.ArgumentTypeError("Unsupported loss: " + s)
