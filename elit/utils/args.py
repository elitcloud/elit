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
import re
from types import SimpleNamespace
from typing import Tuple

import argparse
import mxnet as mx

from elit.model import conv2d_namespace
from elit.utils.file import tsv_reader, json_reader

__author__ = "Gary Lai"

# ======================================== Argument ========================================


def conv2d_args(s: str) -> Tuple[SimpleNamespace, ...]:
    """
    :param s: [ngram,filters,activation,pool,dropout ...]
    :return: a tuple of conf2d_namespace()
    """
    def create(config):
        c = config.split(',')
        pool = c[3] if c[3].lower() != 'none' else None
        return conv2d_namespace(ngram=int(c[0]), filters=int(c[1]), activation=c[2], pool=pool,
                                dropout=float(c[4]))
    return tuple(create(config) for config in s) if s is not None else None


def hidden_args(s: list) -> Tuple[SimpleNamespace, ...]:
    """
    :param s: [dim,activation,dropout ...]
    :return: a tuple of hidden_namespace()
    """

    def create(config):
        c = config.split(',')
        return SimpleNamespace(dim=int(c[0]), activation=c[1], dropout=float(c[2]))

    return tuple(create(config) for config in s) if s is not None else None


def context_args(s: str) -> mx.Context:
    """
    :param s: [cg]\\d*
    :return: a device context
    """
    m = re.match(r"([cg])(\d*)", s)
    if m:
        d = int(m.group(2))
        if m.group(1) == 'c':
            return mx.cpu(d)
        else:
            return mx.gpu(d)
    else:
        import argparse
        raise argparse.ArgumentTypeError


def loss_args(s: str) -> mx.gluon.loss.Loss:
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


def reader_args(s):
    """
    :param s: (tsv|json)(:\d,\d)*
    :return: reader, SimpleNamespace(tok, pos)
    """
    # Semicolon ";" will be seen as an argument separator.
    # Thus, I changed it to colon ":" and comma ",".
    m = re.match(r"(tsv|json)(:\d,\d)*", s)
    if m:
        r = s.split(':')
        if r[0] == 'tsv':
            t = r[1].split(',')
            return tsv_reader, SimpleNamespace(tok=int(t[0]), tag=int(t[1]))
        elif r[0] == 'json':
            return json_reader, None
        else:
            raise argparse.ArgumentTypeError
    else:
        raise argparse.ArgumentTypeError


def feature_windows_args(s):
    """
    :param s: \\d(,\\d)*
    :return: tuple of int
    """
    m = re.match(r"\d(,\d)*")
    if m:
        return tuple(map(int, s.split(',')))
    else:
        raise argparse.ArgumentTypeError