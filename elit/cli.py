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
import abc
import argparse
import inspect
import logging
import os
import re
import sys
from types import SimpleNamespace
from typing import Callable, Any, Dict, Tuple, Optional, Union, Type

import mxnet as mx

from elit.model import NLPModel
from elit.util.io import json_reader, tsv_reader
from elit.util.vsm import FastText, Word2Vec, VectorSpaceModel

__author__ = "Gary Lai, Jinho D. Choi"


class ComponentCLI(abc.ABC):
    """
    :class:`ComponentCLI` is an abstract class to implement a command-line interface for a component.

    Abstract methods to be implemented:
      - :meth:`ComponentCLI.train`
      - :meth:`ComponentCLI.decode`
      - :meth:`ComponentCLI.evaluate`
    """

    def __init__(self, name: str, description: str = None):
        """
        :param name: the name of the component.
        :param description: the description of this component; if ``None``, the name is used instead.
        """
        usage = [
            '        elit %s <command> [<args>]' %
            name,
            '',
            '    commands:',
            '           train: train a model (gold labels required)',
            '          decode: predict labels (gold labels not required)',
            '        evaluate: evaluate the pre-trained model (gold labels required)']

        usage = '\n'.join(usage)
        if not description:
            description = name
        parser = argparse.ArgumentParser(usage=usage, description=description)
        parser.add_argument('command', help='{} command to run'.format(name))
        args = parser.parse_args(sys.argv[2:3])

        if not hasattr(self, args.command):
            logging.info('Unrecognized command: ' + args.command)
            parser.print_help()
            exit(1)
        getattr(self, args.command)()

    @classmethod
    @abc.abstractmethod
    def train(cls):
        """
        :param args: the command-line arguments to be parsed by :class:`argparse.ArgumentParser`.

        Trains a model for this component.
        """
        raise NotImplementedError(
            '%s.%s()' %
            (cls.__class__.__name__, inspect.stack()[0][3]))

    @classmethod
    @abc.abstractmethod
    def decode(cls):
        """
        :param args: the command-line arguments to be parsed by :class:`argparse.ArgumentParser`.

        Predicts labels using this component.
        """
        raise NotImplementedError(
            '%s.%s()' %
            (cls.__class__.__name__, inspect.stack()[0][3]))

    @classmethod
    @abc.abstractmethod
    def evaluate(cls):
        """
        :param args: the command-line arguments to be parsed by :class:`argparse.ArgumentParser`.

        Evaluates the current model of this component.
        """
        raise NotImplementedError(
            '%s.%s()' %
            (cls.__class__.__name__, inspect.stack()[0][3]))


class ELITCLI(object):
    def __init__(self):
        parser = argparse.ArgumentParser(
            usage='''
    elit <command> [<args>]

commands:
    token_tagger    token_tagger
'''
        )
        parser.add_argument('command', help='command to run')
        args = parser.parse_args(sys.argv[1:2])
        if args.command == 'token_tagger':
            from elit.token_tagger import TokenTaggerCLI
            TokenTaggerCLI()
        else:
            print('Unrecognized command')
            parser.print_help()
            exit(1)


def set_logger(filename: str = None,
               level: int = logging.INFO,
               formatter: logging.Formatter = None):
    log = logging.getLogger()
    log.setLevel(level)
    ch = logging.StreamHandler(
        sys.stdout) if filename is None else logging.FileHandler(filename)
    if formatter is not None:
        ch.setFormatter(formatter)
    log.addHandler(ch)


def args_dict_str(s: str, const: Callable[[str], Any]) -> Dict[str, Any]:
    """
    :param s: key:int(,key:int)*
    :param const: type constructor (e.g., int, float)
    :return: a dictionary including key-value pairs from the input string.
    """
    r = re.compile(r'([A-Za-z0-9_-]+):([-]?[\d.]+)')
    d = {}
    for i in re.findall(r, s):
        d[i[0]] = const(i[1])

    if not d:
        raise argparse.ArgumentTypeError
    return d


def args_dict_str_float(s: str) -> Dict[str, float]:
    return args_dict_str(s, float)


def args_tuple_int(a: list) -> Tuple[int, ...]:
    """

    :param a: list of int
    :return: tuple of int
    """
    return tuple(a)

# def args_tuple_int(s: str) -> Tuple[int, ...]:
#     """
#     :param s: int(, int)*
#     :return: tuple of integers
#     """
#     r = re.compile(r'-?\d+(,-?\d+)*$')
#     if r.match(s):
#         return tuple(map(int, s.split(',')))
#     else:
#         raise argparse.ArgumentTypeError


def args_reader(s: str) -> SimpleNamespace:
    """
    :param s: json or tsv=(str:int)(,#1)*
    :return: the output of namespace_reader().
    """
    if s == 'json':
        return namespace_reader(reader_type=json_reader)
    if s.startswith('tsv='):
        return namespace_reader(reader_type=tsv_reader,
                                params=args_dict_str(s[4:], int))
    raise argparse.ArgumentTypeError


def args_vsm(s: list) -> SimpleNamespace:
    """
    :param s: fasttext|word2vec key filepath
    :return: the output of namespace_vsm().
    """

    # if len(s) != s:
    #     raise argparse.ArgumentTypeError()

    if s[0] == 'fasttext':
        vsm = FastText
    elif s[0] == 'word2vec':
        vsm = Word2Vec
    else:
        raise argparse.ArgumentTypeError(
            "Unsupported vector space model: " + s[0])

    key = s[1]
    filepath = s[2]

    return namespace_vsm(vsm_type=vsm, key=key, filepath=filepath)

def args_ngram_conv(s: list) -> Optional[SimpleNamespace]:
    """
    :param s: [ngram, filters, activation, pool, dropout]
    :return: the output of namespace_conf2d().
    """
    ngrams = args_tuple_int(s[0].split(':'))
    filters = int(s[1])
    activation = s[2]
    pool = s[3]
    dropout = float(s[4])
    return NLPModel.namespace_ngram_conv_layer(ngrams=ngrams,
                                               filters=filters,
                                               activation=activation,
                                               pool=pool,
                                               dropout=dropout)


def args_fuse_conv(s: str) -> Optional[SimpleNamespace]:
    """
    :param s: filters:activation:dropout
    :return: the output of namespace_conf2d().
    """
    return NLPModel.namespace_fuse_conv_layer(filters=int(s[0]),
                                              activation=s[1],
                                              dropout=float(s[2]))


def args_hidden_configs(s: str) -> Optional[SimpleNamespace]:
    """
    :param s: dim:activation:dropout
    :return: the output of namespace_hidden()
    """
    hidden_config = s.split(':')
    dim = int(hidden_config[0])
    activation = hidden_config[1]
    dropout = float(hidden_config[2])
    return SimpleNamespace(dim=dim, activation=activation, dropout=dropout)


def args_context(s: list) -> mx.Context:
    """
    :param s: ['ctx', core]
    :return: a device context
    """
    if s[0] == 'c':
        return mx.cpu(int(s[1]))
    elif s[0] == 'g':
        return mx.gpu(int(s[1]))
    else:
        raise argparse.ArgumentTypeError


def mx_loss(s: str) -> mx.gluon.loss.Loss:
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


def namespace_reader(type: Union[json_reader, tsv_reader],
                     params: Optional[Dict] = None) -> SimpleNamespace:
    return SimpleNamespace(type=type, params=params)


def namespace_vsm(l: list) -> SimpleNamespace:
    type = Word2Vec if l[0].lower() == 'word2vec' else FastText
    key = l[1]
    filepath = l[2]
    model = type(filepath)
    return SimpleNamespace(model=model, key=key)


if __name__ == '__main__':
    ELITCLI()
