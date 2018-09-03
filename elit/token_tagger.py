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
import json
import logging
import pickle
import sys
from types import SimpleNamespace
from typing import Tuple, Optional, Union, Sequence

import mxnet as mx
import numpy as np
from mxnet.ndarray import NDArray
from pkg_resources import resource_filename

from elit.cli import ComponentCLI, set_logger
from elit.component import MXNetComponent
from elit.eval import Accuracy, F1
from elit.model import FFNNModel
from elit.state import NLPState
from elit.util.io import pkl, gln, group_states, json_reader, tsv_reader
from elit.util.iterator import SequenceIterator, BatchIterator
from elit.util.structure import Document, to_gold, BILOU
from elit.util.vsm import LabelMap, x_extract, Position2Vec

__author__ = 'Jinho D. Choi'


# ======================================== State =========================

class TokenTaggerState(NLPState):
    """
    :class:`TokenTaggingState` generates a state per token using the one-pass left-to-right decoding strategy
    and predicts tags for individual tokens.
    """

    def __init__(self,
                 document: Document,
                 key: str,
                 label_map: LabelMap,
                 vsm_list: Sequence[SimpleNamespace],
                 feature_windows: Tuple[int, ...],
                 padout: np.ndarray = None):
        """
        :param document: the input document.
        :param key: the key to each sentence in the document where predicted tags are to be saved.
        :param label_map: collects labels during training and maps them to unique class IDs.
        :param vsm_list: the sequence of namespace(:class:`elit.vsm.VectorSpaceModel`, key).
        :param feature_windows: the contextual windows for feature extraction.
        :param padout: the zero-vector whose dimension is the number of classes; if not ``None``, label embeddings are used as features.
        """
        super().__init__(document, key)
        self.label_map = label_map
        self.feature_windows = feature_windows
        self.padout = padout

        # initialize gold-standard labels if available
        key_gold = to_gold(key)
        self.gold = [s[key_gold] for s in document] if key_gold in document.sentences[0] else None

        # initialize output and predicted tags
        self.output = []
        self.pred = []

        # initialize embeddings
        self.embs = [(vsm.model.sentence_embedding_list(document, vsm.key), vsm.model.pad) for vsm in vsm_list]
        if padout is not None:
            self.embs.append((self.output, padout))

        # the followings are initialized in self.init()
        self.sen_id = 0
        self.tok_id = 0
        self.init()

    def init(self):
        del self.output[:]
        del self.pred[:]
        self.sen_id = 0
        self.tok_id = 0

        for s in self.document:
            self.output.append([self.padout] * len(s))
            self.pred.append([None] * len(s))
            s[self.key] = self.pred[-1]

    def process(self, output: Optional[NDArray] = None):
        """
        :param output: the predicted output of the current token.

        Assigns the predicted output to the current token, then processes to the next token.
        """
        # apply the output to the current token
        if output is not None:
            self.output[self.sen_id][self.tok_id] = output.asnumpy()
            self.pred[self.sen_id][self.tok_id] = self.label_map.argmax(output)

        # process to the next token
        self.tok_id += 1
        if self.tok_id == len(self.document.sentences[self.sen_id]):
            self.sen_id += 1
            self.tok_id = 0

    def has_next(self) -> bool:
        """
        :return: ``False`` if no more token is left to be tagged; otherwise, ``True``.
        """
        return 0 <= self.sen_id < len(self.document)

    @property
    def x(self) -> np.ndarray:
        """
        :return: the feature matrix of the current token.
        """
        l = ([x_extract(self.tok_id, w, emb[self.sen_id], pad) for w in self.feature_windows] for emb, pad in self.embs)
        return np.column_stack(l)

    @property
    def y(self) -> Optional[int]:
        """
        :return: the class ID of the current token's gold-standard label if available; otherwise, None.
        """
        return self.label_map.add(self.gold[self.sen_id][self.tok_id]) if self.gold is not None else None


# ======================================== EvalMetric ====================

class TokenAccuracy(Accuracy):
    def __init__(self, key: str):
        super().__init__()
        self.key = key

    def update(self, document: Document):
        for sentence in document:
            gold = sentence[to_gold(self.key)]
            pred = sentence[self.key]
            self.correct += len([1 for g, p in zip(gold, pred) if g == p])
            self.total += len(sentence)


class ChunkF1(F1):
    def __init__(self, key: str):
        super().__init__()
        self.key = key

    def update(self, document: Document):
        for sentence in document:
            gold = BILOU.to_chunks(sentence[to_gold(self.key)])
            pred = BILOU.to_chunks(sentence[self.key])
            self.correct += len(set.intersection(set(gold), set(pred)))
            self.p_total += len(pred)
            self.r_total += len(gold)


# ======================================== Component =====================

class TokenTagger(MXNetComponent):
    """
    :class:`TokenTagger` provides an abstract class to implement a tagger that predicts a tag for every token.
    """

    def __init__(self,
                 ctx: mx.Context,
                 vsm_path: list):
        """
        :param ctx: the (list of) device context(s) for :class:`mxnet.gluon.Block`.
        :param vsm_list: the sequence of namespace(model, key),
                         where the key indicates the key of the values to retrieve embeddings for (e.g., tok).
        """
        super().__init__(ctx)
        self.vsm_list = [self.namespace_vsm(n) for n in vsm_path]

        # to be loaded/saved
        self.key = None
        self.sequence = None
        self.chunking = None
        self.label_map = None
        self.feature_windows = None
        self.position_embedding = None
        self.padout = None
        self.input_config = None
        self.output_config = None
        self.fuse_conv_config = None
        self.ngram_conv_config = None
        self.hidden_configs = None
        self.initializer = None

    def __str__(self):
        s = ('Token Tagger',
             '- type: %s' % ('chunking' if self.chunking else 'tagging'),
             '- sequence: %r' % self.sequence,
             '- feature windows: %s' % str(self.feature_windows),
             '- label embedding: %r' % (self.padout is not None),
             '- position embedding: %r' % self.position_embedding,
             '- %s' % str(self.model))
        return '\n'.join(s)

    # override
    def init(self,
             key: str,
             sequence: bool,
             chunking: bool,
             num_class: int,
             feature_windows: Tuple[int, ...],
             position_embedding: bool,
             label_embedding: bool,
             input_config: Optional[SimpleNamespace] = SimpleNamespace(dropout=0.0),
             output_config: Optional[SimpleNamespace] = None,
             fuse_conv_config: Optional[SimpleNamespace] = None,
             ngram_conv_config: Optional[SimpleNamespace] = None,
             hidden_configs: Optional[Tuple[SimpleNamespace]] = None,
             initializer: mx.init.Initializer = mx.init.Xavier(magnitude=2.24, rnd_type='gaussian'),
             **kwargs):
        """
        :param output_config:
        :param key: the key to each sentence where the tags are to be saved.
        :param sequence: if ``True``, run in sequence mode; otherwise, batch mode.
        :param chunking: if ``True``, run chunking instead of tagging.
        :param num_class: the number of classes (part-of-speech tags).
        :param feature_windows: the content windows for feature extraction.
        :param position_embedding: if ``True``, use position embeddings as features.
        :param label_embedding: if ``True``, use label embeddings as features.
        :param input_config: the dropout rate to be applied to the input layer.
        :param fuse_conv_config: the configuration for the fuse convolution layer.
        :param ngram_conv_config: the configuration for the n-gram convolution layer.
        :param hidden_configs: the configurations for the hidden layers.
        :param initializer: the weight initializer for :class:`mxnet.gluon.Block`.
        :param kwargs: extra parameters to initialize :class:`mxnet.gluon.Block`.
        """
        # configuration
        if position_embedding:
            self.vsm_list.append(SimpleNamespace(model=Position2Vec(), key=None))

        input_dim = sum([vsm.model.dim for vsm in self.vsm_list])
        if label_embedding:
            input_dim += num_class
        input_config.col = input_dim
        input_config.row = len(feature_windows)
        output_config.dim = num_class
        self.initializer = initializer

        # initialization
        self.key = key
        self.sequence = sequence
        self.chunking = chunking
        self.label_map = LabelMap()
        self.feature_windows = feature_windows
        self.position_embedding = position_embedding
        self.padout = np.zeros(num_class).astype('float32') if label_embedding else None
        self.input_config = input_config
        self.output_config = output_config
        self.fuse_conv_config = fuse_conv_config
        self.ngram_conv_config = ngram_conv_config
        self.hidden_configs = hidden_configs

        self.model = FFNNModel(
            input_config=self.input_config,
            output_config=self.output_config,
            fuse_conv_config=self.fuse_conv_config,
            ngram_conv_config=self.ngram_conv_config,
            hidden_configs=self.hidden_configs,
            **kwargs)
        self.model.collect_params().initialize(self.initializer, ctx=self.ctx)
        logging.info(self.__str__())

    # override
    def load(self, model_path: str, **kwargs):
        """
        :param model_path: the path to a pre-trained model to be loaded.
        :type model_path: str
        """
        with open(pkl(model_path), 'rb') as fin:
            self.key = pickle.load(fin)
            self.sequence = pickle.load(fin)
            self.chunking = pickle.load(fin)
            self.label_map = pickle.load(fin)
            self.feature_windows = pickle.load(fin)
            self.position_embedding = pickle.load(fin)
            self.padout = pickle.load(fin)
            self.input_config = pickle.load(fin)
            self.output_config = pickle.load(fin)
            self.fuse_conv_config = pickle.load(fin)
            self.ngram_conv_config = pickle.load(fin)
            self.hidden_configs = pickle.load(fin)

        self.model = FFNNModel(
            input_config=self.input_config,
            output_config=self.output_config,
            fuse_conv_config=self.fuse_conv_config,
            ngram_conv_config=self.ngram_conv_config,
            hidden_configs=self.hidden_configs,
            **kwargs)
        self.model.collect_params().load(gln(model_path), self.ctx)
        if self.position_embedding:
            self.vsm_list.append(SimpleNamespace(model=Position2Vec(), key=None))
        logging.info(self.__str__())

    # override
    def save(self, model_path, **kwargs):
        """
        :param model_path: the filepath where the model is to be saved.
        :type model_path: str
        """
        with open(pkl(model_path), 'wb') as fout:
            pickle.dump(self.key, fout)
            pickle.dump(self.sequence, fout)
            pickle.dump(self.chunking, fout)
            pickle.dump(self.label_map, fout)
            pickle.dump(self.feature_windows, fout)
            pickle.dump(self.position_embedding, fout)
            pickle.dump(self.padout, fout)
            pickle.dump(self.input_config, fout)
            pickle.dump(self.output_config, fout)
            pickle.dump(self.fuse_conv_config, fout)
            pickle.dump(self.ngram_conv_config, fout)
            pickle.dump(self.hidden_configs, fout)

        self.model.collect_params().save(gln(model_path))

    # override
    def data_iterator(self, documents: Sequence[Document], batch_size: int, shuffle: bool,
                      label: bool, **kwargs) -> Union[BatchIterator, SequenceIterator]:
        if self.sequence:
            def create(d: Document) -> TokenTaggerState:
                return TokenTaggerState(d, self.key, self.label_map, self.vsm_list,
                                        self.feature_windows, self.padout)

            states = group_states(documents, create)
        else:
            states = [TokenTaggerState(d, self.key, self.label_map, self.vsm_list, self.feature_windows) for d in documents]

        iterator = SequenceIterator if self.sequence else BatchIterator
        return iterator(states, batch_size, shuffle, label, **kwargs)

    # override
    def eval_metric(self) -> Union[TokenAccuracy, ChunkF1]:
        """
        :return: :class:`ChunkF1` if self.chunking else :class:`TokenAccuracy`.
        """
        return ChunkF1(self.key) if self.chunking else TokenAccuracy(self.key)


class TokenTaggerCLI(ComponentCLI):
    def __init__(self):
        super().__init__('token_tagger', 'Token Tagger')

    # override
    @classmethod
    def train(cls):
        # create a arg-parser
        parser = argparse.ArgumentParser(
            description='Train a token tagger',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # data
        data_group = parser.add_argument_group("data arguments")

        data_group.add_argument(
            'trn_path',
            type=str,
            metavar='TRN_PATH',
            help='filepath to the training data (input)')
        data_group.add_argument(
            'dev_path',
            type=str,
            metavar='DEV_PATH',
            help='filepath to the development data (input)')
        data_group.add_argument(
            'model_path',
            type=str,
            metavar='MODEL_PATH',
            default=None,
            help='filepath to the model data (output); if not set, the model is not saved')
        data_group.add_argument(
            '--vsm_path',
            action='append',
            nargs='+',
            metavar='VSM_PATH',
            required=True,
            help='vsm path'
        )

        # tagger
        tagger_group = parser.add_argument_group("tagger arguments")

        tagger_group.add_argument(
            'key',
            type=str,
            metavar='KEY',
            help='key to the document dictionary where the predicted tags are to be stored')

        # network
        network_group = parser.add_argument_group("network arguments")

        network_group.add_argument(
            '-c',
            '--config',
            type=str,
            metavar='CONFIG',
            help="config file"
        )

        # arguments
        args = parser.parse_args(sys.argv[3:])
        json_config = args.config if args.config else resource_filename('elit.resources.token_tagger', 'pos.json')
        with open(json_config, 'r') as d:
            config = json.load(d, object_hook=lambda d: SimpleNamespace(**d))

        set_logger(config.data.log_path)

        reader = tsv_reader if config.data.reader == 'tsv' else json_reader

        trn_docs, trn_num_class = reader(args.trn_path, config.data.tsv_heads.__dict__, args.key)
        dev_docs, dev_num_class = reader(args.dev_path, config.data.tsv_heads.__dict__, args.key)


        ctx = mx.gpu(config.training.core) if config.training.device == 'gpu' else mx.cpu(config.training.core)

        # component
        comp = TokenTagger(ctx, args.vsm_path)
        comp.init(
            key=args.key,
            sequence=config.tagger.sqeuence,
            chunking=config.tagger.chunking,
            num_class=trn_num_class,
            feature_windows=config.tagger.feature_windows,
            position_embedding=config.tagger.position_embedding,
            label_embedding=config.tagger.label_embedding,
            input_config=config.network.input,
            output_config=config.network.output,
            fuse_conv_config=config.network.fuse_conv,
            ngram_conv_config=config.network.ngrams_conv,
            hidden_configs=config.network.hiddens,
        )

        print(comp)

        # train
        # comp.train(
        #     trn_docs,
        #     dev_docs,
        #     args.model_path,
        #     config.trn_batch,
        #     config.dev_batch,
        #     config.epoch,
        #     config.loss,
        #     config.optimizer,
        #     config.optimizer_params)
        # logging.info('# of classes: %d' % len(comp.label_map))

    # override
    @classmethod
    def decode(cls):
        pass
    #     # create a arg-parser
    #     parser = argparse.ArgumentParser(
    #         description='Decode with the token tagger',
    #         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #     args = parser.parse_args(sys.argv[3:])
    #
    #     # data
    #     group = parser.add_argument_group("data arguments")
    #
    #     group.add_argument('-i', '--input_path', type=str, metavar='filepath',
    #                        help='filepath to the input data')
    #     group.add_argument(
    #         '-o',
    #         '--output_path',
    #         type=str,
    #         metavar='filepath',
    #         default=None,
    #         help='filepath to the output data (default: input_path.json)')
    #     group.add_argument('-m', '--model_path', type=str, metavar='filepath',
    #                        default=None,
    #                        help='filepath to the model data')
    #     group.add_argument(
    #         '-r',
    #         '--reader',
    #         type=args_reader,
    #         metavar='json or tsv=(str:int)(,#1)*',
    #         default=args_reader('json'),
    #         help='type of the reader and its configuration to match the data format (default: json)')
    #     group.add_argument(
    #         '-v',
    #         '--vsm_list',
    #         type=args_vsm,
    #         metavar='(fasttext|word2vec:key:filepath)( #1)*',
    #         nargs='+',
    #         help='list of (type of vector space model, key, filepath)')
    #     group.add_argument(
    #         '-l',
    #         '--log',
    #         type=str,
    #         metavar='filepath',
    #         default=None,
    #         help='filepath to the logging file; if not set, use stdout')
    #
    #     # evaluation
    #     group = parser.add_argument_group("arguments for decoding")
    #
    #     group.add_argument(
    #         '-cx',
    #         '--context',
    #         type=args_context,
    #         metavar='[cg]int',
    #         nargs='+',
    #         default=mx.cpu(),
    #         help='device context(s)')
    #     group.add_argument(
    #         '-ib',
    #         '--input_batch',
    #         type=int,
    #         metavar='int',
    #         default=2048,
    #         help='batch size for the input data')
    #
    #     # arguments
    #     args = parser.parse_args(args)
    #     set_logger(args.log)
    #
    #     args.vsm_list = [
    #         SimpleNamespace(
    #             model=n.type(
    #                 n.filepath),
    #             key=n.key) for n in args.vsm_list]
    #     if isinstance(args.context, list) and len(args.context) == 1:
    #         args.context = args.context[0]
    #     if args.output_path is None:
    #         args.output_path = args.input_path + '.json'
    #
    #     # component
    #     comp = TokenTagger(args.context, args.vsm_list)
    #     comp.load(args.model_path)
    #
    #     # data
    #     docs = args.reader.type(args.input_path, args.reader.params)
    #
    #     # evaluate
    #     comp.decode(docs, args.input_batch)
    #
    #     with open(args.output_path, 'w') as fout:
    #         json.dump(docs, fout)

    # override
    @classmethod
    def evaluate(cls):
        pass
    #     # create a arg-parser
    #     parser = argparse.ArgumentParser(
    #         description='Evaluate the token tagger',
    #         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #     args = parser.parse_args(sys.argv[3:])
    #
    #     # data
    #     group = parser.add_argument_group("data arguments")
    #
    #     group.add_argument('-d', '--dev_path', type=str, metavar='filepath',
    #                        help='filepath to the development data (input)')
    #     group.add_argument('-m', '--model_path', type=str, metavar='filepath',
    #                        default=None,
    #                        help='filepath to the model data')
    #     group.add_argument(
    #         '-r',
    #         '--reader',
    #         type=args_reader,
    #         metavar='json or tsv=(str:int)(,#1)*',
    #         default=args_reader('json'),
    #         help='type of the reader and its configuration to match the data format (default: json)')
    #     group.add_argument(
    #         '-v',
    #         '--vsm_list',
    #         type=args_vsm,
    #         metavar='(fasttext|word2vec:key:filepath)( #1)*',
    #         nargs='+',
    #         help='list of (type of vector space model, key, filepath)')
    #     group.add_argument(
    #         '-l',
    #         '--log',
    #         type=str,
    #         metavar='filepath',
    #         default=None,
    #         help='filepath to the logging file; if not set, use stdout')
    #
    #     # evaluation
    #     group = parser.add_argument_group("arguments for evaluation")
    #
    #     group.add_argument(
    #         '-cx',
    #         '--context',
    #         type=args_context,
    #         metavar='[cg]int',
    #         nargs='+',
    #         default=mx.cpu(),
    #         help='device context(s)')
    #     group.add_argument(
    #         '-db',
    #         '--dev_batch',
    #         type=int,
    #         metavar='int',
    #         default=2048,
    #         help='batch size for the development data')
    #
    #     # arguments
    #     args = parser.parse_args(args)
    #     set_logger(args.log)
    #
    #     args.vsm_list = [
    #         SimpleNamespace(
    #             model=n.type(
    #                 n.filepath),
    #             key=n.key) for n in args.vsm_list]
    #     if isinstance(args.context, list) and len(args.context) == 1:
    #         args.context = args.context[0]
    #
    #     # component
    #     comp = TokenTagger(args.context, args.vsm_list)
    #     comp.load(args.model_path)
    #
    #     # data
    #     dev_docs = args.reader.type(
    #         args.dev_path, args.reader.params, comp.key)
    #
    #     # evaluate
    #     comp.evaluate(dev_docs, args.dev_batch)
