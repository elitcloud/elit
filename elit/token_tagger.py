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
import logging
import pickle
import sys
from types import SimpleNamespace
from typing import Tuple, Optional, List, Union

import mxnet as mx
import numpy as np
from mxnet.ndarray import NDArray

from elit.component import MXNetComponent, SequenceComponent, BatchComponent
from elit.eval import Accuracy, F1
from elit.model import FFNNModel
from elit.state import BatchState, SequenceState, group_states, NLPState
from elit.structure import Document, TOK
from elit.util import BILOU, to_gold, to_out
from elit.utils.cli_util import args_reader, args_vsm, args_tuple_int, args_conv2d, args_hidden, args_loss, args_context, namespace_input, namespace_output
from elit.vsm import LabelMap, get_loc_embeddings, x_extract, X_ANY, FastText

__author__ = 'Jinho D. Choi'


# ======================================== State ========================================

class TokenTaggerState(NLPState):
    """
    TokenTaggingState defines the one-pass left-to-right strategy for tagging individual tokens.
    """

    def __init__(self,
                 document: Document,
                 key: str,
                 label_map: LabelMap,
                 vsm_list: Tuple[SimpleNamespace, ...],
                 windows: Tuple[int, ...]):
        """
        :param document: an input document.
        :param key: the key to each sentence in the document where predicted labels are to be saved.
        :param label_map: collects class labels during training and maps them to unique IDs.
        :param vsm_list: a tuple of namespace(model, key).
        :param windows: contextual windows of adjacent tokens for feature extraction.
        """
        super().__init__(document, key)
        self.label_map = label_map
        self.windows = windows

        # initialize gold-standard labels if available
        key_gold = to_gold(key)
        self.gold = [s[key_gold] for s in document] if key_gold in document.sentences[0] else None

        # initialize embeddings
        self.embs = [vsm.model.sentence_embedding_list(document, vsm.key) for vsm in vsm_list]
        self.embs.append(get_loc_embeddings(document))

        # self.init()
        self.sen_id = 0
        self.tok_id = 0

    def init(self):
        """
        Initializes the pointers to the first token in the first sentence.
        """
        self.sen_id = 0
        self.tok_id = 0

    def process(self, **kwargs):
        """
        Processes to the next token.
        """
        self.tok_id += 1
        if self.tok_id == len(self.document.sentences[self.sen_id]):
            self.sen_id += 1
            self.tok_id = 0

    @property
    def has_next(self) -> bool:
        """
        :return: False if no more token is left to be tagged; otherwise, True.
        """
        return 0 <= self.sen_id < len(self.document)

    @property
    def x(self) -> np.ndarray:
        """
        :return: the feature matrix of the current token.
        """
        l = ([x_extract(self.tok_id, w, emb[self.sen_id], pad) for w in self.windows] for emb, pad in self.embs)
        return np.column_stack(l)

    @property
    def y(self) -> Optional[int]:
        """
        :return: the class ID of the current token's gold-standard label if available; otherwise, None.
        """
        return self.label_map.add(self.gold[self.sen_id][self.tok_id]) if self.gold is not None else None


class TokenTaggerBatchState(TokenTaggerState, BatchState):
    """
    TokenTaggingBatchState defines the one-pass left-to-right strategy for tagging individual tokens in batch mode.
    """

    def __init__(self,
                 document: Document,
                 key: str,
                 label_map: LabelMap,
                 vsm_list: Tuple[SimpleNamespace, ...],
                 windows: Tuple[int, ...]):
        """
        :param document: an input document.
        :param key: the key to each sentence in the document where predicted labels are to be saved.
        :param label_map: collects class labels during training and maps them to unique IDs.
        :param vsm_list: a tuple of namespace(model, key).
        :param windows: contextual windows of adjacent tokens for feature extraction.
        """
        TokenTaggerState.__init__(self, document, key, label_map, vsm_list, windows)

    def assign(self, output: NDArray, begin: int = 0) -> int:
        """
        Assigns the predicted output to the each token in the input document.
        :param output: a matrix where each row contains prediction scores of the corresponding token.
        :param begin: the row index of the output matrix corresponding to the first token in the input document.
        :return: the row index of the output matrix to be assigned by the next document.
        """
        key_out = to_out(self.key)
        for sentence in self.document:
            end = begin + len(sentence)
            sentence[key_out] = output[begin:end].asnumpy()
            begin = end
        return begin


class TokenTaggerSequenceState(TokenTaggerState, SequenceState):
    """
    TokenTaggingSequenceState defines the one-pass left-to-right strategy for tagging individual tokens in sequence mode.
    In other words, predicted outputs from earlier tokens are used as features to predict later tokens.
    """

    def __init__(self,
                 document: Document,
                 key: str,
                 label_map: LabelMap,
                 vsm_list: Tuple[SimpleNamespace, ...],
                 windows: Tuple[int, ...],
                 padout: np.ndarray):
        """
        :param document: an input document.
        :param key: the key to each sentence in the document where predicted labels are to be saved.
        :param label_map: collects class labels during training and maps them to unique IDs.
        :param vsm_list: a list of namespace(model, key).
        :param windows: contextual windows of adjacent tokens for feature extraction.
        :param padout: a zero-vector whose dimension is the number of class labels, used to zero-pad label embeddings.
        """
        TokenTaggerState.__init__(self, document, key, label_map, vsm_list, windows)
        self.padout = padout
        self.output = [[self.padout] * len(s) for s in self.document]

        # initialize embeddings
        if padout is not None: self.embs.append((self.output, self.padout))

    def init(self):
        """
        Initializes the pointers to the first otken in the first sentence and the predicted outputs and labels.
        """
        TokenTaggerState.init(self)

        for i, s in enumerate(self.document):
            self.output[i] = [self.padout] * len(s)

    def process(self, output: np.ndarray):
        """
        Assigns the predicted output to the current token, then processes to the next token.
        :param output: the predicted output of the current token.
        """
        # apply the output to the current token
        self.output[self.sen_id][self.tok_id] = output

        # process to the next token
        TokenTaggerState.process(self)


# ======================================== EvalMetric ========================================

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


# ======================================== Component ========================================

class TokenTagger(MXNetComponent):
    """
    TokenBatchTagger provides an abstract class to implement a tagger that predicts a tag for every token.
    """

    def __init__(self,
                 ctx: mx.Context,
                 vsm_list: Tuple[SimpleNamespace, ...]):
        """
        :param ctx: a device context.
        :param vsm_list: a tuple of namespace(model, key), where the key indicates the key to each sentence to retrieve embeddings for (e.g., tok).
        """
        super().__init__(ctx)
        self.vsm_list = vsm_list

        # to be loaded/saved
        self.key = None
        self.sequence = None
        self.chunking = None
        self.label_map = None
        self.feature_windows = None
        self.label_embedding = None   # sequence mode only
        self.padout = None            # sequence mode only

    @abc.abstractmethod
    def create_states(self, documents: List[Document]) -> List[Union[TokenTaggerBatchState, TokenTaggerSequenceState]]:
        """
        :param documents: a list of input documents.
        :return: the list of sequence or batch states corresponding to the input documents.
        """
        pass

    # override
    def finalize(self, state: Union[TokenTaggerBatchState, TokenTaggerSequenceState]):
        """
        Finalizes by saving the predicted tags to the input document once decoding is done.
        If self.chunking, it has a list of chunks instead.
        :param state: an input state.
        """
        key_out = to_out(self.key)
        for i, sentence in enumerate(state.document):
            if self.sequence: sentence[key_out] = state.output[i]
            sentence[self.key] = [self.label_map.argmax(o) for o in sentence[key_out]]

    # override
    def eval_metric(self) -> Union[TokenAccuracy, ChunkF1]:
        """
        :return: :class:`ChunkF1` if self.chunking else :class:`TokenAccuracy`.
        """
        return ChunkF1(self.key) if self.chunking else TokenAccuracy(self.key)

    # override
    def init(self,
             sequence: bool,
             chunking: bool,
             key: str,
             num_class: int,
             feature_windows: Tuple[int, ...],
             label_embedding: bool = False,
             input_dropout: float = 0.0,
             conv2d_config: Optional[Tuple[SimpleNamespace, ...]] = None,
             hidden_config: Optional[Tuple[SimpleNamespace, ...]] = None,
             initializer: mx.init.Initializer = None,
             **kwargs):
        """
        :param sequence: if True, this tagging is run in sequence mode; otherwise, batch mode.
        :param chunking: if True, this tagging is used for chunking instead.
        :param key: the key to each sentence where the tags are to be saved.
        :param feature_windows: content windows for feature extraction.
        :param label_embedding: if True, use label embeddings as features.
        :param num_class: the number of classes (part-of-speech tags).
        :param input_dropout: a dropout rate to be applied to the input layer.
        :param conv2d_config: configuration for n-gram 2D convolutions.
        :param hidden_config: configuration for hidden layers
        :param initializer: a weight initializer for the gluon block.
        :param kwargs: parameters for the initialization of gluon.Block.
        """
        # configuration
        if initializer is None: initializer = mx.init.Xavier(magnitude=2.24, rnd_type='gaussian')
        input_dim = sum([vsm.model.dim for vsm in self.vsm_list]) + len(X_ANY)
        if label_embedding: input_dim += num_class
        input_config = namespace_input(input_dim, row=len(feature_windows), dropout=input_dropout)
        output_config = namespace_output(num_class)

        # initialization
        self.sequence = sequence
        self.chunking = chunking
        self.key = key
        self.label_map = LabelMap()
        self.feature_windows = feature_windows
        self.label_embedding = label_embedding
        self.padout = np.zeros(output_config.dim).astype('float32')
        self.model = FFNNModel(input_config, output_config, conv2d_config, hidden_config, **kwargs)
        self.model.collect_params().initialize(initializer, ctx=self.ctx)
        print(self.__str__())

    # override
    def load(self, model_path: str, **kwargs):
        """
        :param model_path: the path to a pre-trained model to be loaded.
        :type model_path: str
        """
        with open(model_path, 'rb') as fin:
            self.sequence = pickle.load(fin)
            self.chunking = pickle.load(fin)
            self.key = pickle.load(fin)
            self.label_map = pickle.load(fin)
            self.feature_windows = pickle.load(fin)
            self.label_embedding = pickle.load(fin)
            self.padout = pickle.load(fin)
            self.model = pickle.load(fin)
        print(self.__str__())

    # override
    def save(self, model_path, **kwargs):
        """
        :param model_path: the filepath where the model is to be saved.
        :type model_path: str
        """
        with open(model_path, 'wb') as fout:
            pickle.dump(self.sequence, fout)
            pickle.dump(self.chunking, fout)
            pickle.dump(self.key, fout)
            pickle.dump(self.label_map, fout)
            pickle.dump(self.feature_windows, fout)
            pickle.dump(self.label_embedding, fout)
            pickle.dump(self.padout, fout)
            pickle.dump(self.model, fout)


class TokenBatchTagger(TokenTagger, BatchComponent):
    """
    TokenBatchTagger implements a tagger that predicts a tag for every token in batch mode.
    """

    def __init__(self,
                 ctx: mx.Context,
                 vsm_list: Tuple[SimpleNamespace, ...]):
        """
        :param ctx: a device context.
        :param vsm_list: a tuple of namespace(model, key), where the key indicates the key to each sentence to retrieve embeddings for (e.g., tok).
        """
        TokenTagger.__init__(self, ctx, vsm_list)

    def __str__(self):
        s = ('TokenBatchTagger',
             '- type: %s' % ('chunking' if self.chunking else 'tagging'),
             '- feature windows: %s' % str(self.feature_windows),
             '- model: %s' % str(self.model))
        return '\n'.join(s)

    # override
    def create_states(self, documents: List[Document]) -> List[TokenTaggerBatchState]:
        """
        :param documents: a list of input documents.
        :return: the list of sequence or batch states corresponding to the input documents.
        """
        return [TokenTaggerBatchState(d, self.key, self.label_map, self.vsm_list, self.feature_windows) for d in documents]


class TokenSequenceTagger(TokenTagger, SequenceComponent):
    """
    TokenSequenceTagger implements a tagger that predicts a tag for every token in sequence mode.
    """

    def __init__(self,
                 ctx: mx.Context,
                 vsm_list: Tuple[SimpleNamespace, ...]):
        """
        :param ctx: a device context.
        :param vsm_list: a tuple of namespace(model, key), where the key indicates the key to each sentence to retrieve embeddings for (e.g., tok).
        """
        TokenTagger.__init__(self, ctx, vsm_list)

    def __str__(self):
        s = ('TokenSequenceTagger',
             '- type: %s' % ('chunking' if self.chunking else 'tagging'),
             '- feature windows: %s' % str(self.feature_windows),
             '- label embedding: %r' % self.label_embedding,
             '- model: %s' % str(self.model))
        return '\n'.join(s)

    # override
    def create_states(self, documents: List[Document]) -> List[TokenTaggerSequenceState]:
        """
        :param documents: a list of input documents.
        :return: the list of sequence or batch states corresponding to the input documents.
        """
        def create(d: Document) -> TokenTaggerSequenceState:
            padout = self.padout if self.label_embedding else None
            return TokenTaggerSequenceState(d, self.key, self.label_map, self.vsm_list, self.feature_windows, padout)
        return group_states(documents, create)


# ======================================== Command-Line Interface ========================================

class TokenTaggerCLI:
    def __init__(self):
        usage = '''
    elit tagger <command> [<args>]

commands:
    train  train a model (gold labels required)
    eval   evaluate a pre-trained model (gold labels required) 
'''
        parser = argparse.ArgumentParser(usage=usage, description='Token Tagger')
        parser.add_argument('command', help='command to run')
        args = parser.parse_args(sys.argv[2:3])
        if not hasattr(self, args.command):
            logging.info('Unrecognized command: ' + args.command)
            parser.print_help()
            exit(1)
        getattr(self, args.command)()

    @classmethod
    def train(cls):
        # create a arg-parser
        parser = argparse.ArgumentParser(description='Train a token tagger',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        # tagger
        group = parser.add_argument_group("tagger arguments")

        group.add_argument('-key', '--key', type=str, metavar='str',
                           help='key to the document dictionary where the predicted tags are to be stored')
        group.add_argument('-seq', '--sequence', action='store_true',
                           help='if set, run in sequence mode; otherwise, batch mode')
        group.add_argument('-chu', '--chunking', action='store_true',
                           help='if set, tag chunks (e.g., named entities); otherwise, tag tokens (e.g., part-of-speech tags)')

        # data
        group = parser.add_argument_group("data arguments")

        group.add_argument('-t', '--trn_path', type=str, metavar='filepath',
                           help='filepath to the training data (input)')
        group.add_argument('-d', '--dev_path', type=str, metavar='filepath',
                           help='filepath to the development data (input)')
        group.add_argument('-m', '--model_path', type=str, metavar='filepath',
                           default=None,
                           help='filepath to the model data (output)')
        group.add_argument('-r', '--reader', type=args_reader, metavar='json or tsv=(str:int)(,#1)*',
                           default=args_reader('json'),
                           help='type of the reader and its configuration to match the data format (default: json)')

        # lexicon
        group = parser.add_argument_group("lexicon arguments")

        group.add_argument('-v', '--vsm_list', type=args_vsm, metavar='(fasttext|word2vec:key:filepath)( #1)*', nargs='+',
                           help='list of (type of vector space model, key, filepath)')

        # feature
        group = parser.add_argument_group("feature arguments")

        group.add_argument('-fw', '--feature_windows', type=args_tuple_int, metavar='int(,int)*',
                           default=args_tuple_int('3,2,1,0,-1,-2,-3'),
                           help='contextual windows for feature extraction (default: 3,2,1,0,-1,-2,-3)')

        group.add_argument('-le', '--label_embedding', action='store_true',
                           help='if set, use label embeddings as features in sequence mode')

        # network
        group = parser.add_argument_group("network arguments")

        group.add_argument('-nc', '--num_class', type=int, metavar='int',
                           help='number of classes')
        group.add_argument('-ir', '--input_dropout', type=float, metavar='float',
                           default=0.0,
                           help='dropout rate applied to the input layer (default: 0.0)')
        group.add_argument('-cc', '--conv2d_config', type=args_conv2d, metavar='(ngram:filters:activation:pool:dropout)( #1)*', nargs='+',
                           default=tuple(args_conv2d('%d:128:relu:none:0.2' % i) for i in range(1, 6)),
                           help='configuration for the convolution layer (default: %d:128:relu:none:0.2, where %d = [1..5])')
        group.add_argument('-hc', '--hidden_config', type=args_hidden, metavar='(dim:activation:dropout)( #1)*', nargs='+',
                           default=None,
                           help='configuration for the hidden layer (default: None)')

        # training
        group = parser.add_argument_group("training arguments")

        group.add_argument('-cx', '--context', type=args_context, metavar='[cg]int', nargs='+',
                           default=mx.cpu(),
                           help='device context(s)')
        group.add_argument('-ep', '--epoch', type=int, metavar='int', default=100,
                           help='number of epochs')
        group.add_argument('-tb', '--trn_batch', type=int, metavar='int', default=64,
                           help='batch size for the training data')
        group.add_argument('-db', '--dev_batch', type=int, metavar='int', default=2048,
                           help='batch size for the development data')
        group.add_argument('-lo', '--loss', type=args_loss, metavar='str', default=None,
                           help='loss function')
        group.add_argument('-op', '--optimizer', type=str, metavar='str', default='adagrad',
                           help='optimizer')
        group.add_argument('-lr', '--learning_rate', type=float, metavar='float', default=0.01,
                           help='learning rate')
        group.add_argument('-wd', '--weight_decay', type=float, metavar='float', default=0.0,
                           help='weight decay')

        # arguments
        args = parser.parse_args(sys.argv[3:])
        args.vsm_list = tuple(SimpleNamespace(model=n.type(n.filepath), key=n.key) for n in args.vsm_list)
        if isinstance(args.context, list) and len(args.context) == 1: args.context = args.context[0]
        if isinstance(args.conv2d_config, list) and args.conv2d_config[0] is None: args.conv2d_config = None
        if args.loss is None: args.loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()
        initializer = mx.init.Xavier(magnitude=2.24, rnd_type='gaussian')

        # component
        factory = TokenSequenceTagger if args.sequence else TokenBatchTagger
        comp = factory(args.context, args.vsm_list)
        comp.init(args.sequence, args.chunking, args.key, args.num_class, args.feature_windows, args.label_embedding, args.input_dropout, args.conv2d_config, args.hidden_config, initializer)

        # data
        trn_docs = args.reader.type(args.trn_path, args.reader.params, args.key)
        dev_docs = args.reader.type(args.dev_path, args.reader.params, args.key)

        # train
        comp.train(trn_docs, dev_docs, args.model_path, args.trn_batch, args.dev_batch, args.epoch, args.loss, args.optimizer, args.learning_rate, args.weight_decay)
        print('# of classes: %d' % len(comp.label_map))

    @classmethod
    def eval(cls):
        # create a arg-parser
        parser = argparse.ArgumentParser(description='Evaluate the token tagger',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        # data
        group = parser.add_argument_group("data arguments")

        group.add_argument('-d', '--dev_path', type=str, metavar='filepath',
                           help='filepath to the development data (input)')
        group.add_argument('-m', '--model_path', type=str, metavar='filepath',
                           default=None,
                           help='filepath to the model data (output)')
        group.add_argument('-r', '--reader', type=args_reader, metavar='json or tsv=(str:int)(,#1)*',
                           default=args_reader('json'),
                           help='type of the reader and its configuration to match the data format (default: json)')

        # lexicon
        group = parser.add_argument_group("lexicon arguments")

        group.add_argument('-v', '--vsm_list', type=args_vsm, metavar='(fasttext|word2vec:key:filepath)( #1)*', nargs='+',
                           help='list of (type of vector space model, key, filepath)')

        # evaluation
        group = parser.add_argument_group("training arguments")

        group.add_argument('-cx', '--context', type=args_context, metavar='[cg]int', nargs='+',
                           default=mx.cpu(),
                           help='device context(s)')
        group.add_argument('-db', '--dev_batch', type=int, metavar='int', default=2048,
                           help='batch size for the development data')

        # arguments
        args = parser.parse_args(sys.argv[3:])
        args.vsm_list = tuple(SimpleNamespace(model=n.type(n.filepath), key=n.key) for n in args.vsm_list)
        if isinstance(args.context, list) and len(args.context) == 1: args.context = args.context[0]

        # component
        with open(args.model_path, 'rb') as fin: sequence = pickle.load(fin)
        factory = TokenSequenceTagger if sequence else TokenBatchTagger
        comp = factory(args.context, args.vsm_list)
        comp.load(args.model_path)

        # data
        dev_docs = args.reader.type(args.dev_path, args.reader.params, comp.key)

        # decode
        states = comp.create_states(dev_docs)
        e = comp._evaluate(states, args.dev_batch)
        print(str(e))


if __name__ == '__main__':
    cli = TokenTaggerCLI()
