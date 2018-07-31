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
import glob
import json
import pickle
from types import SimpleNamespace
from typing import Tuple, Optional, List, Union

import mxnet as mx
import numpy as np

from elit.component import MXNetComponent, SequenceComponent, BatchComponent
from elit.eval import Accuracy, F1
from elit.model import input_namespace, output_namespace, FFNNModel, conv2d_args, hidden_args, \
    loss_args
from elit.state import BatchState, SequenceState, group_states
from elit.structure import Document, Sentence, TOK
from elit.util import BILOU, to_gold, to_out
from elit.vsm import LabelMap, VectorSpaceModel, get_loc_embeddings, x_extract, X_ANY, FastText

__author__ = 'Jinho D. Choi'


# ======================================== State ========================================

class TokenBatchTaggerState(BatchState):
    """
    TokenTaggingBatchState defines the one-pass left-to-right strategy for tagging individual tokens in batch mode.
    """

    def __init__(self,
                 document: Document,
                 key: str,
                 label_map: LabelMap,
                 vsm_list: Tuple[VectorSpaceModel, str],
                 windows: Tuple[int, ...]):
        """
        :param document: an input document.
        :param key: the key to each sentence in the document where predicted labels are to be saved.
        :param label_map: collects class labels during training and maps them to unique IDs.
        :param vsm_list: a list of tuple(vector space model, key).
        :param windows: contextual windows of adjacent tokens for feature extraction.
        """
        super().__init__(document, key)
        self.label_map = label_map
        self.windows = windows

        # initialize gold-standard labels if available
        key_gold = to_gold(key)
        self.gold = [s[key_gold] for s in document] if key_gold in document.sentences[0] else None

        # initialize embeddings
        self.embs = [vsm.sentence_embedding_list(document, key) for vsm, key in vsm_list]
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

    def process(self):
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

    def assign(self, output: np.ndarray, begin: int = 0) -> int:
        """
        Assigns the predicted output to the each token in the input document.
        :param output: a matrix where each row contains prediction scores of the corresponding token.
        :param begin: the row index of the output matrix corresponding to the first token in the input document.
        :return: the row index of the output matrix to be assigned by the next document.
        """
        key_out = to_out(self.key)
        for sentence in self.document:
            end = begin + len(sentence)
            sentence[key_out] = output[begin:end]
            begin = end
        return begin


class TokenSequenceTaggerState(SequenceState):
    """
    TokenTaggingSequenceState defines the one-pass left-to-right strategy for tagging individual tokens in sequence mode.
    In other words, predicted outputs from earlier tokens are used as features to predict later tokens.
    """

    def __init__(self,
                 document: Document,
                 key: str,
                 label_map: LabelMap,
                 vsm_list: Tuple[VectorSpaceModel, str],
                 windows: Tuple[int, ...],
                 padout: np.ndarray):
        """
        :param document: an input document.
        :param key: the key to each sentence in the document where predicted labels are to be saved.
        :param label_map: collects class labels during training and maps them to unique IDs.
        :param vsm_list: a list of tuple(vector space model, key).
        :param windows: contextual windows of adjacent tokens for feature extraction.
        :param padout: a zero-vector whose dimension is the number of class labels, used to zero-pad label embeddings.
        """
        super().__init__(document, key)
        self.label_map = label_map
        self.windows = windows
        self.padout = padout
        self.output = [[self.padout] * len(s) for s in self.document]

        # initialize gold-standard labels if available
        key_gold = to_gold(key)
        self.gold = [s[key_gold] for s in document] if key_gold in document.sentences[0] else None

        # initialize embeddings
        self.embs = [vsm.sentence_embedding_list(document, key) for vsm, key in vsm_list]
        self.embs.append(get_loc_embeddings(document))
        self.embs.append((self.output, self.padout))

        # self.init()
        self.sen_id = 0
        self.tok_id = 0

    def init(self):
        """
        Initializes the pointers to the first otken in the first sentence and the predicted outputs and labels.
        """
        self.sen_id = 0
        self.tok_id = 0

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
                 vsm_list: Tuple[VectorSpaceModel, str]):
        """
        :param ctx: a device context.
        :param vsm_list: a list of tuples (vector space model, key), where the key indicates the key to each sentence to retrieve embeddings for (e.g., TOK).
        """
        super().__init__(ctx)
        self.vsm_list = vsm_list

        # to be loaded/saved
        self.key = None
        self.chunking = None
        self.label_map = None
        self.feature_windows = None
        self.padout = None  # used for sequence mode only

    @abc.abstractmethod
    def create_states(self, documents: List[Document]) -> List[Union[TokenBatchTaggerState, TokenSequenceTaggerState]]:
        """
        :param documents: a list of input documents.
        :return: the list of sequence or batch states corresponding to the input documents.
        """
        pass

    # override
    def finalize(self, state: Union[TokenBatchTaggerState, TokenSequenceTaggerState]):
        """
        Finalizes by saving the predicted tags to the input document once decoding is done.
        If self.chunking, it has a list of chunks instead.
        :param state: an input state.
        """
        key_out = to_out(self.key)
        sequence = isinstance(state, TokenSequenceTaggerState)
        for i, sentence in enumerate(state.document):
            if sequence:
                sentence[key_out] = state.output[i]
            sentence[self.key] = [self.label_map.argmax(o) for o in sentence[key_out]]

    # override
    def eval_metric(self) -> Union[TokenAccuracy, ChunkF1]:
        """
        :return: :class:`ChunkF1` if self.chunking else :class:`TokenAccuracy`.
        """
        return ChunkF1(self.key) if self.chunking else TokenAccuracy(self.key)

    # override
    def init(self,
             key: str,
             sequence: bool,
             chunking: bool,
             num_class: int,
             feature_windows: Tuple[int, ...],
             input_dropout: float = 0.0,
             conv2d_config: Optional[Tuple[SimpleNamespace, ...]] = None,
             hidden_config: Optional[Tuple[SimpleNamespace, ...]] = None,
             initializer: mx.init.Initializer = None,
             **kwargs):
        """
        :param sequence:
        :param key: the key to each sentence where the tags are to be saved.
        :param chunking: if True, this tagging is used for chunking instead.
        :param feature_windows: content windows for feature extraction.
        :param num_class: the number of classes (part-of-speech tags).
        :param input_dropout: a dropout rate to be applied to the input layer.
        :param conv2d_config: configuration for n-gram 2D convolutions.
        :param hidden_config: configuration for hidden layers
        :param initializer: a weight initializer for the gluon block.
        :param kwargs: parameters for the initialization of gluon.Block.
        """
        # configuration
        if initializer is None:
            initializer = mx.init.Xavier(magnitude=2.24, rnd_type='gaussian')
        input_dim = sum([vsm.dim for vsm, _ in self.vsm_list]) + len(X_ANY)
        if sequence:
            input_dim += num_class
        input_config = input_namespace(input_dim, row=len(feature_windows), dropout=input_dropout)
        output_config = output_namespace(num_class)

        # initialization
        self.key = key
        self.chunking = chunking
        self.label_map = LabelMap()
        self.feature_windows = feature_windows
        self.padout = np.zeros(output_config.dim).astype('float32')
        self.model = FFNNModel(self.ctx, initializer, input_config, output_config, conv2d_config, hidden_config, **kwargs)
        print(self.__str__())

    # override
    def load(self, model_path: str, **kwargs):
        """
        :param model_path: the path to a pre-trained model to be loaded.
        :type model_path: str
        """
        with open(model_path, 'rb') as fin:
            self.key = pickle.load(fin)
            self.chunking = pickle.load(fin)
            self.label_map = pickle.load(fin)
            self.feature_windows = pickle.load(fin)
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
            pickle.dump(self.key, fout)
            pickle.dump(self.chunking, fout)
            pickle.dump(self.label_map, fout)
            pickle.dump(self.feature_windows, fout)
            pickle.dump(self.padout, fout)
            pickle.dump(self.model, fout)


class TokenBatchTagger(TokenTagger, BatchComponent):
    """
    TokenBatchTagger implements a tagger that predicts a tag for every token in batch mode.
    """

    def __init__(self,
                 ctx: mx.Context,
                 vsm_list: Tuple[VectorSpaceModel, str]):
        """
        :param ctx: a device context.
        :param vsm_list: a list of tuples (vector space model, key), where the key indicates the key to each sentence to retrieve embeddings for (e.g., TOK).
        """
        BatchComponent.__init__(self, ctx)
        TokenTagger.__init__(self, ctx, vsm_list)

    def __str__(self):
        s = ('TokenBatchTagger',
             '- type: %s' % ('chunking' if self.chunking else 'tagging'),
             '- feature windows: %s' % str(self.feature_windows),
             '- model: %s' % str(self.model))
        return '\n'.join(s)

    # override
    def create_states(self, documents: List[Document]) -> List[TokenBatchTaggerState]:
        """
        :param documents: a list of input documents.
        :return: the list of sequence or batch states corresponding to the input documents.
        """
        return [TokenBatchTaggerState(d, self.key, self.label_map, self.vsm_list, self.feature_windows) for d in documents]


class TokenSequenceTagger(TokenTagger, SequenceComponent):
    """
    TokenSequenceTagger implements a tagger that predicts a tag for every token in sequence mode.
    """

    def __init__(self,
                 ctx: mx.Context,
                 vsm_list: Tuple[VectorSpaceModel, str]):
        """
        :param ctx: a device context.
        :param vsm_list: a list of tuples (vector space model, key), where the key indicates the key to each sentence to retrieve embeddings for (e.g., TOK).
        """
        SequenceComponent.__init__(self, ctx)
        TokenTagger.__init__(self, ctx, vsm_list)

    def __str__(self):
        s = ('TokenSequenceTagger',
             '- type: %s' % ('chunking' if self.chunking else 'tagging'),
             '- feature windows: %s' % str(self.feature_windows),
             '- model: %s' % str(self.model))
        return '\n'.join(s)

    # override
    def create_states(self, documents: List[Document]) -> List[TokenSequenceTaggerState]:
        """
        :param documents: a list of input documents.
        :return: the list of sequence or batch states corresponding to the input documents.
        """
        return group_states(documents, lambda d: TokenSequenceTaggerState(d, self.key, self.label_map, self.vsm_list, self.feature_windows, self.padout))


# ======================================== Command-Line ========================================

def tsv_reader(filepath, key, args):
    documents = []
    wc = sc = 0

    for filename in glob.glob(filepath):
        sentences, tokens, tags = [], [], []
        fin = open(filename)

        for line in fin:
            if line.startswith('#'):
                continue
            l = line.split()

            if l:
                tokens.append(l[args.tok])
                tags.append(l[args.tag])
            elif len(tokens) > 0:
                wc += len(tokens)
                sentences.append(Sentence({TOK: tokens, to_gold(key): tags}))
                tokens, tags = [], []

        if len(tokens) > 0:
            wc += len(tokens)
            sentences.append(Sentence({TOK: tokens, to_gold(key): tags}))

        fin.close()
        sc += len(sentences)
        documents.append(Document(sen=sentences))

    print('Reading: dc = %d, sc = %d, wc = %d' % (len(documents), sc, wc))
    return documents


def json_reader(filepath) -> List[Document]:
    # TODO: to be filled
    documents = []
    dc = wc = sc = 0

    for filename in glob.glob('{}/*.json'.format(filepath)):
        assert filename.endswith('.json')
        with open(filename) as f:
            docs = json.load(f)
            for doc in docs:
                sentences = []
                for sen in doc['sen']:
                    wc += len(sen['tok'])
                    sentences.append(Sentence(sen))
                sc += len(sentences)
                documents.append(Document(sen=sentences))
            dc += len(documents)
    print('Reading: dc = %d, sc = %d, wc = %d' % (dc, sc, wc))
    return documents


def reader_args(s):
    """
    :param s: (tsv|json)(;\\d:\\d)*
    :return: reader, SimpleNamespace(tok, pos)
    """
    r = s.split(';')
    if r[0] == 'tsv':
        t = r[1].split(':')
        return tsv_reader, SimpleNamespace(tok=int(t[0]), tag=int(t[1]))
    else:
        return json_reader, None


def train_args():
    def int_tuple(s):
        """
        :param s: \\d(,\\d)*
        :return: tuple of int
        """
        return tuple(map(int, s.split(',')))

    parser = argparse.ArgumentParser('Train: part-of-speech tagging')

    # data
    parser.add_argument('-t', '--trn_path', type=str, metavar='filepath',
                        help='path to the training data (input)')
    parser.add_argument('-d', '--dev_path', type=str, metavar='filepath',
                        help='path to the development data (input)')
    parser.add_argument('-m', '--model_path', type=str, metavar='filepath', default=None,
                        help='path to the model data (output)')
    parser.add_argument('-r', '--reader', type=reader_args, metavar='(tsv|json)(;\\d:\\d)*',
                        default=(tsv_reader, SimpleNamespace(tok=0, tag=1)),
                        help='reader configuration')

    # generic
    parser.add_argument('-key', '--key', type=str, metavar='str',
                        help='path to the model data (output)')
    parser.add_argument('-seq', '--sequence', type=bool, metavar='boolean', default=False,
                        help='if set, use sequence mode')
    parser.add_argument('-chu', '--chunking', type=bool, metavar='boolean', default=False,
                        help='if set, generate chunks')

    # lexicon
    parser.add_argument('-wv', '--word_vsm', type=str, metavar='filepath',
                        help='vector space model for word embeddings')

    # configuration
    parser.add_argument('-nc', '--num_class', type=int, metavar='int',
                        help='number of classes (part-of-speech tags)')
    parser.add_argument('-fw', '--feature_windows', type=int_tuple, metavar='int[,int]*',
                        default=tuple(range(-3, 4)),
                        help='contextual windows for feature extraction')
    parser.add_argument('-ir', '--input_dropout', type=float, metavar='float', default=0.0,
                        help='dropout rate applied to the input layer')
    parser.add_argument('-cc', '--conv2d_config', type=conv2d_args,
                        metavar='(ngram:filters:activation:pool:dropout)(;#1)*',
                        default=tuple(SimpleNamespace(ngram=i, filters=128, activation='relu', pool=None, dropout=0.2) for i in range(1, 6)),
                        help='configuration for the convolution layer')
    parser.add_argument('-hc', '--hidden_config', type=hidden_args, metavar='(dim:activation:dropout)(;#1)*', default=None,
                        help='configuration for the hidden layer')

    # training
    parser.add_argument('-cx', '--ctx', type=str, metavar='[cg]\\d', default=None,
                        help='device context')
    parser.add_argument('-ep', '--epoch', type=int, metavar='int', default=50,
                        help='number of epochs')
    parser.add_argument('-tb', '--trn_batch', type=int, metavar='int', default=64,
                        help='batch size for training')
    parser.add_argument('-db', '--dev_batch', type=int, metavar='int', default=4096,
                        help='batch size for evaluation')
    parser.add_argument('-lo', '--loss', type=loss_args, metavar='str', default=None,
                        help='loss function')
    parser.add_argument('-op', '--optimizer', type=str, metavar='str', default='adagrad',
                        help='optimizer algorithm')
    parser.add_argument('-lr', '--learning_rate', type=float, metavar='float', default=0.01,
                        help='learning rate')
    parser.add_argument('-wd', '--weight_decay', type=float, metavar='float', default=0.0,
                        help='weight decay')

    args = parser.parse_args()
    return args


def evaluate_args():
    parser = argparse.ArgumentParser('Train: part-of-speech tagging')

    # data
    parser.add_argument('-d', '--dev_path', type=str, metavar='filepath',
                        help='path to the evaluation data (input)')
    parser.add_argument('-m', '--model_path', type=str, metavar='filepath', default=None,
                        help='path to the model data (output)')
    parser.add_argument('-r', '--reader', type=reader_args, metavar='(tsv|json)(;\\d:\\d)*',
                        default=(tsv_reader, SimpleNamespace(tok=0, tag=1)),
                        help='reader configuration')

    parser.add_argument('-seq', '--sequence', type=bool, metavar='boolean', default=False,
                        help='if set, use sequence mode')

    # lexicon
    parser.add_argument('-wv', '--word_vsm', type=str, metavar='filepath',
                        help='vector space model for word embeddings')

    # evaluation
    parser.add_argument('-cx', '--ctx', type=str, metavar='[cg]\\d', default=None,
                        help='device context')
    parser.add_argument('-db', '--dev_batch', type=int, metavar='int', default=4096,
                        help='batch size for evaluation')

    args = parser.parse_args()
    return args


def train():
    # cml arguments
    args = train_args()
    if args.ctx is None:
        args.ctx = mx.cpu()
    if args.loss is None:
        args.loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()

    # vector space models
    vsm_list = [(FastText(args.word_vsm), TOK)]

    # component
    TokenTagger = TokenSequenceTagger if args.sequence else TokenBatchTagger
    comp = TokenTagger(args.ctx, vsm_list)
    initializer = mx.init.Xavier(magnitude=2.24, rnd_type='gaussian')
    comp.init(args.key, args.sequence, args.chunking, args.num_class, args.feature_windows, args.input_dropout, args.conv2d_config, args.hidden_config, initializer)

    # data
    reader, reader_args = args.reader
    trn_docs = reader(args.trn_path, args.key, reader_args)
    dev_docs = reader(args.dev_path, args.key, reader_args)

    # train
    comp.train(trn_docs, dev_docs, args.model_path, args.trn_batch, args.dev_batch, args.epoch, args.loss, args.optimizer, args.learning_rate, args.weight_decay)
    print('# of classes: %d' % len(comp.label_map))


def evaluate():
    # cml arguments
    args = train_args()
    if args.ctx is None:
        args.ctx = mx.cpu()

    # vector space models
    vsm_list = [(FastText(args.word_vsm), TOK)]

    # component
    TokenTagger = TokenSequenceTagger if args.sequence else TokenBatchTagger
    comp = TokenTagger(args.ctx, vsm_list)
    comp.load(args.model_path)

    # data
    reader, reader_args = args.reader
    dev_docs = reader(args.dev_path, comp.key, reader_args)

    # decode
    states = comp.create_states(dev_docs)
    e = comp._evaluate(states, args.dev_batch)
    print('DEV: %s' % str(e.get()))


if __name__ == '__main__':
    train()
    evaluate()
