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
import pickle
from types import SimpleNamespace
from typing import Tuple, Optional, Type, List

import mxnet as mx
import numpy as np

from elit.component import MXNetComponent, SequenceComponent, BatchComponent, FFNNComponent
from elit.model import input_namespace, output_namespace

from elit.state import BatchState, SequenceState, NLPState
from elit.structure import Document
from elit.util import group_states, EvalMetric, F1, Accuracy, pkl, gln
from elit.vsm import LabelMap, VectorSpaceModel, get_vsm_embeddings, get_loc_embeddings, x_extract, X_ANY

__author__ = 'Jinho D. Choi'


class TokenTaggerBatchState(BatchState):
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
        self.gold = [s[self.key_gold] for s in document] if self.key_gold in document.sentences[0] else None

        # initialize embeddings
        self.embs = [get_vsm_embeddings(vsm, document, key) for vsm, key in vsm_list]
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
        t = len(self.document.sentences[self.sen_id])
        l = ([x_extract(self.tok_id, w, t, emb[self.sen_id], pad) for w in self.windows] for emb, pad in self.embs)
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
        for sentence in self.document:
            end = begin + len(sentence)
            sentence[self.key_out] = output[begin:end]
            begin = end
        return begin


class TokenTaggerSequenceState(SequenceState):
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
        self.output = []

        # initialize gold-standard labels if available
        self.gold = [s[self.key_gold] for s in document] if self.key_gold in document.sentences[0] else None

        # initialize embeddings
        self.embs = [get_vsm_embeddings(vsm, document, key) for vsm, key in vsm_list]
        self.embs.append(get_loc_embeddings(document))
        self.embs.append((self.output, self.padout))

        # self.init()
        self.sen_id = 0
        self.tok_id = 0

        for s in self.document:
            o = [self.padout] * len(s)
            self.output.append(o)
            s[self.key_out] = o

    def init(self):
        """
        Initializes the pointers to the first otken in the first sentence and the predicted outputs and labels.
        """
        self.sen_id = 0
        self.tok_id = 0

        for i, s in enumerate(self.document):
            o = [self.padout] * len(s)
            self.output[i] = o
            s[self.key_out] = o

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
        t = len(self.document.sentences[self.sen_id])
        l = ([x_extract(self.tok_id, w, t, emb[self.sen_id], pad) for w in self.windows] for emb, pad in self.embs)
        return np.column_stack(l)

    @property
    def y(self) -> Optional[int]:
        """
        :return: the class ID of the current token's gold-standard label if available; otherwise, None.
        """
        return self.label_map.add(self.gold[self.sen_id][self.tok_id]) if self.gold is not None else None


def token_tagger_class(sequence: bool, chunk: bool) -> Type[MXNetComponent]:
    """
    :param sequence: if True, FFNNTokenTagger inherits :class:`SequenceComponent`; otherwise :class:`BatchComponent`.
    :return: a token tagger using a feed-forward neural network model.
    """
    ModeComponent = SequenceComponent if sequence else BatchComponent

    class TokenTagger(FFNNComponent, ModeComponent):
        """
        FFNNTokenTagger provides an abstract class to implement an NLP component that predicts a tag for every token.
        """

        def __init__(self,
                     ctx: mx.Context,
                     vsm_list: Tuple[VectorSpaceModel, str],
                     key: str):
            """
            :param ctx: a device context.
            :param vsm_list: a list of tuple(vector space model, key).
            """
            ModeComponent.__init__(self, ctx)
            FFNNComponent.__init__(self)
            self.vsm_list = vsm_list
            self.padout = None
            self.key = key

            # to be initialized
            self.label_map = None
            self.feature_windows = None

        def __str__(self):
            s = ('TokenTagger',
                 '- feature windows: %s' % str(self.feature_windows),
                 '- input layer    : %s' % str(self.input_config).replace('namespace', ''),
                 '- output layer   : %s' % str(self.output_config).replace('namespace', ''),
                 '- conv2d layer   : %s' % str(self.conv2d_config).replace('namespace', ''),
                 '- hidden layer   : %s' % str(self.hidden_config).replace('namespace', ''))
            return '\n'.join(s)

        def create_states(self, documents: List[Document]) -> List[NLPState]:
            """
            :param documents: a list of input documents.
            :return: the list of initial states corresponding to the input documents.
            """
            if sequence:
                def create(document):
                    return TokenTaggerSequenceState(document, self.key, self.label_map, self.vsm_list, self.feature_windows, self.padout)
                return group_states(documents, create)
            else:
                return [TokenTaggerBatchState(d, self.key, self.label_map, self.vsm_list, self.feature_windows) for d in documents]

        @abc.abstractmethod
        def finalize(self, document: Document):
            """
            Finalizes by saving the predicted labels to the input document once decoding is done.
            """
            pass

        @abc.abstractmethod
        def eval_metric(self) -> EvalMetric:
            """
            :return: the evaluation metric for this component.
            """
            # TODO: EvalMetric
            return F1() if chunk else Accuracy()

        # override
        def init(self,
                 feature_windows: Tuple[int, ...] = tuple(range(-3, 4)),
                 num_class: int = 50,
                 input_dropout: float = 0.0,
                 conv2d_config: SimpleNamespace = None,
                 hidden_config: SimpleNamespace = None,
                 **kwargs):
            """
            :param label_embedding: True if label embeddings are used as features; otherwise, False.
            :type label_embedding: bool
            :param feature_windows: contextual windows for feature extraction.
            :type feature_windows: tuple of int
            :param num_class: the number of classes (part-of-speech tags).
            :type num_class: int
            :param input_dropout: a dropout rate to be applied to the input layer.
            :type input_dropout: float
            :param conv2d_config: configuration for n-gram 2D convolutions.
            :type conv2d_config: list of SimpleNamespace
            :param hidden_config: configuration for hidden layers
            :type hidden_config: list of SimpleNamespace
            :param kwargs: parameters for the initialization of gluon.Block.
            :type kwargs: dict
            :return: self
            :rtype: NLPComponent
            """
            # input dimension
            input_dim = sum([vsm.dim for vsm, _ in self.vsm_list]) + len(X_ANY)
            if sequence: input_dim += num_class
            input_config = input_namespace(input_dim, maxlen=len(feature_windows), dropout=input_dropout)
            output_config = output_namespace(num_class)

            # initialization
            self.label_map = LabelMap()
            self.feature_windows = feature_windows
            self.model = super()._init(self.ctx, input_config, output_config, conv2d_config, hidden_config)
            self.padout = np.zeros(self.output_config.dim).astype('float32') if sequence else None
            print(self.__str__())

        # override
        def load(self, model_path: str, **kwargs):
            """
            :param model_path: the path to a pre-trained model to be loaded.
            :type model_path: str
            :param kwargs: parameters for the initialization of gluon.Block.
            :type kwargs: dict
            :return: self
            :rtype: NLPComponent
            """
            with open(pkl(model_path), 'rb') as fin:
                self.label_map = pickle.load(fin)
                self.feature_windows = pickle.load(fin)
                super()._load(self.ctx, fin, gln(model_path), **kwargs)
            self.padout = np.zeros(self.output_config.dim).astype('float32') if sequence else None
            print(self.__str__())

        # override
        def save(self, model_path, **kwargs):
            """
            :param model_path: the filepath where the model is to be saved.
            :type model_path: str
            """
            with open(pkl(model_path), 'wb') as fout:
                pickle.dump(self.label_map, fout)
                pickle.dump(self.feature_windows, fout)
                super()._save(fout, gln(model_path), self.model)

    return TokenTagger