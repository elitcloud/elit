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
from typing import Tuple, Union, List

import numpy as np

from elit.structure import Document, TOK
from elit.util import EvalMetric
from elit.vsm import LabelMap, VectorSpaceModel, get_vsm_embeddings, get_loc_embeddings, x_extract

__author__ = 'Jinho D. Choi'


class NLPState(abc.ABC):
    def __init__(self, document: Document):
        """
        NLPState provides an abstract class to define a decoding strategy.
        :param document: an input document.
        """
        self.document = document

    @abc.abstractmethod
    def init(self, **kwargs):
        """
        Sets to the initial state.
        :param kwargs: custom arguments.
        """
        pass

    @abc.abstractmethod
    def process(self, **kwargs):
        """
        Applies any custom arguments to the current state if available, then processes to the next state.
        :param kwargs: custom arguments
        """
        pass

    @property
    @abc.abstractmethod
    def has_next(self) -> bool:
        """
        :return: True if there exists a next state to be processed; otherwise, False.
        """
        pass

    @property
    @abc.abstractmethod
    def x(self) -> np.ndarray:
        """
        :return: the feature vector (or matrix) extracted from the current state.
        """
        pass

    @property
    @abc.abstractmethod
    def y(self) -> Union[int, None]:
        """
        :return: the class ID of the gold-standard label for the current state if available; otherwise None.
        """
        pass

    @abc.abstractmethod
    def eval(self, metric: Union[EvalMetric, Tuple[EvalMetric, ...]]):
        """
        Updates the evaluation metric by comparing the gold-standard labels (if available) and the predicted labels.
        :param metric: an evaluation metric or a tuple of evaluation metrics.
        """
        pass


class BatchState(NLPState):
    def __init__(self, document: Document):
        """
        BatchState provides an abstract class to define a decoding strategy in batch mode.
        Batch mode assumes that predictions made by earlier states do not affect predictions made by later states.
        Thus, all predictions can be made in batch where each prediction is independent from one another.
        BatchState is iterable (see self.__iter__() and self.__next__()).
        :param document: an input document.
        """
        super().__init__(document)

    def __iter__(self) -> 'BatchState':
        self.init()
        return self

    def __next__(self) -> Tuple[np.ndarray, Union[int, None]]:
        """
        :return: self.x, self.y
        """
        if not self.has_next: raise StopIteration
        x, y = self.x, self.y
        self.process()
        return x, y

    @abc.abstractmethod
    def assign(self, output: np.ndarray, begin: int = 0) -> int:
        """
        Assigns the predicted output in batch to the input document.
        :param output: a matrix where each row contains prediction scores of the corresponding state.
        :param begin: the row index of the output matrix corresponding to the initial state.
        :return: the number of outputs used by this document.
        """
        pass


class SequenceState(NLPState):
    def __init__(self, document: Document):
        """
        SequenceState provides an abstract class to define a decoding strategy in sequence mode.
        Sequence mode assumes that predictions made by earlier states affect predictions made by later states.
        Thus, predictions need to be made in sequence where earlier predictions get passed onto later states.
        :param document: an input document
        :param gold: True if gold-standard labels are provided in the input document; otherwise, False.
        """
        super().__init__(document)

    @abc.abstractmethod
    def process(self, output: np.ndarray):
        """
        Applies the predicted output to the current state, then processes to the next state.
        :param output: the predicted output of the current state.
        """
        pass


class SentenceClassificationBatchState(BatchState):
    def __init__(self,
                 document: Document,
                 label_map: LabelMap,
                 word_vsm: VectorSpaceModel,
                 maxlen: int,
                 key: str,
                 key_out: Union[str, None] = None,
                 gold: bool = False):
        """
        SentenceClassificationBatchState provides an abstract class that labels each sentence in the input document
        with a certain class (e.g., positive or negative for sentiment analysis).
        :param document: an input document.
        :param label_map: the mapping between class labels and their unique IDs.
        :param word_vsm: a vector space model for word embeddings.
        :param maxlen: the maximum length of a sentence.
        :param key: the key to each sentence in the input document where the predicted labels are saved.
        :param key_out: the key to each sentence in the input document where the output scores are saved;
                        if None, the output scores are not saved in the input document.
        :param gold: True if gold-standard labels are provided in the input document; otherwise, False.
        """
        super().__init__(document)
        self.label_map = label_map
        self.embs = [word_vsm.document_matrix(sentence.tokens, maxlen) for sentence in document]
        self.key = key
        self.key_out = key_out

        # retrieve gold-standard labels if available
        self.gold = [sentence[key] for sentence in document] if gold else None

        # self.init()
        self.sen_id = 0

    @abc.abstractmethod
    def eval(self, metric: Union[EvalMetric, Tuple[EvalMetric, ...]]):
        """
        Updates the evaluation metric by comparing the gold-standard labels (if available) and the predicted labels.
        :param metric: an evaluation metric or a tuple of evaluation metrics.
        """
        pass

    def init(self):
        """
        Sets the pointer to the first sentence.
        """
        self.sen_id = 0

    def process(self):
        """
        Processes to the next sentence.
        """
        self.sen_id += 1

    @property
    def has_next(self) -> bool:
        """
        :return: False if no more sentence is left to be classified; otherwise, True.
        """
        return 0 <= self.sen_id < len(self.document)

    @property
    def x(self) -> np.ndarray:
        """
        :return: the document matrix of the current sentence.
        """
        return self.embs[self.sen_id]

    @property
    def y(self) -> Union[int, None]:
        """
        :return: the class ID of the current sentence's gold-standard label if available; otherwise, None.
        """
        return self.label_map.add(self.gold[self.sen_id]) if self.gold is not None else None

    def assign(self, output: np.ndarray, begin: int = 0) -> int:
        """
        Assigns the predicted output to the input document.
        :param output: a matrix where each row contains prediction scores of the corresponding sentence.
        :param begin: the row index of the output matrix corresponding to the initial state.
        :return: the number of sentences in this document.
        """
        for i, sentence in enumerate(self.document):
            o = output[begin + i]
            sentence[self.key] = self.label_map.argmax(o)
            if self.key_out: sentence[self.key_out] = o

        return len(self.document)


class DocumentClassificationBatchState(BatchState):
    def __init__(self,
                 document: Document,
                 label_map: LabelMap,
                 word_vsm: VectorSpaceModel,
                 maxlen: int,
                 key: str,
                 key_out: Union[str, None] = None,
                 gold: bool = False):
        """
        DocumentClassificationBatchState provides an abstract class that labels input document with a certain class.
        :param document: an input document.
        :param label_map: the mapping between class labels and their unique IDs.
        :param word_vsm: a vector space model for word embeddings.
        :param maxlen: the maximum length of the input document.
        :param key: the key to the input document where the predicted labels are saved.
        :param key_out: the key to the input document where the output scores are saved;
                        if None, the output scores are not saved in the input document.
        :param gold: True if gold-standard labels are provided in the input document; otherwise, False.
        """
        super().__init__(document)
        self.label_map = label_map
        self.emb = word_vsm.document_matrix(document.tokens, maxlen)
        self.key = key
        self.key_out = key_out

        # retrieve gold-standard labels if available
        self.gold = document[key] if gold else None

        # self.init()
        self.doc_id = 0

    @abc.abstractmethod
    def eval(self, metric: Union[EvalMetric, Tuple[EvalMetric, ...]]):
        """
        Updates the evaluation metric by comparing the gold-standard labels (if available) and the predicted labels.
        :param metric: an evaluation metric or a tuple of evaluation metrics.
        """
        pass

    def init(self):
        """
        Sets the pointer to the first sentence.
        """
        self.doc_id = 0

    def process(self):
        """
        Processes to the next sentence.
        """
        self.doc_id += 1

    @property
    def has_next(self) -> bool:
        """
        :return: False if no more sentence is left to be classified; otherwise, True.
        """
        return 0 == self.doc_id

    @property
    def x(self) -> np.ndarray:
        """
        :return: the document matrix of the input document.
        """
        return self.emb

    @property
    def y(self) -> Union[int, None]:
        """
        :return: the class ID of the current sentence's gold-standard label if available; otherwise, None.
        """
        return self.label_map.add(self.gold) if self.gold is not None else None

    def assign(self, output: np.ndarray, begin: int = 0) -> int:
        """
        Assigns the predicted output to the input document.
        :param output: a matrix where the begin'th row contains prediction scores for the input document.
        :param begin: the row index of the output matrix corresponding to this document.
        :return: 1.
        """
        self.document[self.key] = self.label_map.argmax(output)
        if self.key_out: self.document[self.key_out] = output
        return 1


class TokenTaggingBatchState(NLPState):
    def __init__(self,
                 document: Document,
                 label_map: LabelMap,
                 token_vsm: VectorSpaceModel,
                 windows: Tuple[int, ...],
                 padout: Union[np.ndarray, None],
                 key: Union[str, None]):
        """
        TokenTaggingState defines the one-pass left-to-right strategy for tagging individual tokens.
        This class support both batch and sequence modes.
        :param document: an input document.
        :param label_map: collects class labels during training and maps them to unique IDs.
        :param token_vsm: a vector space model to retrieve token embeddings.
        :param windows: contextual windows of adjacent tokens for feature extraction.
        :param padout: a zero-vector whose dimension is same as the number of class labels;
                       if not None, it is used to zero-pad label embeddings.
        :param key: the key to each sentence in the document where predicted labels are to be saved.
        """
        super().__init__(document)
        self.label_map = label_map
        self.windows = windows
        self.padout = padout

        # initialize embeddings
        self.embs = [get_vsm_embeddings(token_vsm, document, TOK)]
        self.embs.append(get_loc_embeddings(document))
        if padout: self.embs.append(self._label_embeddings)

        # initialize labels
        self.gold = []  # gold labels
        self.pred = []  # predicted labels

        for sentence in document:
            if key in sentence: self.gold.append(sentence[key])
            ph = [None] * len(sentence)
            self.pred.append(ph)
            sentence[key] = ph

        # self.init()
        self.sen_id = 0  # sentence ID
        self.tok_id = 0  # token ID
        if padout: self.output = [[self.padout] * len(s) for s in document]

    @abc.abstractmethod
    def eval(self, metric: EvalMetric):
        pass

    def init(self):
        self.sen_id = 0
        self.tok_id = 0

        if self.padout:
            for i, s in enumerate(self.document):
                self.output[i] = [self.padout] * len(s)

    def process(self, output: Union[np.ndarray, None] = None):
        # apply the output to the current state
        if output:
            self.output[self.sen_id][self.tok_id] = output
            self.pred[self.sen_id][self.tok_id] = self.label_map.argmax(output)

        # process to the next state
        self.tok_id += 1
        if self.tok_id == len(self.document.get_sentence(self.sen_id)):
            self.sen_id += 1
            self.tok_id = 0

    @property
    def has_next(self) -> bool:
        return 0 <= self.sen_id < len(self.document)

    @property
    def x(self) -> np.ndarray:
        t = len(self.document.get_sentence(self.sen_id))
        l = ([x_extract(self.tok_id, w, t, emb[self.sen_id], pad) for w in self.windows] for emb, pad in self.embs)
        n = np.column_stack(l)
        return n

    @property
    def y(self) -> np.ndarray:
        return self.label_map.add(self.gold[self.sen_id][self.tok_id]) if self.gold is not None else None

    def assign(self, output: np.ndarray, begin: int = 0) -> int:
        for i, sentence in enumerate(self.document):
            end = begin + len(sentence)
            for j in range(begin, end):
                self.pred[i][j - begin] = self.label_map.argmax(output[j])
            begin = end
        return begin

    @property
    def _label_embeddings(self) -> Union[Tuple[List[List[np.ndarray]], np.ndarray], None]:
        """
        :return: (self.output, self.padout) if sequence mode; otherwise, None
        """
        return self.output, self.padout if self.padout else None


class TokenTaggingSequenceState(NLPState):
    def __init__(self,
                 document: Document,
                 label_map: LabelMap,
                 token_vsm: VectorSpaceModel,
                 windows: Tuple[int, ...],
                 padout: Union[np.ndarray, None],
                 key: Union[str, None]):
        """
        TokenTaggingState defines the one-pass left-to-right strategy for tagging individual tokens.
        This class support both batch and sequence modes.
        :param document: an input document.
        :param label_map: collects class labels during training and maps them to unique IDs.
        :param token_vsm: a vector space model to retrieve token embeddings.
        :param windows: contextual windows of adjacent tokens for feature extraction.
        :param padout: a zero-vector whose dimension is same as the number of class labels;
                       if not None, it is used to zero-pad label embeddings.
        :param key: the key to each sentence in the document where predicted labels are to be saved.
        """
        super().__init__(document)
        self.label_map = label_map
        self.windows = windows
        self.padout = padout

        # initialize embeddings
        self.embs = [get_vsm_embeddings(token_vsm, document, TOK)]
        self.embs.append(get_loc_embeddings(document))
        if padout: self.embs.append(self._label_embeddings)

        # initialize labels
        self.gold = []  # gold labels
        self.pred = []  # predicted labels

        for sentence in document:
            if key in sentence: self.gold.append(sentence[key])
            ph = [None] * len(sentence)
            self.pred.append(ph)
            sentence[key] = ph

        # self.init()
        self.sen_id = 0  # sentence ID
        self.tok_id = 0  # token ID
        if padout: self.output = [[self.padout] * len(s) for s in document]

    @abc.abstractmethod
    def eval(self, metric: EvalMetric):
        pass

    def init(self):
        self.sen_id = 0
        self.tok_id = 0

        if self.padout:
            for i, s in enumerate(self.document):
                self.output[i] = [self.padout] * len(s)

    def process(self, output: Union[np.ndarray, None] = None):
        # apply the output to the current state
        if output:
            self.output[self.sen_id][self.tok_id] = output
            self.pred[self.sen_id][self.tok_id] = self.label_map.argmax(output)

        # process to the next state
        self.tok_id += 1
        if self.tok_id == len(self.document.get_sentence(self.sen_id)):
            self.sen_id += 1
            self.tok_id = 0

    @property
    def has_next(self) -> bool:
        return 0 <= self.sen_id < len(self.document)

    @property
    def x(self) -> np.ndarray:
        t = len(self.document.get_sentence(self.sen_id))
        l = ([x_extract(self.tok_id, w, t, emb[self.sen_id], pad) for w in self.windows] for emb, pad in self.embs)
        n = np.column_stack(l)
        return n

    @property
    def y(self) -> np.ndarray:
        return self.label_map.add(self.gold[self.sen_id][self.tok_id]) if self.gold is not None else None

    def assign(self, output: np.ndarray, begin: int = 0) -> int:
        for i, sentence in enumerate(self.document):
            end = begin + len(sentence)
            for j in range(begin, end):
                self.pred[i][j - begin] = self.label_map.argmax(output[j])
            begin = end
        return begin

    @property
    def _label_embeddings(self) -> Union[Tuple[List[List[np.ndarray]], np.ndarray], None]:
        """
        :return: (self.output, self.padout) if sequence mode; otherwise, None
        """
        return self.output, self.padout if self.padout else None
