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
from typing import Tuple, Optional

import numpy as np

from elit.structure import Document
from elit.vsm import LabelMap, VectorSpaceModel

__author__ = 'Jinho D. Choi'


class NLPState(abc.ABC):
    """
    NLPState provides an abstract class to define a decoding strategy.
    Abstract methods to be implemented: init, process, has_next, x, y.
    """

    def __init__(self, document: Document, key: str):
        """
        :param document: an input document.
        :param key: the key to the input document where the predicted labels are to be saved.
        """
        self.document = document
        self.key = key
        self.key_out = key + '-out'
        self.key_gold = key + '-gold'

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
    def y(self) -> Optional[int]:
        """
        :return: the class ID of the gold-standard label for the current state if available; otherwise None.
        """
        pass


class BatchState(NLPState):
    """
    BatchState provides an abstract class to define a decoding strategy in batch mode.
    Batch mode assumes that predictions made by earlier states do not affect predictions made by later states.
    Thus, all predictions can be made in batch where each prediction is independent from one another.
    BatchState is iterable (see self.__iter__() and self.__next__()).
    Abstract methods to be implemented: init, process, has_next, x, y, assign.
    """

    def __init__(self, document: Document, key: str):
        """
        :param document: an input document.
        :param key: the key to the input document where the predicted labels are to be saved.
        """
        super().__init__(document, key)

    def __iter__(self) -> 'BatchState':
        self.init()
        return self

    def __next__(self) -> Tuple[np.ndarray, Optional[int]]:
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
        :return: the row index of the output matrix to be assigned by the next document.
        """
        pass


class SequenceState(NLPState):
    """
    SequenceState provides an abstract class to define a decoding strategy in sequence mode.
    Sequence mode assumes that predictions made by earlier states affect predictions made by later states.
    Thus, predictions need to be made in sequence where earlier predictions get passed onto later states.
    Abstract methods to be implemented: init, process, has_next, x, y.
    """

    def __init__(self, document: Document, key: str):
        """
        :param document: an input document
        :param key: the key to the input document where the predicted labels are to be saved.
        """
        super().__init__(document, key)

    @abc.abstractmethod
    def process(self, output: np.ndarray):
        """
        Applies the predicted output to the current state, then processes to the next state.
        :param output: the predicted output of the current state.
        """
        pass


class SentenceClassificationBatchState(BatchState):
    """
    SentenceClassificationBatchState labels each sentence in the input document with a certain class (e.g., positive or negative for sentiment analysis).
    """

    def __init__(self,
                 document: Document,
                 key: str,
                 label_map: LabelMap,
                 word_vsm: VectorSpaceModel,
                 maxlen: int):
        """
        :param document: an input document.
        :param key: the key to each sentence in the input document where the predicted labels are to be saved.
        :param label_map: collects class labels during training and maps them to unique IDs.
        :param word_vsm: a vector space model for word embeddings.
        :param maxlen: the maximum length of a sentence.
        """
        super().__init__(document, key)
        self.label_map = label_map

        # initialize gold-standard labels if available
        self.gold = [s[self.key_gold] for s in document] if self.key_gold in document.sentences[0] else None

        # initialize embeddings
        self.embs = [word_vsm.document_matrix(s.tokens, maxlen) for s in document]

        # self.init()
        self.sen_id = 0

    def init(self):
        """
        Initializes the pointer to the first sentence.
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
    def y(self) -> Optional[int]:
        """
        :return: the class ID of the current sentence's gold-standard label if available; otherwise, None.
        """
        return None if self.gold is None else self.label_map.add(self.gold[self.sen_id])

    def assign(self, output: np.ndarray, begin: int = 0) -> int:
        """
        Assigns the predicted output to the input document.
        :param output: a matrix where each row contains prediction scores of the corresponding sentence.
        :param begin: the row index of the output matrix corresponding to the first sentence in the input document.
        :return: the row index of the output matrix to be assigned by the next document.
        """
        for i, sentence in enumerate(self.document):
            sentence[self.key_out] = output[begin + i]

        return begin + len(self.document)


class DocumentClassificationBatchState(BatchState):
    """
    DocumentClassificationBatchState labels input document with a certain class.
    """

    def __init__(self,
                 document: Document,
                 key: str,
                 label_map: LabelMap,
                 word_vsm: VectorSpaceModel,
                 maxlen: int):
        """
        :param document: an input document.
        :param key: the key to the input document where the predicted labels are saved.
        :param label_map: collects class labels during training and maps them to unique IDs.
        :param word_vsm: a vector space model for word embeddings.
        :param maxlen: the maximum length of the input document.
        """
        super().__init__(document, key)
        self.label_map = label_map

        # initialize gold-standard labels if available
        self.gold = document.get(self.key_gold, None)

        # initialize embeddings
        self.emb = word_vsm.document_matrix(document.tokens, maxlen)

        # self.init()
        self.doc_id = 0

    def init(self):
        """
        Initializes the pointer to the first sentence.
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
    def y(self) -> Optional[int]:
        """
        :return: the class ID of the current sentence's gold-standard label if available; otherwise, None.
        """
        return self.label_map.add(self.gold) if self.gold is not None else None

    def assign(self, output: np.ndarray, begin: int = 0) -> int:
        """
        Assigns the predicted output to the input document.
        :param output: a matrix where each row contains prediction scores of the corresponding document.
        :param begin: the row index of the output matrix corresponding to this document.
        :return: the row index of the output matrix to be assigned by the next document.
        """
        self.document[self.key_out] = output
        return begin + 1


