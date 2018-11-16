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
from typing import List, Optional

import numpy as np

from elit.structure import Document, to_gold, to_out
from elit.util.vsm import LabelMap, VectorSpaceModel

__author__ = 'Jinho D. Choi'


# ======================================== State =========================

class DocumentClassifierState(BatchState):
    """
    DocumentClassificationBatchState labels input document with a certain class in batch mode.
    """

    def __init__(self,
                 document: Document,
                 key: str,
                 sentence_level: bool,
                 label_map: LabelMap,
                 word_vsm: VectorSpaceModel,
                 maxlen: int):
        """
        :param document: an input document.
        :param key: the key to the input document where the predicted labels are saved.
        :param sentence_level: if True, treats each sentence as a document.
        :param label_map: collects class labels during training and maps them to unique IDs.
        :param word_vsm: a vector space model for word embeddings.
        :param maxlen: the maximum length of a document;
                       for sentence-level, it is set to the maximum number of words in a sentence, and
                       for document-level, it is set to the maximum number of words across all sentences.
        """
        super().__init__(document, key)
        self.sentence_level = sentence_level
        self.label_map = label_map

        # initialize gold-standard labels and embeddings
        key_gold = to_gold(key)

        if self.sentence_level:
            self.gold = [
                s[key_gold] for s in document] if key_gold in document.sentences[0] else None
            self.embs = [
                word_vsm.embedding_matrix(
                    s.tokens,
                    maxlen) for s in document]
        else:
            self.gold = [document[key_gold]] if key_gold in document else None
            self.emb = [word_vsm.embedding_matrix(document.tokens, maxlen)]

        # self.init()
        self.doc_id = 0

    def init(self):
        """
        Initializes the pointers to the first document.
        """
        self.doc_id = 0

    def process(self):
        """
        Processes to the next document.
        """
        self.doc_id += 1

    @property
    def has_next(self) -> bool:
        """
        :return: False if no more document is left to be classified; otherwise, True.
        """
        if self.sentence_level:
            return self.doc_id < len(self.document)
        return self.doc_id == 0

    @property
    def x(self) -> List[np.ndarray]:
        """
        :return: the embedding matrix of the current document.
        """
        return self.embs[self.doc_id]

    @property
    def y(self) -> Optional[int]:
        """
        :return: the class ID of the current document's gold-standard label if available; otherwise, None.
        """
        return self.label_map.add(
            self.gold[self.doc_id]) if self.gold is not None else None

    def assign(self, output: np.ndarray, begin: int = 0) -> int:
        """
        Assigns the predicted output to the input document.
        :param output: a matrix where each row contains prediction scores of the corresponding document.
        :param begin: the row index of the output matrix corresponding to this document.
        :return: the row index of the output matrix to be assigned by the next document.
        """
        key_out = to_out(self.key)

        if self.sentence_level:
            for i, sentence in enumerate(self.document):
                sentence[key_out] = output[begin + i]
            return begin + len(self.document)
        else:
            self.document[key_out] = output
            return begin + 1
