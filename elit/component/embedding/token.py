# ========================================================================
# Copyright 2018 ELIT
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
import inspect
from typing import List, Sequence

import numpy as np

from elit.nlp.embedding import Embedding
from elit.structure import Document

__author__ = "Gary Lai"


class TokenEmbedding(Embedding):
    """
    :class:`TokenEmbedding` is an abstract class to implement token-based embedding models.

    Subclasses:
      - :class:`Word2Vec`
      - :class:`FastText`

    Abstract methods to be implemented:
      - :meth:`TokenEmbedding.emb`
    """

    def __init__(self, dim: int):
        """
        :param dim: the dimension of each embedding.
        """
        super().__init__(dim)

    # override
    def embed(self, docs: Sequence[Document], key: str, **kwargs):
        """
        Adds a list of embeddings to each sentence corresponding to its tokens.

        :param docs: a sequence of input documents.
        :param key: the key to each sentence where the list of embeddings is to be added.
        """
        for doc in docs:
            for sen in doc:
                sen[key] = self.emb_list(sen.tokens)

    @abc.abstractmethod
    def emb(self, token: str) -> np.ndarray:
        """
        :param token: the input token.
        :return: the embedding of the input token.
        """
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))

    def emb_list(self, tokens: Sequence[str]) -> List[np.ndarray]:
        """
        :param tokens: the sequence of input tokens.
        :return: the list of embeddings for the corresponding tokens.
        """
        return [self.emb(value) for value in tokens]

    def emb_matrix(self, tokens: Sequence[str], maxlen: int) -> List[np.ndarray]:
        """
        :param tokens: the sequence of input tokens.
        :param maxlen: the maximum length of the output list;
                       if ``> len(values)``, the bottom part of the matrix is padded with zero embeddings;
                       if ``< len(values)``, embeddings of the exceeding values are discarded from the resulting matrix.
        :return: the matrix where each row is the embedding of the corresponding value.
        """
        return [self.emb(tokens[i]) if i < len(tokens) else self.pad for i in range(maxlen)]