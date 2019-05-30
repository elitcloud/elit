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
from typing import Sequence

import numpy as np

from elit.structure import Document

__author__ = "Jinho D. Choi, Gary Lai"


class Embedding(abc.ABC):
    """
    :class:`Embedding` is an abstract class to implement embedding models.

    :ivar dim: the dimension of each embedding.
    :ivar pad: the zero vector whose dimension is the same as the embedding (can be used for zero-padding).

    Abstract methods to be implemented:
      - :meth:`Embedding.embed`
    """

    def __init__(self, dim: int):
        """
        :param dim: the dimension of each embedding.
        """
        self.dim = dim
        self.pad = np.zeros(dim).astype('float32')

    @abc.abstractmethod
    def embed(self, docs: Sequence[Document], key: str, **kwargs):
        """
        Adds embeddings to the input documents.

        :param docs: a sequence of input documents.
        :param key: the key to a sentence or a document where embeddings are to be added.
        """
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))
