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

import logging
from typing import Sequence

import gluonnlp as nlp
import numpy as np

from elit.embedding import TokenEmbedding

__author__ = "Gary Lai"


class FastText(TokenEmbedding):
    """
    :class:`FastText` is a token-based model trained by `FastText <https://github.com/facebookresearch/fastText>`_.
    """

    def __init__(self, filepath: str):
        """
        :param filepath: the path to the binary file containing a word embedding model trained by FastText (``*.bin``).
        """
        logging.info('FastText')
        logging.info('- model: {}'.format(filepath))
        self.model = nlp.model.train.FasttextEmbeddingModel.load_fasttext_format(filepath)
        dim = self.model._kwargs['output_dim']
        super().__init__(dim)
        logging.info('- vocab = %d, dim = %d' % (self.model._kwargs['num_words'], dim))

    # override
    def emb(self, token: str) -> np.ndarray:
        assert isinstance(token, str)
        return self.model[token].asnumpy()

    # override
    def emb_list(self, tokens: Sequence[str]) -> Sequence[np.ndarray]:
        """
        :param tokens: the sequence of input tokens.
        :return: the list of embeddings for the corresponding tokens.
        """
        assert isinstance(tokens, list)
        return self.model[tokens].asnumpy()
