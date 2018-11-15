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

import numpy as np
from gensim.models import KeyedVectors

from elit.nlp.embedding import TokenEmbedding

__author__ = "Gary Lai"


class Word2Vec(TokenEmbedding):
    """
    :class:`Word2Vec` is a token-based model trained by `Word2Vec <https://github.com/tmikolov/word2vec>`_.
    """

    def __init__(self, filepath: str):
        """
        :param filepath: the path to the binary file containing a word embedding model trained by Word2Vec (``*.bin``).
        """
        logging.info('Word2Vec')
        logging.info('- model: {}'.format(filepath))
        self.model = KeyedVectors.load(filepath) if filepath.lower().endswith('.gnsm') else KeyedVectors.load_word2vec_format(filepath, binary=True)
        dim = self.model.syn0.shape[1]
        super().__init__(dim)
        logging.info('- vocab = %d, dim = %d' % (len(self.model.vocab), dim))

    # override
    def emb(self, value: str) -> np.ndarray:
        vocab = self.model.vocab.get(value, None)
        return self.model.syn0[vocab.index] if vocab else self.pad