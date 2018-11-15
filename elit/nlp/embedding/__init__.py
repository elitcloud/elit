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

__author__ = "Gary Lai"

from .base import Embedding
from .token import TokenEmbedding
from .fasttext import FastText
from .word2vec import Word2Vec
from .contextual_string_embedding import ContextualStringEmbedding

from typing import Union


def init_emb(config: list) -> Union[Word2Vec, FastText]:
    model, path = config
    if model.lower() == 'word2vec':
        emb = Word2Vec
    elif model.lower() == 'fasttext':
        emb = FastText
    else:
        raise TypeError('model {} is not supported'.format(model))
    return emb(path)
