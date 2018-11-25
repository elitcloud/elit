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
# -*- coding:utf-8 -*-
# Author：hankcs
# Date: 2018-09-27 20:29
from elit.nlp.dep.common.savable import Savable


class LanguageModelConfig(Savable):
    def __init__(self, dictionary, is_forward_lm, hidden_size, nlayers, embedding_size, nout, dropout) -> None:
        self.dictionary = dictionary
        self.is_forward_lm = is_forward_lm
        self.hidden_size = hidden_size
        self.nlayers = nlayers
        self.embedding_size = embedding_size
        self.nout = nout
        self.dropout = dropout
