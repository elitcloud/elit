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
from typing import Sequence, List

import mxnet as mx
from mxnet.gluon.data import DataLoader

from elit.component import MXComponent
from elit.nlp.embedding import Embedding
from elit.model import RNNModel
from elit.structure import Document

__author__ = "Gary Lai"


class RNNComponent(MXComponent):

    def __init__(self, ctx: mx.Context, key: str, embs: List[Embedding], rnn_config=None, output_config=None, **kwargs):
        super().__init__(ctx, key, embs, **kwargs)
        self.rnn_config = rnn_config
        self.output_config = output_config
        if rnn_config is not None and output_config is not None:
            self.model = RNNModel(rnn_config=rnn_config, output_config=output_config)
        else:
            self.model = None
        logging.info(self.__str__())

    def train_block(self, data_iter: DataLoader, docs: Sequence[Document]) -> float:
        pass

    def decode_block(self, data_iter: DataLoader, docs: Sequence[Document], **kwargs):
        pass

    def evaluate_block(self, data_iter: DataLoader, docs: Sequence[Document]):
        pass

    def data_loader(self, docs: Sequence[Document], batch_size, shuffle, **kwargs):
        pass

    def load(self, model_path: str, **kwargs):
        pass

    def save(self, model_path: str, **kwargs):
        pass
