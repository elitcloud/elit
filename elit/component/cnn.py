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
from types import SimpleNamespace
from typing import Sequence, Optional, Tuple, List

import mxnet as mx
from mxnet.gluon.data import DataLoader

from elit.component import MXComponent
from elit.nlp.embedding import Embedding
from elit.model import CNNModel
from elit.structure import Document

__author__ = "Gary Lai"


class CNNComponent(MXComponent):

    def __init__(self, ctx: mx.Context, key: str, embs: List[Embedding],
                 input_config: Optional[SimpleNamespace] = SimpleNamespace(dropout=0.0),
                 output_config: Optional[SimpleNamespace] = SimpleNamespace(num_class=1, flatten=False),
                 fuse_conv_config: Optional[SimpleNamespace] = None,
                 ngram_conv_config: Optional[SimpleNamespace] = None,
                 hidden_configs: Optional[Tuple[SimpleNamespace]] = None,
                 initializer: mx.init.Initializer = mx.init.Xavier(magnitude=2.24, rnd_type='gaussian'), **kwargs):
        super().__init__(ctx, key, embs, **kwargs)

        self.input_config = input_config
        self.output_config = output_config
        self.fuse_conv_config = fuse_conv_config
        self.ngram_conv_config = ngram_conv_config
        self.hidden_configs = hidden_configs
        self.initializer = initializer
        self.model = CNNModel(
            input_config=self.input_config,
            output_config=self.output_config,
            fuse_conv_config=self.fuse_conv_config,
            ngram_conv_config=self.ngram_conv_config,
            hidden_configs=self.hidden_configs,
            **kwargs)
        self.model.collect_params().initialize(self.initializer, ctx=self.ctx)
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
