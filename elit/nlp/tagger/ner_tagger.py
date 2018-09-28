# -*- coding:utf-8 -*-
# Author：hankcs
# Date: 2018-09-27 21:03
from typing import Sequence

from elit.component import NLPComponent
from elit.util.structure import Document


class NERTagger(NLPComponent):

    def __init__(self) -> None:
        super().__init__()

    def train(self, trn_docs: Sequence[Document], dev_docs: Sequence[Document], model_path: str, **kwargs) -> float:
        pass

    def decode(self, docs: Sequence[Document], **kwargs):
        pass

    def evaluate(self, docs: Sequence[Document], **kwargs):
        pass

