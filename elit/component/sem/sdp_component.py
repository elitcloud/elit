# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-06-08 18:28
from typing import Sequence

from elit.component.nlp import NLPComponent
from elit.component.sem.sdp_parser import BiaffineSDPParser
from elit.structure import Document


class SDPParser(NLPComponent):

    def __init__(self) -> None:
        super().__init__()
        self._parser = None  # type: BiaffineSDPParser

    def train(self, trn_docs: Sequence[Document], dev_docs: Sequence[Document], model_path: str, **kwargs) -> float:
        pass

    def decode(self, docs: Sequence[Document], **kwargs):
        pass

    def evaluate(self, docs: Sequence[Document], **kwargs):
        pass

    def load(self, model_path: str, **kwargs):
        pass

    def save(self, model_path: str, **kwargs):
        pass
