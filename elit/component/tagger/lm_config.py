# -*- coding:utf-8 -*-
# Author：hankcs
# Date: 2018-09-27 20:29
from elit.component.dep.common.savable import Savable
from elit.component.tagger.corpus import Dictionary


class LanguageModelConfig(Savable):
    def __init__(self, dictionary: Dictionary, is_forward_lm, hidden_size, nlayers, embedding_size, nout,
                 dropout) -> None:
        self.dictionary = dictionary
        self.is_forward_lm = is_forward_lm
        self.hidden_size = hidden_size
        self.nlayers = nlayers
        self.embedding_size = embedding_size
        self.nout = nout
        self.dropout = dropout

    def to_dict(self) -> dict:
        d = vars(self).copy()
        d['dictionary'] = self.dictionary.to_dict()
        return d

    @staticmethod
    def from_dict(d: dict):
        d['dictionary'] = Dictionary.from_dict(d['dictionary'])
        return LanguageModelConfig(**d)
