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
