# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-05-28 17:38
from elit.component import NERTagger
from elit.component.tagger.corpus import conll_to_documents
from elit.component.tagger.pos_tagger import POSTagger
from elit.resources.pre_trained_models import POS_JUMBO
from elit.structure import SENS, POS, Document
from elit.tokenizer import EnglishTokenizer

tagger = NERTagger()
tagger.load()
components = [EnglishTokenizer(), tagger]
docs = 'buy Apple TV'
for c in components:
    docs = c.decode(docs)
for d in docs:  # type: Document
    print(d)
