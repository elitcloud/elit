# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-05-28 17:38
from elit.component import NERFlairTagger
from elit.component.tokenizer import EnglishTokenizer
from elit.structure import Document

tagger = NERFlairTagger()
tagger.load()
components = [EnglishTokenizer(), tagger]
docs = 'buy Apple TV'
for c in components:
    docs = c.decode(docs)
for d in docs:  # type: Document
    print(d)
