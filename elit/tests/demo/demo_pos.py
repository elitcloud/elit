# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-05-28 17:38
from elit.component.tagger.pos_tagger import POSFlairTagger
from elit.component.tokenizer import EnglishTokenizer
from elit.resources.pre_trained_models import ELIT_POS_FLAIR_EN_MIXED
from elit.structure import Document

tagger = POSFlairTagger()
model_path = ELIT_POS_FLAIR_EN_MIXED
tagger.load(model_path)
print(tagger.tag('Is this the future of chamber music ?'.split()))
components = [EnglishTokenizer(), tagger]
docs = 'Is this the future of chamber music ?'
for c in components:
    docs = c.decode(docs)
for d in docs:  # type: Document
    print(d)
