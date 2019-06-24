# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-05-28 17:38
from elit.component.tagger.corpus import conll_to_documents
from elit.component.tagger.pos_tagger import POSFlairTagger
from elit.resources.pre_trained_models import POS_JUMBO
from elit.structure import SENS, POS, Document
from elit.tokenizer import EnglishTokenizer

tagger = POSFlairTagger()
model_path = POS_JUMBO
tagger.load(model_path)
print(tagger.tag('Is this the future of chamber music ?'.split()))
components = [EnglishTokenizer(), tagger]
docs = 'Is this the future of chamber music ?'
for c in components:
    docs = c.decode(docs)
for d in docs:  # type: Document
    print(d)