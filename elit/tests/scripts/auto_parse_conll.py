# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-06-16 11:37
from elit.component import POSTagger
from elit.component.tagger.corpus import conll_to_documents
from elit.resources.pre_trained_models import POS_JUMBO

tagger = POSTagger()
model_path = POS_JUMBO
tagger.load(model_path)
print(tagger.evaluate(conll_to_documents('data/dat/en-ddr.tst', headers={1: 'text', 3: 'pos'})))
