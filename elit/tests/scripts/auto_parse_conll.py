# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-06-16 11:37
from elit.component import POSFlairTagger
from elit.component.tagger.corpus import conll_to_documents
from elit.resources.pre_trained_models import POS_FLAIR_EN_MIXED

tagger = POSFlairTagger()
model_path = POS_FLAIR_EN_MIXED
tagger.load(model_path)
print(tagger.evaluate(conll_to_documents('data/dat/en-ddr.trn', headers={1: 'text', 3: 'pos'}), output_dir='data/dat',
                      dropout=0.7))  # 97.96%
