# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-05-28 17:38
from elit.component.tagger.corpus import conll_to_documents
from elit.component.tagger.ner_tagger import NERTagger
from elit.structure import SENS, NER

tagger = NERTagger()
model_path = 'data/model/ner/jumbo'
tagger.load(model_path)
test = conll_to_documents('data/dat/en-ner.debug.tsv', headers={0: 'text', 2: 'ner'})
sent = tagger.decode(test)[0][SENS][-2]
print(sent[NER])
print(tagger.evaluate(test))
