# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-05-28 17:38
import time

from elit.component.tagger.pos_tagger import POSTagger

tagger = POSTagger()
model_path = 'data/model/pos/jumbo'
tagger.load(model_path)
print(tagger.tag('Is this the future of chamber music ?'.split()))
