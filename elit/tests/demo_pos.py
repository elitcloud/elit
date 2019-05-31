# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-05-28 17:38

from elit.component.tagger.pos_tagger import POSTagger
from elit.resources.pre_trained_models import POS_JUMBO

tagger = POSTagger()
model_path = POS_JUMBO
tagger.load(model_path)
print(tagger.tag('Is this the future of chamber music ?'.split()))
