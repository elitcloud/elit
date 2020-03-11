# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-11-02 12:16
from elit.resources.pre_trained_models import ELIT_POS_FLAIR_EN_MIXED

from elit.component import EnglishTokenizer, POSFlairTagger
from elit.structure import Document

from elit.component.lemmatizer import EnglishLemmatizer

tagger = POSFlairTagger()
model_path = ELIT_POS_FLAIR_EN_MIXED
tagger.load(model_path)
components = [EnglishTokenizer(), tagger, EnglishLemmatizer()]
docs = ['Sentence one. Sentence two.']
for c in components:
    docs = c.decode(docs)
for d in docs:  # type: Document
    print(d)
