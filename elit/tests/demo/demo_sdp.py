# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-05-30 18:24
from elit.component import POSFlairTagger, SDPBiaffineParser
from elit.component.tokenizer import EnglishTokenizer

parser = SDPBiaffineParser()
parser.load()
pos_tagger = POSFlairTagger()
pos_tagger.load()
components = [EnglishTokenizer(), pos_tagger, parser]
docs = 'Is this the future of chamber music ?'
for c in components:
    docs = c.decode(docs)
for d in docs:  # type: Document
    for sent in d.to_conll():
        print(sent)
    print(d)
