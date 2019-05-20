# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-04-17 22:35
import glob

from iparsermodels import PTB_DEP
from iparser.parser.dep_parser import DepParser
from elit.nlp.tagger.corpus import conll_to_documents
from elit.structure import POS

parser = DepParser(PTB_DEP)
parser.load()

for file in glob.glob('data/tsv/*.pos.tsv'):
    print(file)
    with open(file.split('.')[0] + '.conll', 'w') as out:
        docs = conll_to_documents(file, headers={2: 'pos', 1: 'text'})
        for doc in docs:
            for sent in doc.sentences:
                tags = [(word, tag) for word, tag in zip(sent.tokens, sent[POS])]
                conll = parser.parse(tags)
                out.write(str(conll))
                out.write('\n\n')
