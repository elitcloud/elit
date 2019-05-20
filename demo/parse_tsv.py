# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-04-17 22:35
import glob

from elit.nlp.tagger.corpus import conll_to_documents
from elit.nlp.tagger.pos_tagger import POSTagger
import mxnet as mx

from elit.structure import POS

tagger = POSTagger(context=mx.gpu(2))
model_path = 'data/model/pos/jumbo'
tagger.load(model_path)
for file in glob.glob('data/tsv/*.tsv'):
    print(file)
    with open(file.split('.')[0] + '.pos.tsv', 'w') as out:
        test = conll_to_documents(file, headers={0: 'pos', 1: 'text'})
        docs = tagger.decode(test)
        for doc in docs:
            for sent in doc.sentences:
                for idx, (word, pos) in enumerate(zip(sent.tokens, sent[POS])):
                    out.write('{}\t{}\t{}\n'.format(idx, word, pos))
                out.write('\n')
