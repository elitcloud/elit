# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-04-17 22:35
import glob

from iparser.tagger.pos_tagger import POSTagger
from iparsermodels import PTB_POS

from elit.nlp.tagger.corpus import conll_to_documents

postagger = POSTagger(PTB_POS)
postagger.load()
# postagger.evaluate()
print(postagger.tag('Good boy .'.split()))

for file in glob.glob('data/tsv/*.tsv'):
    print(file)
    with open(file.split('.')[0] + '.pos.tsv', 'w') as out:
        docs = conll_to_documents(file, headers={0: 'pos', 1: 'text'})
        for doc in docs:
            for sent in doc.sentences:
                tags = postagger.tag(sent.tokens)
                for idx, (word, pos) in enumerate(tags):
                    out.write('{}\t{}\t{}\n'.format(idx, word, pos))
                out.write('\n')
