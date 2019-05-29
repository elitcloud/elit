# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-05-28 17:38
from elit.nlp.tagger.corpus import conll_to_documents
from elit.nlp.tagger.embeddings import StackedEmbeddings, WordEmbeddings, CharLMEmbeddings
from elit.nlp.tagger.ner_tagger import NERTagger
from elit.structure import SENS, NER

embedding_types = [
    WordEmbeddings(('fasttext', 'crawl-300d-2M-subword')),
    # CharLMEmbeddings('data/model/lm-news-forward'),
    # CharLMEmbeddings('data/model/lm-news-backward'),
]

embeddings = StackedEmbeddings(embeddings=embedding_types)
print(embeddings.to_list())

# tagger = NERTagger()
# model_path = 'data/model/conll-03'
# tagger.load(model_path)
# test = conll_to_documents('data/conll-03/bilou/eng.test.debug.tsv', headers={0: 'text', 1: 'ner'})
# sent = tagger.decode(test)[0][SENS][3]
# print(sent[NER])
# print(tagger.evaluate(test))
