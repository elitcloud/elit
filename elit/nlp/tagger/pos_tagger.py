# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2018-11-14 10:42
import tempfile
from typing import Sequence

import mxnet as mx

from elit.nlp.tagger.corpus import NLPTaskDataFetcher, conll_to_documents
from elit.nlp.tagger.mxnet_util import mxnet_prefer_gpu
from elit.nlp.tagger.sequence_tagger_trainer import SequenceTaggerTrainer
from elit.nlp.tagger.tagger import Tagger
from elit.structure import Document, POS


class POSTagger(Tagger):
    def train(self, trn_docs: Sequence[Document], dev_docs: Sequence[Document], model_path: str,
              pretrained_embeddings,
              forward_language_model,
              backward_language_model,
              learning_rate: float = 0.1,
              mini_batch_size: int = 32,
              max_epochs: int = 100,
              anneal_factor: float = 0.5,
              patience: int = 2,
              save_model: bool = True,
              embeddings_in_memory: bool = True,
              train_with_dev: bool = False,
              **kwargs) -> float:
        return self._train(trn_docs, dev_docs, model_path, pretrained_embeddings, forward_language_model,
                           backward_language_model,
                           'pos', learning_rate, mini_batch_size, max_epochs, anneal_factor, patience, save_model,
                           embeddings_in_memory, train_with_dev)

    def decode(self, docs: Sequence[Document], **kwargs):
        samples = NLPTaskDataFetcher.convert_elit_documents(docs)
        with self.context:
            sentences = self.tagger.predict(samples)
        idx = 0
        for d in docs:
            for s in d:
                s[POS] = [t.tags['pos'] for t in sentences[idx]]
                idx += 1
        return docs

    def evaluate(self, docs: Sequence[Document], **kwargs):
        print('test... ')
        with self.context:
            trainer = SequenceTaggerTrainer(self.tagger, corpus=None, test_mode=True)
            test_score, _, _ = trainer.evaluate(NLPTaskDataFetcher.convert_elit_documents(docs),
                                                tempfile.gettempdir(),
                                                evaluation_method='accuracy',
                                                embeddings_in_memory=False)
        print('TEST   \t%d\t' % test_score)
        return test_score


if __name__ == '__main__':
    tagger = POSTagger(context=mx.gpu(3))
    model_path = 'data/model/pos/wsj'
    tagger.load(model_path)
    # tagger.train(conll_to_documents('data/dat/en-pos.dev', headers={0: 'text', 1: 'pos'}),
    #              conll_to_documents('data/dat/en-pos.dev', headers={0: 'text', 1: 'pos'}),
    #              model_path, pretrained_embeddings='data/embedding/glove/glove.6B.100d.debug.txt',
    #              forward_language_model='data/model/lm-news-forward',
    #              backward_language_model='data/model/lm-news-backward',
    #              max_epochs=1,
    #              embeddings_in_memory=False)
    test = conll_to_documents('data/dat/en-pos.tst', headers={0: 'text', 1: 'pos'})
    # sent = tagger.decode(test)[0][SENS][3]
    # print(sent[POS])
    print(tagger.evaluate(test))
