# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2018-11-14 10:42
import tempfile
from typing import Sequence, Union

import mxnet as mx

from elit.component.tagger.corpus import NLPTaskDataFetcher, conll_to_documents, Sentence, Token
from elit.component.tagger.mxnet_util import mxnet_prefer_gpu
from elit.component.tagger.sequence_tagger_trainer import SequenceTaggerTrainer
from elit.component.tagger.tagger import Tagger
from elit.resources.pre_trained_models import POS_JUMBO
from elit.structure import Document, POS, SENS


class POSTagger(Tagger):
    def train(self, trn_docs: Sequence[Document], dev_docs: Sequence[Document], model_path: str,
              pretrained_embeddings,
              forward_language_model=None,
              backward_language_model=None,
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
        if isinstance(docs, Document):
            docs = [docs]
        samples = NLPTaskDataFetcher.convert_elit_documents(docs)
        with self.context:
            sentences = self.tagger.predict(samples)
        idx = 0
        for d in docs:
            for s in d:
                s[POS] = [t.tags['pos'] for t in sentences[idx]]
                idx += 1
        return docs

    def evaluate(self, docs: Sequence[Document], dropout=0, output_dir=tempfile.gettempdir(), **kwargs):
        print('test... ')
        with self.context:
            trainer = SequenceTaggerTrainer(self.tagger, corpus=None, test_mode=True)
            test_score, _, _ = trainer.evaluate(NLPTaskDataFetcher.convert_elit_documents(docs),
                                                output_dir,
                                                evaluation_method='accuracy',
                                                embeddings_in_memory=False, dropout=dropout)
        print('Accuracy: %.2f%%' % (test_score * 100))
        return test_score

    def tag(self, tokens: Union[Sequence[str], Sequence[Sequence[str]]]):
        if len(tokens) == 0:
            return []
        if isinstance(tokens[0], str):
            tokens = [tokens]
        samples = []
        for sent in tokens:  # type:Sentence
            sentence = Sentence()
            for word in sent:
                t = Token(word)
                sentence.add_token(t)
            samples.append(sentence)
        with self.context:
            sentences = self.tagger.predict(samples)
            results = []
            for sent in sentences:
                results.append([(word.text, word.pos) for word in sent.tokens])
            if len(results) == 1:
                results = results[0]
            return results

    def load(self, model_path: str = POS_JUMBO, **kwargs):
        super().load(model_path, **kwargs)
        return self


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
    test = conll_to_documents('data/wsj-pos/test.tsv', headers={0: 'text', 1: 'pos'})
    sent = tagger.decode(test)[0][SENS][3]
    print(sent[POS])
    print(tagger.evaluate(test))
