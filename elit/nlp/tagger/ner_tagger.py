# -*- coding:utf-8 -*-
# Author：hankcs
# Date: 2018-09-27 21:03
import tempfile
from typing import Sequence

import mxnet as mx

from elit.component import NLPComponent
from elit.nlp.tagger.corpus import NLPTaskDataFetcher, conll_to_documents, get_chunks
from elit.nlp.tagger.corpus import TaggedCorpus
from elit.nlp.tagger.embeddings import WordEmbeddings, CharLMEmbeddings, StackedEmbeddings
from elit.nlp.tagger.mxnet_util import mxnet_prefer_gpu
from elit.nlp.tagger.sequence_tagger_model import SequenceTagger
from elit.nlp.tagger.sequence_tagger_trainer import SequenceTaggerTrainer
from elit.structure import Document, NER, SENS


class NERTagger(NLPComponent):
    def __init__(self) -> None:
        super().__init__()
        self.tagger = None

    def init(self, **kwargs):
        pass

    def load(self, model_path: str, **kwargs):
        self.tagger = SequenceTagger.load_from_file(model_path)

    def save(self, model_path: str, **kwargs):
        self.tagger.save(model_path)

    def train(self, trn_docs: Sequence[Document], dev_docs: Sequence[Document], model_path: str,
              learning_rate: float = 0.1,
              mini_batch_size: int = 32,
              max_epochs: int = 100,
              anneal_factor: float = 0.5,
              patience: int = 2,
              save_model: bool = True,
              embeddings_in_memory: bool = True,
              train_with_dev: bool = False, **kwargs) -> float:
        tag_type = 'ner'
        corpus = TaggedCorpus(NLPTaskDataFetcher.convert_elit_documents(trn_docs),
                              NLPTaskDataFetcher.convert_elit_documents(dev_docs),
                              [])
        tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

        with mx.Context(mxnet_prefer_gpu()):
            embedding_types = [

                WordEmbeddings('glove'),

                # comment in this line to use character embeddings
                # CharacterEmbeddings(),

                # comment in these lines to use contextual string embeddings
                CharLMEmbeddings('data/model/lm-news-forward'),
                CharLMEmbeddings('data/model/lm-news-backward'),
            ]

            embeddings = StackedEmbeddings(embeddings=embedding_types)

            self.tagger = SequenceTagger(hidden_size=256,
                                         embeddings=embeddings,
                                         tag_dictionary=tag_dictionary,
                                         tag_type=tag_type,
                                         use_crf=True)

            trainer = SequenceTaggerTrainer(self.tagger, corpus, test_mode=False)

            return trainer.train(model_path, learning_rate,
                                 mini_batch_size,
                                 max_epochs,
                                 anneal_factor,
                                 patience,
                                 save_model,
                                 embeddings_in_memory,
                                 train_with_dev)

    def decode(self, docs: Sequence[Document], **kwargs):
        samples = NLPTaskDataFetcher.convert_elit_documents(docs)
        sentences = self.tagger.predict(samples)
        idx = 0
        for d in docs:
            for s in d:
                s = s
                s[NER] = get_chunks([t.tags['ner'] for t in sentences[idx]])
                idx += 1
        return docs

    def evaluate(self, docs: Sequence[Document], **kwargs):
        print('test... ')
        with mx.Context(mxnet_prefer_gpu()):
            trainer = SequenceTaggerTrainer(self.tagger, corpus=None, test_mode=True)
            test_score, test_fp, test_result = trainer.evaluate(NLPTaskDataFetcher.convert_elit_documents(docs),
                                                                tempfile.gettempdir(),
                                                                evaluation_method='span-F1',
                                                                embeddings_in_memory=False)
        print('TEST   \t%d\t' % test_fp + test_result)
        return test_score


if __name__ == '__main__':
    tagger = NERTagger()
    model_path = 'data/model/ner/debug'
    tagger.train(conll_to_documents('data/conll-03/debug/eng.trn'), conll_to_documents('data/conll-03/debug/eng.dev'),
                 model_path, max_epochs=1)
    test = conll_to_documents('data/conll-03/debug/eng.tst')
    sent = tagger.decode(test)[0][SENS][3]
    print(sent[NER])
    tagger.evaluate(test)
