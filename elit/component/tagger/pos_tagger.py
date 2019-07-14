# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2018-11-14 10:42
import tempfile
from typing import Sequence, Union

import mxnet as mx

from elit.component.tagger.corpus import NLPTaskDataFetcher, conll_to_documents, Sentence, Token
from elit.component.tagger.sequence_tagger_trainer import SequenceTaggerTrainer
from elit.component.tagger.tagger import Tagger
from elit.resources.pre_trained_models import ELIT_POS_FLAIR_EN_MIXED
from elit.structure import Document, POS, SENS


class POSFlairTagger(Tagger):
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
        """
        Train a pos tagger
        :param trn_docs: training set
        :param dev_docs: dev set
        :param model_path: the path to store trained model
        :param pretrained_embeddings: which pretrained embeddings to use, see https://gluon-nlp.mxnet.io/examples/word_embedding/word_embedding.html
        :param forward_language_model: which forward language model to use
        :param backward_language_model: which backward language mdoel to use
        :param learning_rate: learning rate
        :param mini_batch_size: mini batch size
        :param max_epochs: max epochs
        :param anneal_factor: anneal factor for learning rate
        :param patience: early top after how many epochs
        :param save_model: whether save model or not
        :param embeddings_in_memory: whether put embeddings in GPU memory or not
        :param train_with_dev: merge dev set with training set
        :param kwargs: not used
        :return:
        """
        return self._train(trn_docs, dev_docs, model_path, pretrained_embeddings, forward_language_model,
                           backward_language_model,
                           'pos', learning_rate, mini_batch_size, max_epochs, anneal_factor, patience, save_model,
                           embeddings_in_memory, train_with_dev)

    def decode(self, docs: Sequence[Document], **kwargs):
        """
        Decode documents
        :param docs: list of documents
        :param kwargs: not used
        :return: documents passed in
        """
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
        """
        Evaluate this tagger
        :param docs: test set
        :param dropout: dropout in test phase, for simulating noise on training set
        :param output_dir: the folder to store test output
        :param kwargs: not used
        :return: accuracy
        """
        print('test... ')
        with self.context:
            trainer = SequenceTaggerTrainer(self.tagger, corpus=None, test_mode=True)
            test_score, _, _ = trainer.evaluate(NLPTaskDataFetcher.convert_elit_documents(docs),
                                                output_dir,
                                                evaluation_method='accuracy',
                                                embeddings_in_gpu=False, dropout=dropout)
        print('Accuracy: %.2f%%' % (test_score * 100))
        return test_score

    def tag(self, tokens: Union[Sequence[str], Sequence[Sequence[str]]]):
        """
        Shortcut for tagging
        :param tokens: text
        :return: tags
        """
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

    def load(self, model_path: str = ELIT_POS_FLAIR_EN_MIXED, model_root=None, **kwargs):
        """
        Load model
        :param model_path: path to stored model
        :param model_root: the root for model_path
        :param kwargs: not used
        :return: self
        """
        super().load(model_path, model_root=model_root, **kwargs)
        return self


if __name__ == '__main__':
    tagger = POSFlairTagger(context=mx.gpu(3))
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
