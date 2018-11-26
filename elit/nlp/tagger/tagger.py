# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2018-11-14 14:52
from typing import Sequence, List

import mxnet as mx

from elit.component import NLPComponent
from elit.nlp.tagger.corpus import NLPTaskDataFetcher
from elit.nlp.tagger.corpus import TaggedCorpus
from elit.nlp.tagger.embeddings import TokenEmbeddings, WordEmbeddings, CharLMEmbeddings, StackedEmbeddings
from elit.nlp.tagger.mxnet_util import mxnet_prefer_gpu
from elit.nlp.tagger.sequence_tagger_model import SequenceTagger
from elit.nlp.tagger.sequence_tagger_trainer import SequenceTaggerTrainer
from elit.structure import Document


class Tagger(NLPComponent):
    def __init__(self, context: mx.Context = None) -> None:
        super().__init__()
        self.tagger = None
        self.context = context if context else mxnet_prefer_gpu()

    def init(self, **kwargs):
        pass

    def load(self, model_path: str, **kwargs):
        self.tagger = SequenceTagger.load_from_file(model_path, context=self.context, **kwargs)

    def save(self, model_path: str, **kwargs):
        self.tagger.save(model_path)

    def _train(self, trn_docs: Sequence[Document], dev_docs: Sequence[Document], model_path: str,
               pretrained_embeddings,
               forward_language_model,
               backward_language_model,
               tag_type='ner',
               learning_rate: float = 0.1,
               mini_batch_size: int = 32,
               max_epochs: int = 100,
               anneal_factor: float = 0.5,
               patience: int = 2,
               save_model: bool = True,
               embeddings_in_memory: bool = True,
               train_with_dev: bool = False,
               **kwargs) -> float:
        corpus = TaggedCorpus(NLPTaskDataFetcher.convert_elit_documents(trn_docs),
                              NLPTaskDataFetcher.convert_elit_documents(dev_docs),
                              [])
        tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

        with mx.Context(self.context):
            embedding_types = [

                WordEmbeddings(pretrained_embeddings),

                # comment in this line to use character embeddings
                # CharacterEmbeddings(),

                # comment in these lines to use contextual string embeddings
                CharLMEmbeddings(forward_language_model, self.context),
                CharLMEmbeddings(backward_language_model, self.context),
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
