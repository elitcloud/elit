# ========================================================================
# Copyright 2018 ELIT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========================================================================
# -*- coding:utf-8 -*-
# Author：hankcs
# Date: 2018-09-27 21:03
import tempfile
from typing import Sequence

import mxnet as mx

from elit.nlp.tagger.corpus import NLPTaskDataFetcher, conll_to_documents, get_chunks
from elit.nlp.tagger.mxnet_util import mxnet_prefer_gpu
from elit.nlp.tagger.sequence_tagger_trainer import SequenceTaggerTrainer
from elit.nlp.tagger.tagger import Tagger
from elit.structure import Document, NER, SENS


class NERTagger(Tagger):
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
                           'ner', learning_rate, mini_batch_size, max_epochs, anneal_factor, patience, save_model,
                           embeddings_in_memory, train_with_dev)

    def decode(self, docs: Sequence[Document], **kwargs):
        samples = NLPTaskDataFetcher.convert_elit_documents(docs)
        with self.context:
            sentences = self.tagger.predict(samples)
        idx = 0
        for d in docs:
            for s in d:
                s[NER] = get_chunks([t.tags['ner'] for t in sentences[idx]])
                idx += 1
        return docs

    def evaluate(self, docs: Sequence[Document], **kwargs):
        print('test... ')
        with self.context:
            trainer = SequenceTaggerTrainer(self.tagger, corpus=None, test_mode=True)
            test_score, test_fp, test_result = trainer.evaluate(NLPTaskDataFetcher.convert_elit_documents(docs),
                                                                tempfile.gettempdir(),
                                                                evaluation_method='span-F1',
                                                                embeddings_in_memory=False)
        print('TEST   \t%d\t' % test_fp + test_result)
        return test_score


if __name__ == '__main__':
    tagger = NERTagger(mx.gpu(3))
    model_path = 'data/model/ner/jumbo'
    # tagger.train(conll_to_documents('data/conll-03/debug/eng.trn'), conll_to_documents('data/conll-03/debug/eng.dev'),
    #              model_path, pretrained_embeddings='data/embedding/glove/glove.6B.100d.debug.txt',
    #              forward_language_model='data/model/lm-news-forward',
    #              backward_language_model='data/model/lm-news-backward',
    #              max_epochs=1)
    tagger.load(model_path)
    test = conll_to_documents('data/dat/en-ner.tst', headers={0: 'text', 1: 'pos', 2: 'ner'})
    sent = tagger.decode(test)[0][SENS][3]
    print(sent[NER])
    print(tagger.evaluate(test))
