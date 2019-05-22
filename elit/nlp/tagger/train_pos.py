# -*- coding:utf-8 -*-
# Author：ported from PyTorch implementation of flair: https://github.com/zalandoresearch/flair to MXNet
# Date: 2018-09-19 15:12
from typing import List

from elit.nlp.tagger.corpus import NLPTaskDataFetcher
from elit.nlp.tagger.embeddings import TokenEmbeddings, WordEmbeddings, CharLMEmbeddings, StackedEmbeddings
from elit.nlp.tagger.mxnet_util import mxnet_prefer_gpu
from elit.nlp.tagger.sequence_tagger_model import SequenceTagger
from elit.nlp.tagger.sequence_tagger_trainer import SequenceTaggerTrainer
import mxnet as mx

if __name__ == '__main__':
    # data_folder = 'data/dat'
    data_folder = 'data/wsj-pos'

    # get training, test and dev data
    columns = {0: 'text', 1: 'pos'}
    corpus = NLPTaskDataFetcher.fetch_column_corpus(data_folder,
                                                    columns,
                                                    # train_file='en-pos.trn',
                                                    # test_file='en-pos.tst',
                                                    # dev_file='en-pos.dev',
                                                    train_file='train.tsv',
                                                    test_file='test.tsv',
                                                    dev_file='dev.tsv',
                                                    )

    # 2. what tag do we want to predict?
    tag_type = 'pos'

    # 3. make the tag dictionary from the corpus
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
    print(tag_dictionary.idx2item)

    # 4. initialize embeddings
    with mx.Context(mxnet_prefer_gpu()):
        embedding_types: List[TokenEmbeddings] = [
            WordEmbeddings('data/embedding/fasttext100.vec.txt'),
            CharLMEmbeddings('data/model/lm-news-forward'),
            CharLMEmbeddings('data/model/lm-news-backward'),
        ]

        embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

        # 5. initialize sequence tagger
        tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                                embeddings=embeddings,
                                                tag_dictionary=tag_dictionary,
                                                tag_type=tag_type,
                                                use_crf=True)

        # 6. initialize trainer
        trainer: SequenceTaggerTrainer = SequenceTaggerTrainer(tagger, corpus, test_mode=False)

        # 7. start training
        trainer.train('data/model/pos/wsj', learning_rate=0.1, mini_batch_size=32, max_epochs=150,
                      embeddings_in_memory=False)
