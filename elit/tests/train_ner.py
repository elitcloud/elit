# -*- coding:utf-8 -*-
# Author：ported from PyTorch implementation of flair: https://github.com/zalandoresearch/flair to MXNet
# Date: 2018-09-19 15:12
from typing import List

from elit.component.tagger.corpus import NLPTaskDataFetcher
from elit.component.tagger.embeddings import TokenEmbeddings, WordEmbeddings, CharLMEmbeddings, StackedEmbeddings
from elit.util.mx import mxnet_prefer_gpu
from elit.component.tagger.sequence_tagger_model import SequenceTagger
from elit.component.tagger.sequence_tagger_trainer import SequenceTaggerTrainer
import mxnet as mx

if __name__ == '__main__':
    # use your own data path
    # data_folder = 'data/conll-03/debug'
    # data_folder = 'data/conll-03'
    data_folder = 'data/dat'

    # get training, test and dev data
    columns = {0: 'text', 1: 'pos', 2: 'ner'}
    corpus = NLPTaskDataFetcher.fetch_column_corpus(data_folder,
                                                    columns,
                                                    train_file='en-ner.trn',
                                                    test_file='en-ner.tst',
                                                    dev_file='en-ner.dev',
                                                    tag_to_biloes='ner',
                                                    source_scheme='ioblu')
    # 2. what tag do we want to predict?
    tag_type = 'ner'

    # 3. make the tag dictionary from the corpus
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
    print(tag_dictionary.idx2item)

    # 4. initialize embeddings
    with mx.Context(mxnet_prefer_gpu()):
        embedding_types: List[TokenEmbeddings] = [

            WordEmbeddings('data/embedding/fasttext100.vec.txt'),

            # comment in this line to use character embeddings
            # CharacterEmbeddings(),

            # comment in these lines to use contextual string embeddings
            CharLMEmbeddings('data/model/lm-news-forward'),
            CharLMEmbeddings('data/model/lm-news-backward'),
            # CharLMEmbeddings('data/model/lm-jumbo-forward256'),
            # CharLMEmbeddings('data/model/lm-jumbo-backward256'),
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
        trainer.train('data/model/ner/jumbo',
                      learning_rate=0.1,
                      mini_batch_size=32,
                      max_epochs=150,
                      embeddings_in_memory=False)
