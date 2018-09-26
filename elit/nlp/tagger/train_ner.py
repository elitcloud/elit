# -*- coding:utf-8 -*-
# Author：hankcs
# Date: 2018-09-19 15:12
from typing import List

from elit.nlp.tagger.corpus import NLPTaskDataFetcher
from elit.nlp.tagger.embeddings import TokenEmbeddings, WordEmbeddings, CharLMEmbeddings, StackedEmbeddings
from elit.nlp.tagger.sequence_tagger_model import SequenceTagger
from elit.nlp.tagger.sequence_tagger_trainer import SequenceTaggerTrainer

if __name__ == '__main__':
    # use your own data path
    data_folder = 'data/conll-03/debug'

    # get training, test and dev data
    columns = {0: 'text', 1: 'pos', 2: 'np', 3: 'ner'}
    corpus = NLPTaskDataFetcher.fetch_column_corpus(data_folder,
                                                    columns,
                                                    train_file='eng.trn',
                                                    test_file='eng.tst',
                                                    dev_file='eng.dev',
                                                    tag_to_biloes='ner')
    # 2. what tag do we want to predict?
    tag_type = 'ner'

    # 3. make the tag dictionary from the corpus
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
    print(tag_dictionary.idx2item)

    # 4. initialize embeddings
    embedding_types: List[TokenEmbeddings] = [

        WordEmbeddings('glove'),

        # comment in this line to use character embeddings
        # CharacterEmbeddings(),

        # comment in these lines to use contextual string embeddings
        # CharLMEmbeddings('news-forward'),
        # CharLMEmbeddings('news-backward'),
    ]

    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

    # 5. initialize sequence tagger
    tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                            embeddings=embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type=tag_type,
                                            use_crf=True)

    # 6. initialize trainer
    trainer: SequenceTaggerTrainer = SequenceTaggerTrainer(tagger, corpus, test_mode=True)

    # 7. start training
    trainer.train('resources/taggers/example-ner',
                  learning_rate=0.1,
                  mini_batch_size=32,
                  max_epochs=150)
