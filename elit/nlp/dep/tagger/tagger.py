# -*- coding:utf-8 -*-
# Filename: Tagger.py
# Author：hankcs
# Date: 2018-02-23 18:04
from elit.dep.common.utils import init_logger, Evaluator
from elit.dep.tagger.BiLSTM_CRF import tagger_from_vocab_config
from elit.dep.tagger.corpus import TSVCorpus, Vocabulary
from elit.dep.tagger.tagger_config import TaggerConfig


class Tagger(object):
    def __init__(self, config_file_path) -> None:
        super().__init__()
        self._tagger = None
        self._config = TaggerConfig(config_file_path)
        self._evaluator = Evaluator

    def train(self):
        config = self._config
        vocab = Vocabulary(config.lower_case, config.char_lstm_dim > 0)  # ready for training
        train_set = TSVCorpus(config.train_file, vocab)
        vocab.add_pret_words(config.pretrained_embeddings_file, keep_oov=config.keep_oov)
        self._tagger = tagger_from_vocab_config(vocab, config)
        self._tagger.evaluator = self._evaluator
        logger = init_logger(self._config.save_dir, 'train.log')
        self._tagger.train_on_config(train_set, self._config, logger)
        return self

    def evaluate(self, logger=None):
        test = TSVCorpus(self._config.test_file, self._tagger.vocab)
        acc = self._tagger.evaluate(test)[-1]
        if logger is None:
            logger = init_logger(self._config.save_dir, 'test.log')
        logger.info('Test score: %.2f%%' % acc)

    def load(self):
        config = self._config
        vocab = Vocabulary.load(config.save_vocab_path)
        self._tagger = tagger_from_vocab_config(vocab, config)
        self._tagger.load(self._config.save_model_path)
        self._tagger.evaluator = self._evaluator
        return self
