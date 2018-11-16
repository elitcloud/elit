# -*- coding:utf-8 -*-
# Filename: biaffine_parser.py
# Author：hankcs
# Date: 2018-02-28 12:39
import argparse
import os

import math
import numpy as np

from elit.dep.common.utils import file_newer, init_logger, Progbar, make_sure_path_exists
from elit.dep.parser.biaffine_parser import BiaffineParser
from elit.dep.parser.common import ParserVocabulary, DataLoader, sys, ConllWord, ConllSentence, get_word_id
from elit.dep.parser.evaluate import evaluate_official_script
from elit.dep.parser.parser_config import ParserConfig
from elit.dep.common.utils import stdchannel_redirected

with stdchannel_redirected(sys.stderr, os.devnull):
    import dynet as dy


class DepParser(object):
    def __init__(self, config_file_path, extra_args=None) -> None:
        super().__init__()
        self._config = ParserConfig(config_file_path, extra_args)
        self._parser = None
        self._vocab = None

    @property
    def vocab(self) -> ParserVocabulary:
        return self._vocab

    def train(self):
        config = self._config
        self._vocab = vocab = ParserVocabulary(config.train_file,
                                               None if config.debug else config.pretrained_embeddings_file,
                                               config.min_occur_count)
        make_sure_path_exists(config.save_dir)
        vocab.save(self._config.save_vocab_path)
        logger = init_logger(config.save_dir)
        vocab.log_info(logger)
        self._parser = parser = BiaffineParser(vocab, config.char_dims, config.word_dims, config.tag_dims,
                                               config.dropout_emb,
                                               config.lstm_layers,
                                               config.lstm_hiddens, config.dropout_lstm_input,
                                               config.dropout_lstm_hidden,
                                               config.mlp_arc_size,
                                               config.mlp_rel_size, config.dropout_mlp, config.debug)

        data_loader = DataLoader(config.train_file, config.num_buckets_train, vocab)
        pc = parser.parameter_collection
        trainer = dy.AdamTrainer(pc, config.learning_rate, config.beta_1, config.beta_2, config.epsilon)

        global_step = 0

        def update_parameters():
            trainer.learning_rate = config.learning_rate * config.decay ** (global_step / config.decay_steps)
            trainer.update()

        best_UAS = 0.
        batch_id = 0
        epoch = 1
        total_epoch = math.ceil(config.train_iters / config.validate_every)
        logger.info("Epoch {} out of {}".format(epoch, total_epoch))
        bar = Progbar(target=min(config.validate_every, data_loader.samples))
        while global_step < config.train_iters:
            for chars, cased_words, words, tags, arcs, rels in data_loader.get_batches(
                    batch_size=config.train_batch_size,
                    shuffle=not config.debug):
                batch_id += 1
                dy.renew_cg()
                arc_accuracy, rel_accuracy, overall_accuracy, loss = parser.run(chars, cased_words, words, tags, arcs,
                                                                                rels)
                loss_value = loss.scalar_value()
                loss.backward()
                try:
                    bar.update(batch_id,
                               exact=[("UAS", arc_accuracy, 2),
                                      # ("LAS", rel_accuracy, 2),
                                      # ("ALL", overall_accuracy, 2),
                                      ("loss", loss_value)])
                except OverflowError:
                    pass  # sometimes loss can be 0 or infinity, crashes the bar

                update_parameters()

                global_step += 1
                if global_step % config.validate_every == 0:
                    bar = Progbar(target=min(config.validate_every, config.train_iters - global_step))
                    batch_id = 0
                    UAS, LAS, speed = evaluate_official_script(parser, vocab, config.num_buckets_valid,
                                                               config.test_batch_size,
                                                               config.dev_file,
                                                               os.path.join(config.save_dir, 'valid_tmp'))
                    logger.info('Dev: UAS %.2f%% LAS %.2f%% %d sents/s' % (UAS, LAS, speed))
                    epoch += 1
                    if global_step < config.train_iters:
                        logger.info("Epoch {} out of {}".format(epoch, total_epoch))
                    if global_step > config.save_after and UAS > best_UAS:
                        logger.info('- new best score!')
                        best_UAS = UAS
                        parser.save(config.save_model_path)

        # When validate_every is too big
        if not os.path.isfile(config.save_model_path):
            parser.save(config.save_model_path)

        return self

    def load(self):
        config = self._config
        self._vocab = vocab = ParserVocabulary.load(config.save_vocab_path)
        self._parser = BiaffineParser(vocab, config.char_dims, config.word_dims, config.tag_dims, config.dropout_emb,
                                      config.lstm_layers,
                                      config.lstm_hiddens, config.dropout_lstm_input, config.dropout_lstm_hidden,
                                      config.mlp_arc_size,
                                      config.mlp_rel_size, config.dropout_mlp, config.debug)
        self._parser.load(config.save_model_path)
        return self

    def evaluate(self, logger=None):
        parser = self._parser
        vocab = self._vocab
        config = self._config
        UAS, LAS, speed = evaluate_official_script(parser, vocab, config.num_buckets_valid, config.test_batch_size,
                                                   config.dev_file, os.path.join(config.save_dir, 'valid_tmp'))
        if logger is None:
            logger = init_logger(config.save_dir, 'test.log')
        logger.info('UAS %.2f%% LAS %.2f%% %d sents/s' % (UAS, LAS, speed))

        return UAS, LAS

    def parse(self, sentence: list):
        words = np.zeros((len(sentence) + 1, 1), np.int32)
        cased_words = np.zeros((len(sentence) + 1, 1), np.int32)
        tags = np.zeros((len(sentence) + 1, 1), np.int32)
        words[0, 0] = ParserVocabulary.ROOT
        cased_words[0, 0] = ParserVocabulary.ROOT
        tags[0, 0] = ParserVocabulary.ROOT
        vocab = self.vocab

        cased_w2i = dict()
        cased_i2w = []
        get_word_id('\0', cased_w2i, cased_i2w)
        get_word_id('\1', cased_w2i, cased_i2w)
        get_word_id('\2', cased_w2i, cased_i2w)

        for i, (word, tag) in enumerate(sentence):
            cased_words[i + 1, 0], words[i + 1, 0], tags[i + 1, 0] = get_word_id(word, cased_w2i, cased_i2w), \
                                                                     vocab.word2id(word.lower()), vocab.tag2id(tag)
        char_vocab = []
        for word in cased_i2w:
            char_vocab.append([self.vocab.char2id(char) for char in word])

        outputs = self._parser.run(char_vocab, cased_words, words, tags, is_train=False)
        words = []
        for arc, rel, (word, tag) in zip(outputs[0][0], outputs[0][1], sentence):
            words.append(ConllWord(id=len(words) + 1, form=word, pos=tag, head=arc, relation=vocab.id2rel(rel)))
        return ConllSentence(words)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config_file', default='data/ptb/dep/config-debug.ini')
    args, extra_args = arg_parser.parse_known_args()
    parser = DepParser(args.config_file, extra_args)
    parser.train()
    parser.load()
    parser.evaluate()
    sentence = [('Is', 'VBZ'), ('this', 'DT'), ('the', 'DT'), ('future', 'NN'), ('of', 'IN'), ('chamber', 'NN'),
                ('music', 'NN'), ('?', '.')]
    print(parser.parse(sentence))
