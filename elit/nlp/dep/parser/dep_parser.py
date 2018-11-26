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
# Filename: biaffine_parser.py
# Author：hankcs
# Date: 2018-02-28 12:39
import argparse
import math
import os

import mxnet as mx
import numpy as np
from elit.nlp.dep.common.utils import init_logger, Progbar
from elit.nlp.dep.parser.biaffine_parser import BiaffineParser
from elit.nlp.dep.parser.common.data import ParserVocabulary, DataLoader, ConllWord, ConllSentence
from elit.nlp.dep.parser.common.exponential_scheduler import ExponentialScheduler
from elit.nlp.dep.parser.evaluate import evaluate_official_script
from elit.nlp.dep.parser.parser_config import ParserConfig
from elit.nlp.tagger.mxnet_util import mxnet_prefer_gpu
from mxnet import gluon, autograd


class DepParser(object):
    def __init__(self, config_file_path, context: mx.Context = None, extra_args=None) -> None:
        super().__init__()
        np.random.seed(666)
        self._config = ParserConfig(config_file_path, extra_args)
        self._parser = None
        self._vocab = None
        self.context = context if context else mxnet_prefer_gpu()

    @property
    def vocab(self) -> ParserVocabulary:
        return self._vocab

    def train(self):
        config = self._config
        logger = init_logger(config.save_dir)
        config.save()
        self._vocab = vocab = ParserVocabulary(config.train_file,
                                               config.pretrained_embeddings_file,
                                               config.min_occur_count)
        vocab.save(self._config.save_vocab_path)
        vocab.log_info(logger)

        with self.context:

            self._parser = parser = BiaffineParser(vocab, config.word_dims, config.tag_dims,
                                                   config.dropout_emb,
                                                   config.lstm_layers,
                                                   config.lstm_hiddens, config.dropout_lstm_input,
                                                   config.dropout_lstm_hidden,
                                                   config.mlp_arc_size,
                                                   config.mlp_rel_size, config.dropout_mlp, config.debug)
            parser.initialize()
            scheduler = ExponentialScheduler(config.learning_rate, config.decay, config.decay_steps)
            optimizer = mx.optimizer.Adam(config.learning_rate, config.beta_1, config.beta_2, config.epsilon,
                                          lr_scheduler=scheduler)
            trainer = gluon.Trainer(parser.collect_params(), optimizer=optimizer)
            data_loader = DataLoader(config.train_file, config.num_buckets_train, vocab)
            global_step = 0
            best_UAS = 0.
            batch_id = 0
            epoch = 1
            total_epoch = math.ceil(config.train_iters / config.validate_every)
            logger.info("Epoch {} out of {}".format(epoch, total_epoch))
            bar = Progbar(target=min(config.validate_every, data_loader.samples))
            while global_step < config.train_iters:
                for words, tags, arcs, rels in data_loader.get_batches(batch_size=config.train_batch_size,
                                                                       shuffle=True):
                    with autograd.record():
                        arc_accuracy, rel_accuracy, overall_accuracy, loss = parser.run(words, tags, arcs,
                                                                                        rels)
                        loss_value = loss.asscalar()
                    loss.backward()
                    trainer.step(config.train_batch_size)
                    batch_id += 1
                    try:
                        bar.update(batch_id,
                                   exact=[("UAS", arc_accuracy, 2),
                                          # ("LAS", rel_accuracy, 2),
                                          # ("ALL", overall_accuracy, 2),
                                          ("loss", loss_value)])
                    except OverflowError:
                        pass  # sometimes loss can be 0 or infinity, crashes the bar

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
        if not os.path.isfile(config.save_model_path) or best_UAS != UAS:
            parser.save(config.save_model_path)

        return self

    def load(self):
        config = self._config
        self._vocab = vocab = ParserVocabulary.load(config.save_vocab_path)
        with self.context:
            self._parser = BiaffineParser(vocab, config.word_dims, config.tag_dims, config.dropout_emb,
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
        with self.context:
            UAS, LAS, speed = evaluate_official_script(parser, vocab, config.num_buckets_valid, config.test_batch_size,
                                                       config.test_file, os.path.join(config.save_dir, 'valid_tmp'))
        if logger is None:
            logger = init_logger(config.save_dir, 'test.log')
        logger.info('Test: UAS %.2f%% LAS %.2f%% %d sents/s' % (UAS, LAS, speed))

        return UAS, LAS

    def parse(self, sentence: list):
        words = np.zeros((len(sentence) + 1, 1), np.int32)
        tags = np.zeros((len(sentence) + 1, 1), np.int32)
        words[0, 0] = ParserVocabulary.ROOT
        tags[0, 0] = ParserVocabulary.ROOT
        vocab = self.vocab

        for i, (word, tag) in enumerate(sentence):
            words[i + 1, 0], tags[i + 1, 0] = vocab.word2id(word.lower()), vocab.tag2id(tag)

        with self.context:
            outputs = self._parser.run(words, tags, is_train=False)
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
