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
# Date: 2018-08-26 15:42
import math
import os
import tempfile
from typing import Sequence, Tuple

import mxnet as mx
from mxnet import gluon, autograd
import numpy as np
from elit.component.dep.common.config import _Config
from elit.component.dep.common.data import DataLoader, ParserVocabulary
from elit.component.dep.common.exponential_scheduler import ExponentialScheduler
from elit.component.dep.common.utils import init_logger, Progbar, _load_conll, fetch_resource
from elit.component.dep.parser.biaffine_parser import BiaffineParser
from elit.component.dep.parser.evaluate import evaluate_official_script
from elit.component.nlp import NLPComponent
from elit.component.tagger.mxnet_util import mxnet_prefer_gpu
from elit.resources.pre_trained_models import DEP_JUMBO
from elit.structure import Document, DEP
from elit.component.dep.common.conll import ConllWord, ConllSentence


class DependencyParser(NLPComponent):
    """
    An implementation of "Deep Biaffine Attention for Neural Dependency Parsing" Dozat and Manning (2016)
    """

    def __init__(self, context: mx.Context = None) -> None:
        super().__init__()
        self._parser = None  # type: BiaffineParser
        self._vocab = None  # type: ParserVocabulary
        self.context = context if context else mxnet_prefer_gpu()

    def train(self, trn_docs: Sequence[Document], dev_docs: Sequence[Document], save_dir, pretrained_embeddings=None,
              min_occur_count=2, lstm_layers=3, word_dims=100, tag_dims=100, dropout_emb=0.33, lstm_hiddens=400,
              dropout_lstm_input=0.33, dropout_lstm_hidden=0.33, mlp_arc_size=500, mlp_rel_size=100,
              dropout_mlp=0.33, learning_rate=2e-3, decay=.75, decay_steps=5000, beta_1=.9, beta_2=.9, epsilon=1e-12,
              num_buckets_train=40, num_buckets_valid=10, num_buckets_test=10, train_iters=50000, train_batch_size=5000,
              test_batch_size=5000, validate_every=100, save_after=5000, debug=False, **kwargs) -> float:

        logger = init_logger(save_dir)
        config = _Config(trn_docs, dev_docs, '', save_dir, pretrained_embeddings, min_occur_count,
                         lstm_layers, word_dims, tag_dims, dropout_emb, lstm_hiddens, dropout_lstm_input,
                         dropout_lstm_hidden, mlp_arc_size, mlp_rel_size, dropout_mlp, learning_rate, decay,
                         decay_steps,
                         beta_1, beta_2, epsilon, num_buckets_train, num_buckets_valid, num_buckets_test, train_iters,
                         train_batch_size, debug)
        config.save()
        self._vocab = vocab = ParserVocabulary(trn_docs,
                                               pretrained_embeddings,
                                               min_occur_count)
        vocab.save(config.save_vocab_path)
        vocab.log_info(logger)

        with mx.Context(mxnet_prefer_gpu()):

            self._parser = parser = BiaffineParser(vocab, word_dims, tag_dims,
                                                   dropout_emb,
                                                   lstm_layers,
                                                   lstm_hiddens, dropout_lstm_input,
                                                   dropout_lstm_hidden,
                                                   mlp_arc_size,
                                                   mlp_rel_size, dropout_mlp, debug)
            parser.initialize()
            scheduler = ExponentialScheduler(learning_rate, decay, decay_steps)
            optimizer = mx.optimizer.Adam(learning_rate, beta_1, beta_2, epsilon,
                                          lr_scheduler=scheduler)
            trainer = gluon.Trainer(parser.collect_params(), optimizer=optimizer)
            data_loader = DataLoader(trn_docs, num_buckets_train, vocab)
            global_step = 0
            best_UAS = 0.
            batch_id = 0
            epoch = 1
            total_epoch = math.ceil(train_iters / validate_every)
            logger.info("Epoch {} out of {}".format(epoch, total_epoch))
            bar = Progbar(target=min(validate_every, data_loader.samples))
            while global_step < train_iters:
                for words, tags, arcs, rels in data_loader.get_batches(batch_size=train_batch_size,
                                                                       shuffle=True):
                    with autograd.record():
                        arc_accuracy, rel_accuracy, overall_accuracy, loss = parser.forward(words, tags, arcs,
                                                                                            rels)
                        loss_value = loss.asscalar()
                    loss.backward()
                    trainer.step(train_batch_size)
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
                    if global_step % validate_every == 0:
                        bar = Progbar(target=min(validate_every, train_iters - global_step))
                        batch_id = 0
                        UAS, LAS, speed = evaluate_official_script(parser, vocab, num_buckets_valid,
                                                                   num_buckets_valid,
                                                                   dev_docs,
                                                                   os.path.join(save_dir, 'valid_tmp'))
                        logger.info('Dev: UAS %.2f%% LAS %.2f%% %d sents/s' % (UAS, LAS, speed))
                        epoch += 1
                        if global_step < train_iters:
                            logger.info("Epoch {} out of {}".format(epoch, total_epoch))
                        if global_step > save_after and UAS > best_UAS:
                            logger.info('- new best score!')
                            best_UAS = UAS
                            parser.save(config.save_model_path)

        # When validate_every is too big
        if not os.path.isfile(config.save_model_path) or best_UAS != UAS:
            parser.save(config.save_model_path)

        return best_UAS

    def decode(self, docs: Sequence[Document], num_buckets_test=10, test_batch_size=5000, **kwargs):
        if isinstance(docs, Document):
            docs = [docs]
        assert isinstance(docs, Sequence), 'Expect docs to be Sequence of Document'
        for d in docs:
            for s in d:
                s[DEP] = [(0, self._vocab.id2rel(0))] * len(s)  # placeholder
        data_loader = DataLoader(docs, num_buckets_test, self._vocab)
        record = data_loader.idx_sequence
        results = [None] * len(record)
        idx = 0
        parser = self._parser
        with self.context:
            for words, tags, arcs, rels in data_loader.get_batches(
                    batch_size=test_batch_size, shuffle=False):
                outputs = parser.forward(words, tags)
                for output in outputs:
                    sent_idx = record[idx]
                    results[sent_idx] = output
                    idx += 1
        idx = 0
        for d in docs:
            for s in d:
                s[DEP] = []
                for head, rel in zip(results[idx][0], results[idx][1]):
                    s[DEP].append((head, self._vocab.id2rel(rel)))
                idx += 1
        return docs

    def evaluate(self, test_file, save_dir=None, logger=None, num_buckets_test=10, test_batch_size=5000):
        """Run evaluation on test set

        Parameters
        ----------
        test_file : str or Sequence
            path to test set
        save_dir : str
            where to store intermediate results and log
        logger : logging.logger
            logger for printing results
        num_buckets_test : int
            number of clusters for sentences from test set
        test_batch_size : int
            batch size of test set

        Returns
        -------
        tuple
            UAS, LAS
        """
        parser = self._parser
        vocab = self._vocab
        if not save_dir:
            save_dir = tempfile.mkdtemp()
        with mx.Context(mxnet_prefer_gpu()):
            UAS, LAS, speed = evaluate_official_script(parser, vocab, num_buckets_test, test_batch_size,
                                                       test_file, os.path.join(save_dir, 'valid_tmp'))
        if logger is None:
            logger = init_logger(save_dir, 'test.log')
        logger.info('Test: UAS %.2f%% LAS %.2f%% %d sents/s' % (UAS, LAS, speed))

        return UAS, LAS, speed

    def load(self, path=DEP_JUMBO, **kwargs):
        """Load from disk

        Parameters
        ----------
        path : str
            path to the directory which typically contains a config.pkl file and a model.bin file

        Returns
        -------
        DepParser
            parser itself
            :param **kwargs:
        """
        path = fetch_resource(path)
        config = _Config.load(os.path.join(path, 'config.pkl'))
        config.save_dir = path  # redirect root path to what user specified
        self._vocab = vocab = ParserVocabulary.load(config.save_vocab_path)
        with mx.Context(mxnet_prefer_gpu()):
            self._parser = BiaffineParser(vocab, config.word_dims, config.tag_dims, config.dropout_emb,
                                          config.lstm_layers,
                                          config.lstm_hiddens, config.dropout_lstm_input, config.dropout_lstm_hidden,
                                          config.mlp_arc_size,
                                          config.mlp_rel_size, config.dropout_mlp, True)
            self._parser.load(config.save_model_path)
        return self

    def save(self, model_path: str, **kwargs):
        self._parser.save(model_path)
        self._vocab.save(os.path.join(model_path, 'vocab.pkl'))

    def init(self, **kwargs):
        pass

    def parse(self, sentence: Sequence[Tuple]) -> ConllSentence:
        """Parse raw sentence into ConllSentence

        Parameters
        ----------
        sentence : list
            a list of (word, tag) tuples

        Returns
        -------
        elit.component.dep.common.conll.ConllSentence
            ConllSentence object
        """
        words = np.zeros((len(sentence) + 1, 1), np.int32)
        tags = np.zeros((len(sentence) + 1, 1), np.int32)
        words[0, 0] = ParserVocabulary.ROOT
        tags[0, 0] = ParserVocabulary.ROOT
        vocab = self._vocab

        for i, (word, tag) in enumerate(sentence):
            words[i + 1, 0], tags[i + 1, 0] = vocab.word2id(word.lower()), vocab.tag2id(tag)

        with mx.Context(mxnet_prefer_gpu()):
            outputs = self._parser.forward(words, tags)
        words = []
        for arc, rel, (word, tag) in zip(outputs[0][0], outputs[0][1], sentence):
            words.append(ConllWord(id=len(words) + 1, form=word, pos=tag, head=arc, relation=vocab.id2rel(rel)))
        return ConllSentence(words)


if __name__ == '__main__':
    train = _load_conll('data/ptb/dep/train-debug.conllx')
    dev = _load_conll('data/ptb/dep/dev-debug.conllx')
    # _save_conll([dev], 'dev.conllx')
    parser = DependencyParser()
    model_path = 'data/model/ptb/dep-debug'
    parser.train([train], [dev], save_dir=model_path, train_iters=200,
                 pretrained_embeddings=('fasttext', 'crawl-300d-2M-subword'), debug=True)
    parser.load(model_path)
    parser.decode([dev])
    test = _load_conll('data/ptb/dep/test-debug.conllx')
    UAS, LAS, speed = parser.evaluate([test])
    print('UAS %.2f%% LAS %.2f%% %d sents/s' % (UAS, LAS, speed))
    sentence = [('Is', 'VBZ'), ('this', 'DT'), ('the', 'DT'), ('future', 'NN'), ('of', 'IN'), ('chamber', 'NN'),
                ('music', 'NN'), ('?', '.')]
    print(parser.parse(sentence))
