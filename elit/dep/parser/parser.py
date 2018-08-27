# -*- coding:utf-8 -*-
# Author：hankcs
# Date: 2018-08-26 15:42
import math
import os
import sys
from typing import Sequence, Tuple

from elit.component import NLPComponent
from elit.dep.common.utils import init_logger, stdchannel_redirected, Progbar
from elit.dep.parser import DEFAULT_CONFIG_FILE
from elit.dep.parser.biaffine_parser import BiaffineParser
from elit.dep.parser.common import ParserVocabulary, DataLoader, np, get_word_id, ConllSentence, ConllWord
from elit.dep.parser.evaluate import evaluate_official_script
from elit.dep.parser.parser_config import ParserConfig
from elit.util.structure import Document, Sentence, HEA, DEP, POS, SEN

with stdchannel_redirected(sys.stderr, os.devnull):
    import dynet as dy


class DependencyParser(NLPComponent):
    """
    An implementation of "Deep Biaffine Attention for Neural Dependency Parsing" Dozat and Manning (2016)
    """

    def __init__(self) -> None:
        super().__init__()
        self._config = None  # type: ParserConfig
        self._vocab = None  # type: ParserVocabulary
        self._parser = None  # type: BiaffineParser

    def train(self, trn_docs: Sequence[Document], dev_docs: Sequence[Document], model_path: str, **kwargs) -> float:
        # read config file
        config_file = kwargs.get('config_file', DEFAULT_CONFIG_FILE)
        self._config = config = ParserConfig(config_file, kwargs)
        config.save_dir = model_path
        config.save()
        # prepare vocabulary and logger
        self._vocab = vocab = ParserVocabulary(config.train_file,
                                               None if config.debug else config.pretrained_embeddings_file,
                                               config.min_occur_count, documents=trn_docs)
        self._vocab.save(self._config.save_vocab_path)
        logger = init_logger(config.save_dir)
        vocab.log_info(logger)
        # training
        self._parser = parser = self._create_parser(config, vocab)

        data_loader = DataLoader(config.train_file, config.num_buckets_train, vocab, documents=trn_docs)
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
                    UAS, LAS = evaluate_official_script(parser, vocab, config.num_buckets_valid, config.test_batch_size,
                                                        config.dev_file,
                                                        os.path.join(config.save_dir, 'score.txt'), documents=dev_docs)
                    logger.info('Dev: UAS %.2f%% LAS %.2f%%' % (UAS, LAS))
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

        return best_UAS

    def decode(self, docs: Sequence[Document], **kwargs):
        assert isinstance(docs, Sequence), 'Expect docs to be Sequence of Document'
        data_loader = DataLoader(None, self._config.num_buckets_test, self._vocab, docs)
        record = data_loader.idx_sequence
        results = [None] * len(record)
        idx = 0
        for chars, cased_words, words, tags, arcs, rels in data_loader.get_batches(
                batch_size=self._config.test_batch_size, shuffle=False):
            outputs = self._parser.run(chars, cased_words, words, tags, is_train=False)
            for output in outputs:
                sent_idx = record[idx]
                results[sent_idx] = output
                idx += 1
        idx = 0
        for d in docs:
            for s in d:
                s[HEA] = []
                s[DEP] = []
                for head, rel in zip(results[idx][0], results[idx][1]):
                    s[HEA].append(head)
                    s[DEP].append(self._vocab.id2rel(rel))
                idx += 1

    def evaluate(self, docs: Sequence[Document], **kwargs):
        """
        Evaluation on test set
        :param docs: gold test set
        :param kwargs: None
        :return: (UAS, LAS)
        """
        UAS, LAS = evaluate_official_script(self._parser, self._vocab, self._config.num_buckets_valid,
                                            self._config.test_batch_size,
                                            self._config.dev_file,
                                            None, documents=docs)
        return UAS, LAS

    def load(self, model_path: str, **kwargs):
        self._config = ParserConfig(os.path.join(model_path, 'config.ini'))
        self._vocab = ParserVocabulary.load(self._config.save_vocab_path)
        self._parser = self._create_parser(self._config, self._vocab)
        pass

    def save(self, model_path: str, **kwargs):
        self._config.save_dir = model_path
        self._config.save()
        self._vocab.save(self._config.save_vocab_path)
        self._parser.save(self._config.save_model_path)

    def init(self, **kwargs):
        pass

    def parse(self, sentence: Sequence[Tuple]) -> ConllSentence:
        """
        Parse raw sentence.

        :param sentence: A list of (word, tag) pair. Both word and pair are raw strings
        :return: A CoNLLSentence object
        """
        words = np.zeros((len(sentence) + 1, 1), np.int32)
        cased_words = np.zeros((len(sentence) + 1, 1), np.int32)
        tags = np.zeros((len(sentence) + 1, 1), np.int32)
        words[0, 0] = ParserVocabulary.ROOT
        cased_words[0, 0] = ParserVocabulary.ROOT
        tags[0, 0] = ParserVocabulary.ROOT
        vocab = self._vocab

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
            char_vocab.append([self._vocab.char2id(char) for char in word])

        outputs = self._parser.run(char_vocab, cased_words, words, tags, is_train=False)
        words = []
        for arc, rel, (word, tag) in zip(outputs[0][0], outputs[0][1], sentence):
            words.append(ConllWord(id=len(words) + 1, form=word, pos=tag, head=arc, relation=vocab.id2rel(rel)))
        return ConllSentence(words)

    def _create_parser(self, config, vocab):
        return BiaffineParser(vocab, config.char_dims, config.word_dims, config.tag_dims,
                              config.dropout_emb,
                              config.lstm_layers,
                              config.lstm_hiddens, config.dropout_lstm_input,
                              config.dropout_lstm_hidden,
                              config.mlp_arc_size,
                              config.mlp_rel_size, config.dropout_mlp, config.debug)


def _load_conll(path) -> Document:
    """
    Load whole conll file as a document
    :param path: .conll or .conllx file
    :return: single document
    """

    def create_sentence() -> Sentence:
        sent = Sentence()
        sent[POS] = []
        sent[HEA] = []
        sent[DEP] = []
        return sent

    sents = []
    with open(path) as src:
        sent = create_sentence()
        for line in src:
            info = line.strip().split()
            if info:
                assert (len(info) == 10), 'Illegal line: %s' % line
                word, tag, head, rel = info[1], info[3], info[6], info[7]
                sent.tokens.append(word)
                sent.part_of_speech_tags.append(tag)
                sent[HEA].append(head)
                sent[DEP].append(rel)
            else:
                sents.append(sent)
                sent = create_sentence()
    return Document({SEN: sents})


if __name__ == '__main__':
    train = _load_conll('data/ptb/dep/train-debug.conllx')
    dev = _load_conll('data/ptb/dep/dev-debug.conllx')
    parser = DependencyParser()
    model_path = 'data/model/ptb/dep-debug'
    parser.train([train], [dev], model_path=model_path, config_file='data/ptb/dep/config-debug.ini')
    parser.load(model_path)
    parser.decode([dev])
    sentence = [('Is', 'VBZ'), ('this', 'DT'), ('the', 'DT'), ('future', 'NN'), ('of', 'IN'), ('chamber', 'NN'),
                ('music', 'NN'), ('?', '.')]
    print(parser.parse(sentence))
