# -*- coding:utf-8 -*-
# Filename: BiLSTMCRFTagger.py
# Author：hankcs
# Date: 2018-02-21 22:00

import math
import os
import random
import sys

import numpy as np

from elit.dep.common import utils
from elit.dep.common.utils import stdchannel_redirected, Evaluator
from elit.dep.tagger.corpus import Vocabulary, TSVCorpus
from elit.dep.tagger.tagger_config import TaggerConfig

with stdchannel_redirected(sys.stderr, os.devnull):
    import dynet as dy


class BiLSTM_CRF(object):
    def __init__(self, vocab: Vocabulary, num_lstm_layers, hidden_dim, word_embeddings, no_we_update,
                 char_embeddings, char_hidden_dim, margins, word_embedding_dim=100,
                 char_embedding_dim=50, tie_two_embeddings=False, bigram_embeddings=None):
        self.dropout = None
        self.model = dy.Model()
        self.tagset_size = tagset_size = vocab.tagset_size()
        self.margins = margins
        self.we_update = not no_we_update
        self.vocab = vocab
        self.evaluator = Evaluator

        vocab_size = vocab.word_vocab.size()

        # Word embedding parameters
        self.use_we = word_embedding_dim > 0
        if self.use_we:
            if word_embeddings is not None:  # Use pretrained embeddings
                vocab_size = word_embeddings.shape[0]
                word_embedding_dim = word_embeddings.shape[1]
            self.word_lookup = self.model.add_lookup_parameters((vocab_size, word_embedding_dim))
            if word_embeddings is not None:
                self.word_lookup.init_from_array(word_embeddings)
        else:
            self.word_lookup = None

        # bigram embeddings
        if bigram_embeddings:
            # self.bigram_lookup = self.model.add_lookup_parameters((len(b2i), word_embedding_dim))
            self.bigram_lookup = None
            self.bigram_lookup.init_from_array(bigram_embeddings)
        else:
            self.bigram_lookup = None

        # Char LSTM Parameters
        self.use_char_rnn = char_hidden_dim > 0
        if char_hidden_dim:
            charset_size = vocab.char_vocab.size()
            if char_embeddings is not None:
                charset_size = char_embeddings.shape[0]
                char_embedding_dim = char_embeddings.shape[1]
            self.char_embedding_dim = char_embedding_dim
            if tie_two_embeddings:
                self.char_lookup = self.word_lookup
            else:
                self.char_lookup = self.model.add_lookup_parameters((charset_size, self.char_embedding_dim))
                if char_embeddings is not None:
                    self.char_lookup.init_from_array(char_embeddings)
            self.char_bi_lstm = dy.BiRNNBuilder(1, self.char_embedding_dim, char_hidden_dim, self.model, dy.LSTMBuilder)

        # Word LSTM parameters
        if self.use_char_rnn:
            if self.use_we:
                input_dim = word_embedding_dim + char_hidden_dim
            else:
                input_dim = char_hidden_dim
        else:
            input_dim = word_embedding_dim
        self.bi_lstm = dy.BiRNNBuilder(num_lstm_layers, input_dim, hidden_dim, self.model, dy.LSTMBuilder)
        if bigram_embeddings:
            self.lstm_to_tags_params = self.model.add_parameters((tagset_size, hidden_dim + word_embedding_dim * 2))
        else:
            self.lstm_to_tags_params = self.model.add_parameters((tagset_size, hidden_dim))
        self.lstm_to_tags_bias = self.model.add_parameters(tagset_size)
        self.mlp_out = self.model.add_parameters((tagset_size, tagset_size))
        self.mlp_out_bias = self.model.add_parameters(tagset_size)

        # Transition matrix for tagging layer, [i,j] is score of transitioning to i from j
        self.transitions = self.model.add_lookup_parameters((tagset_size, tagset_size))

    def set_dropout(self, p):
        self.bi_lstm.set_dropout(p)
        self.dropout = p

    def disable_dropout(self):
        self.bi_lstm.disable_dropout()
        self.dropout = None

    def word_rep(self, word):
        """
        word to its representation
        :param word: tuple of (char_ids, word_id)
        :return: a vector
        """
        if self.bigram_lookup:
            word = word[1]
            pass

        if self.use_char_rnn:
            char_ids, word = word
            char_embs = [self.char_lookup[cid] for cid in char_ids]
            char_exprs = self.char_bi_lstm.transduce(char_embs)
            if self.use_we:
                wemb = dy.lookup(self.word_lookup, word, update=self.we_update)
                return dy.concatenate([wemb, char_exprs[-1]])
            else:
                return char_exprs[-1]
        else:
            wemb = dy.lookup(self.word_lookup, word, update=self.we_update)
            return wemb

    def build_tagging_graph(self, sentence):
        dy.renew_cg()

        embeddings = [self.word_rep(w) for w in sentence]

        lstm_out = self.bi_lstm.transduce(embeddings)

        H = dy.parameter(self.lstm_to_tags_params)
        Hb = dy.parameter(self.lstm_to_tags_bias)
        O = dy.parameter(self.mlp_out)
        Ob = dy.parameter(self.mlp_out_bias)
        scores = []
        if self.bigram_lookup:
            for rep, word in zip(lstm_out, sentence):
                bi1 = dy.lookup(self.bigram_lookup, word[0], update=self.we_update)
                bi2 = dy.lookup(self.bigram_lookup, word[1], update=self.we_update)
                if self.dropout is not None:
                    bi1 = dy.dropout(bi1, self.dropout)
                    bi2 = dy.dropout(bi2, self.dropout)
                score_t = O * dy.tanh(H * dy.concatenate(
                    [bi1,
                     rep,
                     bi2]) + Hb) + Ob
                scores.append(score_t)
        else:
            for rep in lstm_out:
                score_t = O * dy.tanh(H * rep + Hb) + Ob
                scores.append(score_t)

        return scores

    def score_sentence(self, observations, tags):
        if len(tags) == 0:
            tags = [-1] * len(observations)
        assert len(observations) == len(tags)
        score_seq = [0]
        score = dy.scalarInput(0)
        tags = [self.vocab.START_ID] + tags
        for i, obs in enumerate(observations):
            score = score + dy.pick(self.transitions[tags[i + 1]], tags[i]) + dy.pick(obs, tags[i + 1])
            score_seq.append(score.value())
        score = score + dy.pick(self.transitions[self.vocab.END_ID], tags[-1])
        return score

    def viterbi_loss(self, sentence, gold_tags, use_margins=False):
        observations = self.build_tagging_graph(sentence)
        viterbi_tags, viterbi_score = self.viterbi_decoding(observations, gold_tags, use_margins)
        if gold_tags and viterbi_tags != gold_tags:
            gold_score = self.score_sentence(observations, gold_tags)
            return (viterbi_score - gold_score), viterbi_tags
        else:
            return dy.scalarInput(0), viterbi_tags

    def neg_log_loss(self, sentence, gold_tags):
        observations = self.build_tagging_graph(sentence)
        gold_score = self.score_sentence(observations, gold_tags)
        forward_score = self.forward(observations)
        return forward_score - gold_score

    def loss(self, sentence, gold_tags):
        observations = self.build_tagging_graph(sentence)
        errors = []
        for obs, tag in zip(observations, gold_tags):
            err_t = dy.pickneglogsoftmax(obs, tag)
            errors.append(err_t)
        return dy.esum(errors)

    def forward(self, observations):

        def log_sum_exp(scores):
            npval = scores.npvalue()
            argmax_score = np.argmax(npval)
            max_score_expr = dy.pick(scores, argmax_score)
            max_score_expr_broadcast = dy.concatenate([max_score_expr] * self.tagset_size)
            return max_score_expr + dy.log(dy.sum_dim(dy.transpose(dy.exp(scores - max_score_expr_broadcast)), [1]))

        init_alphas = [-1e10] * self.tagset_size
        init_alphas[Vocabulary.START_ID] = 0
        for_expr = dy.inputVector(init_alphas)
        for obs in observations:
            alphas_t = []
            for next_tag in range(self.tagset_size):
                obs_broadcast = dy.concatenate([dy.pick(obs, next_tag)] * self.tagset_size)
                next_tag_expr = for_expr + self.transitions[next_tag] + obs_broadcast
                alphas_t.append(log_sum_exp(next_tag_expr))
            for_expr = dy.concatenate(alphas_t)
        terminal_expr = for_expr + self.transitions[Vocabulary.END_ID]
        alpha = log_sum_exp(terminal_expr)
        return alpha

    def viterbi_decoding(self, observations, gold_tags=None, use_margins=False):
        backpointers = []
        init_vvars = [-1e10] * self.tagset_size
        init_vvars[Vocabulary.START_ID] = 0  # <Start> has all the probability
        for_expr = dy.inputVector(init_vvars)
        trans_exprs = [self.transitions[idx] for idx in range(self.tagset_size)]
        for idx, obs in enumerate(observations):
            bptrs_t = []
            vvars_t = []
            for next_tag in range(self.tagset_size):
                next_tag_expr = for_expr + trans_exprs[next_tag]
                next_tag_arr = next_tag_expr.npvalue()
                best_tag_id = np.argmax(next_tag_arr)
                bptrs_t.append(best_tag_id)
                vvars_t.append(dy.pick(next_tag_expr, best_tag_id))
            for_expr = dy.concatenate(vvars_t) + obs

            # optional margin adaptation
            if use_margins and self.margins != 0 and gold_tags:
                adjust = [self.margins] * self.tagset_size
                adjust[gold_tags[idx]] = 0
                for_expr = for_expr + dy.inputVector(adjust)
            backpointers.append(bptrs_t)
        # Perform final transition to terminal
        terminal_expr = for_expr + trans_exprs[Vocabulary.END_ID]
        terminal_arr = terminal_expr.npvalue()
        best_tag_id = np.argmax(terminal_arr)
        path_score = dy.pick(terminal_expr, best_tag_id)
        # Reverse over the backpointers to get the best path
        best_path = [best_tag_id]  # Start with the tag that was best for terminal
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()  # Remove the start symbol
        best_path.reverse()
        assert start == Vocabulary.START_ID
        # Return best path and best path's score
        return best_path, path_score

    def evaluate(self, instances, evaluator=None):
        if not evaluator:
            evaluator = self.evaluator(self.vocab.tag_vocab.i2s)
            pass
        self.disable_dropout()
        # total_loss = 0.0
        total_instance = 0
        batch_size = math.ceil(len(instances) * 0.01)
        nbatches = (len(instances) + batch_size - 1) // batch_size
        bar = utils.Progbar(target=nbatches)
        for batch_id, batch in enumerate(utils.minibatches(instances, batch_size)):
            for idx, instance in enumerate(batch):
                sentence = instance.words
                if len(sentence) == 0: continue

                gold_tags = instance.tags
                # loss = self.neg_log_loss(sentence, gold_tags)
                # loss = loss.scalar_value()
                viterbi_tags = self.predict(sentence)
                evaluator.add_instance(gold_tags, viterbi_tags)
                total_instance += 1
                # total_loss += loss

            bar.update(batch_id + 1,
                       exact=[
                           # ("dev loss", total_loss / total_instance),
                           ("f1", evaluator.result()[-1])])
        return evaluator.result()

    def predict(self, sentence):
        observations = self.build_tagging_graph(sentence)
        viterbi_tags, viterbi_score = self.viterbi_decoding(observations)
        return viterbi_tags

    def train(self, training_instances, dev_instances=None, num_epochs=60, batch_size=20, learning_rate=0.01,
              learning_rate_decay=0.9, dropout=0.2, clip_norm=None, save_path=None, logger=None, debug=False):
        if dev_instances is None:
            dev_instances = training_instances[:int(len(training_instances) * 0.1)]
        if type(training_instances) is TSVCorpus:
            training_instances = training_instances.sentences
        if type(dev_instances) is TSVCorpus:
            dev_instances = dev_instances.sentences
        if debug:
            training_instances = training_instances[:200]
            dev_instances = dev_instances[:100]
            num_epochs = 2
        trainer = dy.MomentumSGDTrainer(self.model, learning_rate, 0.9)
        if clip_norm > 0:
            trainer.set_clip_threshold(clip_norm)
        if logger:
            # logger.info("Training Algorithm: {}".format(type(trainer)))
            logger.info("# training instances: {}".format(len(training_instances)))
            logger.info("# dev instances: {}".format(len(dev_instances)))
        training_total_tokens = 0
        best_f1 = 0.
        for epoch in range(num_epochs):
            if logger:
                logger.info("Epoch {} out of {}".format(epoch + 1, num_epochs))
            random.shuffle(training_instances)
            train_loss = 0.0
            train_total_instance = 0  # size of trained instances

            if dropout > 0:
                self.set_dropout(dropout)

            nbatches = (len(training_instances) + batch_size - 1) // batch_size
            bar = utils.Progbar(target=nbatches)
            for batch_id, batch in enumerate(utils.minibatches(training_instances, batch_size)):
                for instance in batch:
                    train_total_instance += 1
                    loss_expr = self.neg_log_loss(instance.words, instance.tags)
                    # Forward pass
                    loss = loss_expr.scalar_value()
                    # Backward pass
                    loss_expr.backward()

                    # Bail if loss is NaN
                    if math.isnan(loss):
                        assert False, "NaN occured"

                    train_loss += loss
                    training_total_tokens += len(instance.words)

                trainer.update()
                if batch_size == 1 and batch_id % 10 != 0 and batch_id + 1 != train_total_instance:
                    # online learning, don't print too often
                    continue
                bar.update(batch_id + 1, exact=[("train loss", train_loss / train_total_instance)])

            trainer.learning_rate *= learning_rate_decay
            f1 = self.evaluate(dev_instances)[-1]
            if f1 > best_f1:
                best_f1 = f1
                if logger:
                    logger.info('%.2f%% - new best dev score' % f1)
                if save_path:
                    self.save(save_path)
            else:
                if logger:
                    logger.info('%.2f%%' % f1)

    def train_on_config(self, train_set, config: TaggerConfig, logger=None):
        # if train_set is None:
        #     self.vocab = Vocabulary(config.lower_case)  # ready for training
        #     train_set = TSVCorpus(config.train_file, self.vocab)
        #     self.vocab.add_pret_words(config.pretrained_embeddings_file, keep_oov=config.keep_oov)
        dev_set = TSVCorpus(config.dev_file, self.vocab)
        self.vocab.save(config.save_vocab_path)
        self.train(train_set, dev_set, config.num_epochs, config.batch_size, learning_rate=config.learning_rate,
                   learning_rate_decay=config.learning_rate_decay, dropout=config.dropout, clip_norm=config.clip_norm,
                   save_path=config.save_model_path, logger=logger, debug=config.debug)

    def save(self, file_name):
        self.model.save(file_name)

    def load(self, file_name):
        self.model.populate(file_name)

    def tag(self, sentence: list):
        words = [self.vocab.ensure_word_ids(word) for word in sentence]
        tags = self.predict(words)
        return [self.vocab.tag_vocab.get_str(t) for t in tags]


def tagger_from_vocab_config(vocab: Vocabulary, config: TaggerConfig) -> BiLSTM_CRF:
    return BiLSTM_CRF(vocab, config.word_lstm_layers, config.word_lstm_dim, vocab.pret_word_embs, False,
                      None, config.char_lstm_dim, config.margins, config.word_embedding_dim,
                      config.char_embedding_dim, config.tie_two_embeddings)


def tagger_from_config_file(config_file_path: str) -> (BiLSTM_CRF, TaggerConfig, TSVCorpus):
    config = TaggerConfig(config_file_path)
    if os.path.isfile(config.save_vocab_path):  # trained model
        vocab = Vocabulary.load(config.save_vocab_path)
        return tagger_from_vocab_config(vocab, config), config, None
    else:
        vocab = Vocabulary(config.lower_case, config.char_lstm_dim > 0)  # ready for training
        train_set = TSVCorpus(config.train_file, vocab)
        vocab.add_pret_words(config.pretrained_embeddings_file, keep_oov=config.keep_oov)
        return tagger_from_vocab_config(vocab, config), config, train_set


def test():
    config = TaggerConfig('data/pku/config.ini')
    vocab = Vocabulary.load(config.save_vocab_path)
    model = tagger_from_vocab_config(vocab, config)
    model.load(config.save_model_path)
    print(model.tag([c for c in '商品和服务']))


def train():
    config = TaggerConfig('data/pku/config.ini')
    vocab = Vocabulary()
    train = TSVCorpus(config.train_file, vocab)
    vocab.add_pret_words(config.pretrained_embeddings_file, keep_oov=False)
    dev = TSVCorpus(config.dev_file, vocab)
    model = tagger_from_vocab_config(vocab, config)
    model.train(train, dev, config.num_epochs, config.batch_size, save_path=config.save_model_path,
                debug=False)
    vocab.save(config.save_vocab_path)


if __name__ == '__main__':
    # train()
    test()
