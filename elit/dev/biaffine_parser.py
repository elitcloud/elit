# -*- coding:utf-8 -*-
# Filename: biaffine_parser.py
# Author：hankcs
# Date: 2018-01-30 18:27
import argparse
import pickle
from os.path import isfile

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

from elit.dev.biaffineparser.common import orthonormal_initializer, bilinear, Tarjan, Vocabulary, CoNLLSentence, eprint, \
    Config, DataSet

__author__ = 'Han He'


class BiaffineParser(object):
    """
    A re-implementation of Deep Biaffine Attention for Neural Dependency Parsing (2017).
    """

    def __init__(self, vocab,
                 word_dims,
                 tag_dims,
                 dropout_dim,
                 lstm_layers,
                 lstm_hiddens,
                 dropout_lstm_input,
                 dropout_lstm_hidden,
                 mlp_arc_size,
                 mlp_rel_size,
                 dropout_mlp,
                 learning_rate,
                 beta_1,
                 beta_2,
                 epsilon,
                 model_output,
                 debug=False
                 ):
        """
        Create a parser, build the computation graph

        :param vocab: vocabulary
        :param word_dims: word embedding dimension
        :param tag_dims: pos-tag embedding dimension
        :param dropout_dim: word dropout, not implemented yet
        :param lstm_layers: layers of rnn
        :param lstm_hiddens: hidden size of rnn
        :param dropout_lstm_input: dropout for lstm input
        :param dropout_lstm_hidden: dropout for lstm hidden, not correctly implemented yet.
                                    Need to read A Theoretically Grounded Application of Dropout in
                                    Recurrent Neural Networks
        :param mlp_arc_size: MLP_arc size
        :param mlp_rel_size: MLP_rel size
        :param dropout_mlp: word dropout of inputs to MLP
        :param learning_rate: float
        :param beta_1: Adam optimizer beta_1
        :param beta_2: Adam optimizer beta_2
        :param epsilon: Adam optimizer epsilon
        :param model_output: where to save model
        :param debug: debug mode, will use some simple tricks to save time
        """
        self._vocab = vocab
        self.model_output = model_output
        self.ensure_tree = True

        # placeholder
        # shape = (batch size, max length of sentence in batch)
        self.word_inputs = tf.placeholder(tf.int32, [None, None], name="word_placeholder")
        self.tag_inputs = tf.placeholder(tf.int32, [None, None], name="tag_placeholder")
        self.arc_targets = tf.placeholder(tf.int32, shape=[None, None], name="arc_targets")
        self.rel_targets = tf.placeholder(tf.int32, shape=[None, None], name="rel_targets")
        self.batch_size = tf.placeholder(dtype=tf.int32, name="batch_size")
        self.max_seq_len = tf.placeholder(dtype=tf.int32, name="seq_len")
        self.is_training = tf.placeholder_with_default(True, [], name='is_training')

        def get_dropout(dropout, name):
            return tf.cond(self.is_training, lambda: tf.constant(dropout, tf.float32, name=name),
                           lambda: tf.constant(1.0, tf.float32, name=name))

        self.dropout_lstm_input = get_dropout(dropout_lstm_input, 'dropout_lstm_input')
        self.dropout_lstm_hidden = get_dropout(dropout_lstm_hidden, 'dropout_lstm_hidden')
        self.dropout_mlp = get_dropout(dropout_mlp, 'dropout_mlp')

        mask = tf.greater(self.word_inputs, tf.constant(self._vocab.ROOT, tf.int32))

        with tf.variable_scope('Embedding'):
            def add_embeddings(initial_lookup_table, input_holder, trainable, name):
                embedded = tf.Variable(initial_lookup_table, trainable=trainable, name=name)
                embeddings = tf.nn.embedding_lookup(embedded, input_holder)
                return embeddings

            word_unked = tf.where(self.word_inputs < tf.constant(self._vocab.words_in_train, tf.int32),
                                  self.word_inputs,
                                  tf.fill(tf.shape(self.word_inputs), self._vocab.UNK))
            word_embs = add_embeddings(vocab.get_word_embs(word_dims), word_unked, True, 'word_embs')
            pret_word_embs = add_embeddings(vocab.get_pret_embs(), self.word_inputs, False, 'pret_word_embs')
            tag_embs = add_embeddings(vocab.get_tag_embs(tag_dims), self.tag_inputs, True, 'tag_embs')
            emb_inputs = tf.concat([word_embs + pret_word_embs, tag_embs], axis=-1)

        with tf.variable_scope('RNN'):
            def lstm_cell(keep_prob):
                cell = rnn.LSTMCell(lstm_hiddens, reuse=tf.get_variable_scope().reuse)
                return rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)

            # The dropout_lstm_input and dropout_lstm_hidden are not correctly implemented, need to
            # modify bidirectional_dynamic_rnn
            multi_lstm_cell_fw = tf.contrib.rnn.MultiRNNCell(
                [lstm_cell(self.dropout_lstm_input)] + [lstm_cell(self.dropout_lstm_hidden)
                                                        for _ in range(lstm_layers - 1)], state_is_tuple=True)
            multi_lstm_cell_bw = tf.contrib.rnn.MultiRNNCell(
                [lstm_cell(self.dropout_lstm_input)] + [lstm_cell(self.dropout_lstm_hidden)
                                                        for _ in range(lstm_layers - 1)], state_is_tuple=True)
            # self.debug = tf.shape(emb_inputs)
            # (batch_size, max_time, output_size)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(multi_lstm_cell_fw, multi_lstm_cell_bw,
                                                                        inputs=emb_inputs, dtype=tf.float32)
            # (batch_size, max_time, 2 * output_size)
            top_recur = tf.concat([output_fw, output_bw], axis=-1, name='top_recur_concat')
            top_recur = tf.reshape(top_recur, [self.batch_size * self.max_seq_len, 2 * lstm_hiddens],
                                   name='reshape_top_recur')

        with tf.variable_scope('MLP'):
            mlp_size = mlp_arc_size + mlp_rel_size
            W = orthonormal_initializer(2 * lstm_hiddens, mlp_size, debug)
            mlp_dep_W = tf.Variable(W)
            mlp_head_W = tf.Variable(W)
            mlp_dep_b = tf.Variable(tf.constant(0., shape=[mlp_size]), dtype=tf.float32)
            mlp_head_b = tf.Variable(tf.constant(0., shape=[mlp_size]), dtype=tf.float32)
            arc_W = tf.Variable(tf.constant(0., shape=(mlp_arc_size, mlp_arc_size + 1)), dtype=tf.float32)
            rel_W = tf.Variable(tf.constant(0., shape=(vocab.rel_size * (mlp_rel_size + 1), mlp_rel_size + 1)),
                                dtype=tf.float32)
            dep = tf.nn.relu(tf.matmul(top_recur, mlp_dep_W) + mlp_dep_b)
            dep = tf.reshape(dep, [self.batch_size, self.max_seq_len, mlp_size], name='reshape_dep')
            head = tf.nn.relu(tf.matmul(top_recur, mlp_head_W) + mlp_head_b)
            head = tf.reshape(head, [self.batch_size, self.max_seq_len, mlp_size], name='reshape_head')
            dep_arc, dep_rel = dep[:, :, :mlp_arc_size], dep[:, :, mlp_arc_size:]
            head_arc, head_rel = head[:, :, :mlp_arc_size], head[:, :, mlp_arc_size:]

        with tf.variable_scope('Arc'):
            # (n x b x d) * (d x 1 x d) * (n x b x d).T -> (n x b x b)
            arc_logits = self.bilinear(dep_arc, arc_W, head_arc, output_size=1, add_bias1=True, add_bias2=False)
            # (n x b x b)
            self.arc_probs = tf.nn.softmax(arc_logits)
            # (n x b)
            arc_preds = tf.to_int32(tf.argmax(arc_logits, axis=-1))
            # (n x b)
            arc_targets = self.arc_targets
            # (n x b)
            arc_correct = tf.to_int32(tf.boolean_mask(tf.equal(arc_preds, arc_targets), mask))
            # ()
            arc_loss = tf.losses.sparse_softmax_cross_entropy(arc_targets, arc_logits, mask)

        with tf.variable_scope('Rel'):
            # (n x b x d) * (d x r x d) * (n x b x d).T -> (n x b x r x b)
            rel_logits = self.bilinear(dep_rel, rel_W, head_rel, output_size=self._vocab.rel_size,
                                       add_bias1=True, add_bias2=True)
            # self.debug = rel_logits
            # (n x b x r x b)
            self.rel_probs = tf.nn.softmax(rel_logits, dim=2)
            # (n x b x b)
            one_hot = tf.one_hot(tf.where(self.is_training, self.arc_targets, arc_preds), self.max_seq_len)
            # (n x b x b) -> (n x b x b x 1)
            one_hot = tf.expand_dims(one_hot, axis=3)
            # (n x b x r x b) * (n x b x b x 1) -> (n x b x r x 1)
            select_rel_logits = tf.matmul(rel_logits, one_hot)
            # (n x b x r x 1) -> (n x b x r)
            select_rel_logits = tf.squeeze(select_rel_logits, axis=3)
            # (n x b)
            rel_preds = tf.to_int32(tf.argmax(select_rel_logits, axis=-1))
            # (n x b)
            rel_targets = self.rel_targets
            # (n x b)
            rel_correct = tf.to_int32(tf.boolean_mask(tf.equal(rel_preds, rel_targets), mask)) * arc_correct
            # ()
            rel_loss = tf.losses.sparse_softmax_cross_entropy(rel_targets, select_rel_logits, mask)

        self.loss = tf.reduce_mean(arc_loss + rel_loss)
        optimizer = tf.train.AdamOptimizer(learning_rate, beta_1, beta_2, epsilon)
        self.train_op = optimizer.minimize(self.loss)

        # accuracy
        num_tokens = tf.reduce_sum(tf.to_int32(mask), name='reduce_sum_mask')
        self.arc_accuracy = tf.reduce_sum(arc_correct) / num_tokens * 100.
        self.rel_accuracy = tf.reduce_sum(rel_correct) / num_tokens * 100.

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def save(self):
        """Saves session = weights"""
        self.saver.save(self.sess, self.model_output)

    def load(self):
        """Reload weights into session
        """
        # print("Reloading the latest trained model...")
        self.saver.restore(self.sess, self.model_output)

    def close(self):
        """Closes the session"""
        self.sess.close()

    def bilinear(self, inputs1, weights, inputs2, output_size, add_bias1=True, add_bias2=True):
        """
        Perform inputs1 x weights x inputs2, (n x b x d) * (d x r x d) * (n x b x d).T -> (n x b x r x b)
        where n is batch size, b is max sequence length, (d x r x d) is the shape of tensor weights, r is output size

        Adopted from Timothy Dozat https://github.com/tdozat, with some modifications

        :param inputs1:
        :param weights:
        :param inputs2:
        :param output_size:
        :param add_bias1:
        :param add_bias2:
        :return:
        """
        if isinstance(inputs1, (list, tuple)):
            n_dims1 = len(inputs1[0].get_shape().as_list())
            inputs1 = tf.concat(inputs1, n_dims1 - 1)
        else:
            n_dims1 = len(inputs1.get_shape().as_list())
        inputs1_size = inputs1.get_shape().as_list()[-1]

        if isinstance(inputs2, (list, tuple)):
            n_dims2 = len(inputs2[0].get_shape().as_list())
            inputs2 = tf.concat(inputs2, n_dims2 - 1)
        else:
            n_dims2 = len(inputs2.get_shape().as_list())
        inputs2_size = inputs2.get_shape().as_list()[-1]
        try:
            assert n_dims1 == n_dims2
        except AssertionError:
            raise ValueError('Inputs1 and Inputs2 to bilinear have different no. of dims')

        def add_noise(inputs, inputs_size, n_dims):
            noise_shape = tf.stack([self.batch_size] + [1] * (n_dims - 2) + [inputs_size])
            inputs = tf.nn.dropout(inputs, self.dropout_mlp, noise_shape=noise_shape)
            return inputs

        inputs1 = tf.cond(self.is_training, lambda: add_noise(inputs1, inputs1_size, n_dims1), lambda: inputs1)
        inputs2 = tf.cond(self.is_training, lambda: add_noise(inputs2, inputs2_size, n_dims2), lambda: inputs2)

        bilin = bilinear(inputs1, weights, inputs2, output_size, add_bias1=add_bias1, add_bias2=add_bias2)

        if output_size == 1:
            if isinstance(bilin, list):
                bilin = [tf.squeeze(x, axis=(n_dims1 - 1)) for x in bilin]
            else:
                bilin = tf.squeeze(bilin, axis=(n_dims1 - 1))
        return bilin

    def train_batch(self, word_inputs, tag_inputs, arc_targets, rel_targets):
        """
        Train on a single batch. Denote n for batch size, b for max sequence length

        :param word_inputs: (b)
        :param tag_inputs: (b)
        :param arc_targets: (b)
        :param rel_targets: (b)
        :return: arc_accuracy, rel_accuracy, train_loss
        """
        batch_size, max_seq_len = word_inputs.shape
        feed = {
            self.word_inputs: word_inputs,
            self.tag_inputs: tag_inputs,
            self.arc_targets: arc_targets,
            self.rel_targets: rel_targets,
            self.max_seq_len: max_seq_len,
            self.batch_size: batch_size,
        }
        # debug = self.sess.run([self.debug], feed_dict=feed)
        # print(debug)
        _, train_loss, arc_accuracy, rel_accuracy = self.sess.run(
            [self.train_op, self.loss, self.arc_accuracy, self.rel_accuracy], feed_dict=feed)
        return arc_accuracy, rel_accuracy, train_loss
        # print('UAS:%.2f%% LAS:%.2f%% %.4f' % (arc_accuracy * 100, rel_accuracy * 100, train_loss))

    def train(self, train, dev, train_batch_size, dev_batch_size, num_epochs):
        """
        Train on datasets

        :param train: Training set
        :param dev: Validation Set
        :param train_batch_size: Batch size of training set
        :param dev_batch_size: Batch size of validation set
        :param num_epochs: Run how many epochs
        """
        global_step = 0
        best_UAS = 0
        for epoch in range(num_epochs):
            print("Epoch {:} out of {:}".format(epoch + 1, num_epochs))
            for words, tags, arcs, rels in train.get_batches(batch_size=train_batch_size, shuffle=False):
                arc_accuracy, rel_accuracy, train_loss = self.train_batch(words, tags, arcs, rels)
                print("Batch #%d: UAS: %.2f, LAS: %.2f, loss %.3f\r" % (
                    global_step, arc_accuracy, rel_accuracy, train_loss), end='', flush=True)
                global_step += 1
            UAS, LAS = self.evaluate(dev, dev_batch_size)
            print('Dev) UAS:%.2f%% LAS:%.2f%%                       ' % (UAS, LAS))
            if UAS > best_UAS:
                best_UAS = UAS
                print('- new best score!')
                self.save()
        pass

    def arc_argmax(self, arc_probs, tokens_to_keep):
        """
        Build a tree out of arc probabilities
        Adopted from Timothy Dozat https://github.com/tdozat, with some modifications

        :param arc_probs: (b x b)
        :param tokens_to_keep: Mask, (b)
        :return: (b)
        """
        if self.ensure_tree:
            tokens_to_keep[0] = True
            length = np.sum(tokens_to_keep)
            I = np.eye(len(tokens_to_keep))
            # block loops and pad heads
            arc_probs = arc_probs * tokens_to_keep * (1 - I)
            parse_preds = np.argmax(arc_probs, axis=1)
            tokens = np.arange(1, length)
            roots = np.where(parse_preds[tokens] == 0)[0] + 1
            # ensure at least one root
            if len(roots) < 1:
                # The current root probabilities
                root_probs = arc_probs[tokens, 0]
                # The current head probabilities
                old_head_probs = arc_probs[tokens, parse_preds[tokens]]
                # Get new potential root probabilities
                new_root_probs = root_probs / old_head_probs
                # Select the most probable root
                new_root = tokens[np.argmax(new_root_probs)]
                # Make the change
                parse_preds[new_root] = 0
            # ensure at most one root
            elif len(roots) > 1:
                # The probabilities of the current heads
                root_probs = arc_probs[roots, 0]
                # Set the probability of depending on the root zero
                arc_probs[roots, 0] = 0
                # Get new potential heads and their probabilities
                new_heads = np.argmax(arc_probs[roots][:, tokens], axis=1) + 1
                new_head_probs = arc_probs[roots, new_heads] / root_probs
                # Select the most probable root
                new_root = roots[np.argmin(new_head_probs)]
                # Make the change
                parse_preds[roots] = new_heads
                parse_preds[new_root] = 0
            # remove cycles
            tarjan = Tarjan(parse_preds, tokens)
            for SCC in tarjan.SCCs:
                if len(SCC) > 1:
                    dependents = set()
                    to_visit = set(SCC)
                    while len(to_visit) > 0:
                        node = to_visit.pop()
                        if not node in dependents:
                            dependents.add(node)
                            to_visit.update(tarjan.edges[node])
                    # The indices of the nodes that participate in the cycle
                    cycle = np.array(list(SCC))
                    # The probabilities of the current heads
                    old_heads = parse_preds[cycle]
                    old_head_probs = arc_probs[cycle, old_heads]
                    # Set the probability of depending on a non-head to zero
                    non_heads = np.array(list(dependents))
                    arc_probs[
                        np.repeat(cycle, len(non_heads)), np.repeat([non_heads], len(cycle), axis=0).flatten()] = 0
                    # Get new potential heads and their probabilities
                    new_heads = np.argmax(arc_probs[cycle][:, tokens], axis=1) + 1
                    new_head_probs = arc_probs[cycle, new_heads] / old_head_probs
                    # Select the most probable change
                    change = np.argmax(new_head_probs)
                    changed_cycle = cycle[change]
                    old_head = old_heads[change]
                    new_head = new_heads[change]
                    # Make the change
                    parse_preds[changed_cycle] = new_head
                    tarjan.edges[new_head].add(changed_cycle)
                    tarjan.edges[old_head].remove(changed_cycle)
            return parse_preds
        else:
            tokens_to_keep[0] = True
            # block and pad heads
            arc_probs = arc_probs * tokens_to_keep
            parse_preds = np.argmax(arc_probs, axis=1)
            return parse_preds

    def rel_argmax(self, rel_probs, tokens_to_keep):
        """
        Find arc relations

        Adopted from Timothy Dozat https://github.com/tdozat, with some modifications

        :param rel_probs: (b x r)
        :param tokens_to_keep: (b)
        :return: (b)
        """
        if self.ensure_tree:
            tokens_to_keep[0] = True
            rel_probs[:, Vocabulary.PAD] = 0
            root = Vocabulary.ROOT
            length = np.sum(tokens_to_keep)
            tokens = np.arange(1, length)
            rel_preds = np.argmax(rel_probs, axis=1)
            roots = np.where(rel_preds[tokens] == root)[0] + 1
            if len(roots) < 1:
                rel_preds[1 + np.argmax(rel_probs[tokens, root])] = root
            elif len(roots) > 1:
                root_probs = rel_probs[roots, root]
                rel_probs[roots, root] = 0
                new_rel_preds = np.argmax(rel_probs[roots], axis=1)
                new_rel_probs = rel_probs[roots, new_rel_preds] / root_probs
                new_root = roots[np.argmin(new_rel_probs)]
                rel_preds[roots] = new_rel_preds
                rel_preds[new_root] = root
            return rel_preds
        else:
            rel_probs[:, Vocabulary.PAD] = 0
            rel_preds = np.argmax(rel_probs, axis=1)
            return rel_preds

    def prob_argmax(self, arc_probs, rel_probs, tokens_to_keep):
        """
        Find the most reasonable tree

        Adopted from Timothy Dozat https://github.com/tdozat, with some modifications

        :param arc_probs:
        :param rel_probs:
        :param tokens_to_keep:
        :return:
        """

        parse_preds = self.arc_argmax(arc_probs, tokens_to_keep)
        rel_probs = rel_probs[np.arange(len(parse_preds)), :, parse_preds]
        rel_preds = self.rel_argmax(rel_probs, tokens_to_keep)
        return parse_preds, rel_preds

    def parse_batch(self, word_inputs, tag_inputs):
        """
        Parse one batch

        :param word_inputs:
        :param tag_inputs:
        :return:
        """
        batch_size, max_seq_len = word_inputs.shape
        feed = {
            self.word_inputs: word_inputs,
            self.tag_inputs: tag_inputs,
            self.max_seq_len: max_seq_len,
            self.batch_size: batch_size,
            self.is_training: False
        }

        b_arc_probs, b_rel_probs = self.sess.run([self.arc_probs, self.rel_probs], feed_dict=feed)
        b_arc_preds = np.zeros((batch_size, max_seq_len), dtype=np.int32)
        b_rel_preds = np.zeros((batch_size, max_seq_len), dtype=np.int32)
        b_tokens_to_keep = np.greater(word_inputs, Vocabulary.ROOT)
        for idx, (inputs, parse_probs, rel_probs, tokens_to_keep) in enumerate(
                zip(word_inputs, b_arc_probs, b_rel_probs,
                    b_tokens_to_keep)):
            arc_preds, rel_preds = self.prob_argmax(parse_probs, rel_probs, tokens_to_keep)
            b_arc_preds[idx, :] = arc_preds
            b_rel_preds[idx, :] = rel_preds
        return b_arc_preds, b_rel_preds, b_tokens_to_keep

    def parse(self, sentence):
        """
        Parse raw sentence.

        :param sentence: A list of (word, tag) pair. Both word and pair are raw strings
        :return: A CoNLLSentence
        """
        length = len(sentence)
        word_ids, tag_ids = self._vocab.sentence2id(sentence)
        word_ids = np.expand_dims(word_ids, axis=0)
        tag_ids = np.expand_dims(tag_ids, axis=0)
        arc_preds, rel_preds, tokens_to_keep = self.parse_batch(word_ids, tag_ids)
        arc_preds, rel_preds = arc_preds[0][1:length + 1], rel_preds[0][1:length + 1]
        conll = CoNLLSentence([p[0] for p in sentence], [p[1] for p in sentence], arc_preds.tolist(),
                              self._vocab.id2rel(rel_preds.tolist()))
        return conll

    def evaluate_batch(self, word_inputs, tag_inputs, arc_targets, rel_targets):
        """
        Perform evaluation on a single batch.

        :param word_inputs:
        :param tag_inputs:
        :param arc_targets:
        :param rel_targets:
        :return: arc_correct, rel_correct, length of sentences in batch
        """
        arc_preds, rel_preds, tokens_to_keep = self.parse_batch(word_inputs, tag_inputs)
        arc_equal = np.equal(arc_preds, arc_targets) * tokens_to_keep
        arc_correct = np.sum(arc_equal)
        rel_correct = np.sum(arc_equal * np.equal(rel_preds, rel_targets))
        length = np.sum(tokens_to_keep)
        return arc_correct, rel_correct, length

    def evaluate(self, dataset, batch_size=128):
        """
        Perform evaluation on whole dataset

        :param dataset:
        :param batch_size:
        :return: UAS, LAS
        """
        total_tokens = 0
        total_arc_cor = 0
        total_rel_cor = 0
        for words, tags, arcs, rels in dataset.get_batches(batch_size=batch_size, shuffle=False):
            arc_correct, rel_correct, length = self.evaluate_batch(words, tags, arcs, rels)
            total_arc_cor += arc_correct
            total_rel_cor += rel_correct
            total_tokens += length
        UAS = total_arc_cor / total_tokens * 100
        LAS = total_rel_cor / total_tokens * 100
        # print('UAS:%.2f%% LAS:%.2f%%' % (UAS, LAS))
        return UAS, LAS


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config_file', default='configs/ptb-debug.ini', help="Configuration file of model")
    args, extra_args = arg_parser.parse_known_args()
    if not isfile(args.config_file):
        eprint('%s not exist' % args.config_file)
        exit(1)
    config = Config(args.config_file, extra_args)

    if isfile('config.save_vocab_path'):
        vocab = pickle.load(open(config.save_vocab_path, 'rb'))
    else:
        vocab = Vocabulary(config.train_file, config.pretrained_embeddings_file, config.min_occur_count)
        pickle.dump(vocab, open(config.save_vocab_path, 'wb'))
    parser = BiaffineParser(vocab, config.word_dims, config.tag_dims, config.dropout_emb, config.lstm_layers,
                            config.lstm_hiddens, config.dropout_lstm_input, config.dropout_lstm_hidden,
                            config.mlp_arc_size, config.mlp_rel_size, config.dropout_mlp, config.learning_rate,
                            config.beta_1, config.beta_2, config.epsilon, config.save_model_path, config.debug)
    train = DataSet(config.train_file, config.num_buckets_train, vocab)
    dev = DataSet(config.dev_file, config.num_buckets_valid, vocab)
    parser.train(train, dev, config.train_batch_size, config.test_batch_size, config.train_iters)
    parser.load()

    # print(parser.parse([('Is', 'VBZ'), ('this', 'DT'), ('the', 'DT'), ('future', 'NN'), ('of', 'IN'),
    #                     ('chamber', 'NN'), ('music', 'NN'), ('?', '.')]))
    test = DataSet(config.test_file, config.num_buckets_test, vocab)
    UAS, LAS = parser.evaluate(test, batch_size=config.test_batch_size)
    print('Test) UAS:%.2f%% LAS:%.2f%%                          ' % (UAS, LAS))
