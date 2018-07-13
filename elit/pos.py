# ========================================================================
# Copyright 2017 Emory University
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
import argparse
import logging
import pickle
import time
from types import SimpleNamespace

import mxnet as mx
import numpy as np
from mxnet import gluon

from elit.component import ForwardState, NLPComponent, FFNNModel, LSTMModel, TokenTagger
from elit.lexicon import FastText, Word2Vec, X_ANY, get_loc_embeddings, get_embeddings, x_extract, LabelMap
from elit.utils import TOKEN, POS, Accuracy
from elit.utils.file import read_tsv
from elit.util import pkl, gln

__author__ = 'Jinho D. Choi'


class POSState(ForwardState):
    def __init__(self, document, params):
        """
        POSState inherits the one-pass, left-to-right decoding strategy from ForwardState.
        :param document: the input document.
        :type document: elit.util.Document
        :param params: parameters created by POSTagger.create_params()
        :type params: SimpleNamespace
        """
        super().__init__(document, params.label_map, params.zero_output, POS)
        self.windows = params.windows
        self.embs = [get_loc_embeddings(document), get_embeddings(params.word_vsm, document)]
        if params.ambi_vsm:
            self.embs.append(get_embeddings(params.ambi_vsm, document))
        self.embs.append((self.output, self.zero_output))

    def eval(self, metric):
        """
        :type metric: elit.nlp.metric.Accuracy
        """
        preds = self.labels

        for i, sentence in enumerate(self.document):
            gold = sentence[POS]
            pred = preds[i]
            metric.correct += len([1 for g, p in zip(gold, pred) if g == p])
            metric.total += len(gold)

    def x(self):
        """
        :return: the n x d matrix where n = # of windows and d = word_emb.dim + ambi_emb.dim + num_class + 2
        """
        t = len(self.document[self.sen_id])
        l = ([x_extract(self.tok_id, w, t, emb[self.sen_id], zero) for w in self.windows] for
             emb, zero in self.embs)
        return np.column_stack(l)


class POSTagger(TokenTagger):
    def __init__(self, ctx, vsm_list):
        """
        A part-of-speech tagger.
        :param ctx: "[cg]\d*"; the context (e.g., CPU or GPU) to process.
        :type ctx: str
        :param vsm_list: a list of vector space models (must include at least one).
        :type vsm_list: list of elit.lexicon.VectorSpaceModel
        """
        super().__init__(ctx, vsm_list)

    # override
    def create_state(self, document):
        return POSState(document)

    # override
    def eval_metric(self):
        return Accuracy()


# ======================================== Train ========================================

def tsv_reader(filepath, create_state):
    # TODO:
    return []

def train():
    # processor
    args = train_args()
    if not args.trn_path:
        predict(args)
        return

    log = ['Configuration',
           '- train batch    : %d' % args.trn_batch,
           '- learning rate  : %f' % args.learning_rate,
           '- dropout ratio  : %f' % args.dropout,
           '- n-gram filters : %s' % str(args.ngram_filters),
           '- num of classes : %d' % args.num_class,
           '- windows        : %s' % str(args.windows)]

    logging.info('\n'.join(log))

    word_vsm = FastText(args.word_vsm)
    ambi_vsm = Word2Vec(args.ambi_vsm) if args.ambi_vsm else None
    comp = POSTagger()
    comp.load(args.mod_path, args.ctx, word_vsm, ambi_vsm, args.num_class, args.windows,
              args.ngram_filters, args.dropout)

    # states
    cols = {TOKEN: args.tsv_tok, POS: args.tsv_pos}
    trn_states = read_tsv(args.trn_path, cols, comp.create_state)
    dev_states = read_tsv(args.dev_path, cols, comp.create_state)

    # optimizer
    loss_func = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(comp.model.collect_params(), 'adagrad',
                            {'learning_rate': args.learning_rate})

    # train
    best_e, best_eval = -1, -1
    trn_metric = Accuracy()
    dev_metric = Accuracy()

    for e in range(args.epoch):
        trn_metric.reset()
        dev_metric.reset()

        st, mt, et, trn_eval, dev_eval = comp.train(trn_states, dev_states, args.trn_batch, trainer,
                                                    loss_func, trn_metric, args.dev_batch,
                                                    dev_metric)

        if best_eval < dev_eval:
            best_e, best_eval = e, dev_eval
            if args.mod_path:
                comp.save(args.mod_path)

        logging.info(
            '%4d: trn-time: %d, dev-time: %d, trn-acc: %5.2f (%d), dev-acc: %5.2f (%d), '
            'num-class: %d, best-acc: %5.2f @%4d' %
            (e, mt - st, et - mt, trn_eval, trn_metric.total, dev_eval, dev_metric.total,
             len(comp.params.label_map), best_eval, best_e))


def predict(args):
    word_vsm = FastText(args.word_vsm)
    ambi_vsm = Word2Vec(args.ambi_vsm) if args.ambi_vsm else None
    comp = POSTagger()
    comp.load(args.mod_path, args.ctx, )
    cols = {TOKEN: args.tsv_tok, POS: args.tsv_pos}
    dev_states = read_tsv(args.dev_path, cols, comp.create_state)
    dev_metric = Accuracy()
    dev_eval = comp.evaluate(dev_states, args.dev_batch, dev_metric)
    print("Test accuracy for %s is %5.2f" % (args.mod_path, dev_eval))
