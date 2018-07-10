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

from elit.component import ForwardState, NLPComponent, CNN2DModel, LSTMModel
from elit.lexicon import FastText, Word2Vec
from elit.structure import TOKEN, POS
from elit.util.embeddings import X_ANY, x_extract, get_embeddings, get_loc_embeddings
from elit.util.file import pkl, gln, read_tsv
from elit.util.metric import Accuracy

__author__ = 'Jinho D. Choi'


class POSState(ForwardState):
    def __init__(self, document, params):
        """
        POSState inherits the one-pass, left-to-right decoding strategy from ForwardState.
        :param document: the input document.
        :type document: elit.nlp.structure.Document
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


class POSModel(CNN2DModel):
    def __init__(self, params, **kwargs):
        """
        :param params: parameters to initialize POSModel.
        :type params: SimpleNamespace
        :param kwargs: parameters to initialize gluon.Block.
        :type kwargs: dict
        """
        loc_dim = len(X_ANY)
        word_dim = params.word_vsm.dim
        ambi_dim = params.ambi_vsm.dim if params.ambi_vsm else 0

        input_col = loc_dim + word_dim + ambi_dim + params.num_class
        ngram_conv = [SimpleNamespace(filters=f, kernel_row=i, activation='relu') for i, f in
                      enumerate(params.ngram_filters, 1)]
        super().__init__(input_col, params.num_class, ngram_conv, params.dropout, **kwargs)


class POSModelLSTM(LSTMModel):
    def __init__(self, params, **kwargs):
        """
        :param params: parameters to initialize POSModel.
        :type params: SimpleNamespace
        :param kwargs: parameters to initialize gluon.Block.
        :type kwargs: dict
        """
        loc_dim = len(X_ANY)
        word_dim = params.word_vsm.dim
        ambi_dim = params.ambi_vsm.dim if params.ambi_vsm else 0

        input_col = loc_dim + word_dim + ambi_dim
        n_hidden = 128
        super().__init__(input_col, params.num_class, n_hidden, params.dropout, **kwargs)


class POSTagger(NLPComponent):

    def __init__(self, log_path=None):
        super(POSTagger, self).__init__()
        self.params = SimpleNamespace()
        if log_path:
            logging.basicConfig(filename=log_path + '.log', filemode='w', format='%(message)s',
                                level=logging.INFO)

    # override
    def load(self, model_path, ctx=0, word_vsm=None, ambi_vsm=None, *args, **kwargs):
        """
        :param model_path: if not None, this component is initialized by objects saved
        in the model_path.
        :type model_path: str
        :param ctx: the context (e.g., CPU or GPU) to process this component.
        :type ctx: mxnet.context.Context
        :param word_vsm:
        :param ambi_vsm:
        """
        assert word_vsm is not None
        with open(pkl(model_path), 'rb') as f:
            label_map = pickle.load(f)
            num_class = pickle.load(f)
            windows = pickle.load(f)
            ngram_filters = pickle.load(f)
            dropout = pickle.load(f)
        self.params = self.create_params(word_vsm, ambi_vsm, num_class, windows, ngram_filters,
                                         dropout, label_map)
        self.ctx = ctx
        self.model = POSModel(self.params)
        self.model.load_params(gln(model_path), ctx=ctx)

    # override
    def train(self, trn_data, dev_data, model_path=".", ctx=0, tsv_tok=0, tsv_pos=1,
              word_vsm=None, ambi_vsm=None, num_class=0,
              windows=(-3, -2, -1, 0, 1, 2, 3), ngram_filters=(128, 128, 128, 128, 128),
              dropout=0.2, epoch=50, trn_batch=64, dev_batch=1024, learning_rate=0.01,
              *args, **kwargs):

        log = ['Configuration',
               '- num of classes : %d' % num_class,
               '- windows        : %s' % str(windows),
               '- n-gram filters : %s' % str(ngram_filters),
               '- dropout ratio  : %f' % dropout,
               '- epoch          : %f' % epoch,
               '- train batch    : %d' % trn_batch,
               '- dev batch      : %d' % dev_batch,
               '- learning rate  : %f' % learning_rate,
               ]
        logging.info('\n'.join(log))

        self.ctx = ctx
        label_map = None
        word_vsm = FastText(word_vsm)
        ambi_vsm = Word2Vec(ambi_vsm) if ambi_vsm else None
        self.params = self.create_params(word_vsm, ambi_vsm, num_class, windows, ngram_filters,
                                         dropout, label_map)

        cols = {TOKEN: tsv_tok, POS: tsv_pos}
        trn_states = read_tsv(trn_data, cols, self.create_state)
        dev_states = read_tsv(dev_data, cols, self.create_state)

        # optimizer
        loss_func = gluon.loss.SoftmaxCrossEntropyLoss()
        trainer = gluon.Trainer(self.model.collect_params(), 'adagrad',
                                {'learning_rate': learning_rate})
        # train
        best_e, best_eval = -1, -1
        trn_metric = Accuracy()
        dev_metric = Accuracy()

        for e in range(epoch):
            trn_metric.reset()
            dev_metric.reset()

            st = time.time()
            trn_eval = self._train(trn_states, trn_batch, trainer, loss_func, trn_metric)
            mt = time.time()
            dev_eval = self._evaluate(dev_states, dev_batch, dev_metric)
            et = time.time()
            if best_eval < dev_eval:
                best_e, best_eval = e, dev_eval
                self.save(model_path=model_path)

            logging.info(
                '%4d: trn-time: %d, dev-time: %d, trn-acc: %5.2f (%d), dev-acc: %5.2f (%d), '
                'num-class: %d, best-acc: %5.2f @%4d' %
                (e, mt - st, et - mt, trn_eval, trn_metric.total, dev_eval, dev_metric.total,
                 len(self.params.label_map), best_eval, best_e))

    # override
    def save(self, model_path, *args, **kwargs):
        with open(pkl(model_path), 'wb') as f:
            pickle.dump(self.params.label_map, f)
            pickle.dump(self.params.num_class, f)
            pickle.dump(self.params.windows, f)
            pickle.dump(self.params.ngram_filters, f)
            pickle.dump(self.params.dropout, f)
        self.model.save_params(gln(model_path))

    def create_state(self, document):
        return POSState(document, self.params)


# ======================================== Train ========================================

def train_args():
    def int_tuple(s):
        return tuple(map(int, s.split(',')))

    def context(s):
        d = int(s[1:]) if len(s) > 1 else 0
        return mx.cpu(d) if s[0] == 'c' else mx.gpu(d)

    parser = argparse.ArgumentParser('Train: part-of-speech tagging')

    # data
    parser.add_argument('-t', '--trn_path', type=str, metavar='filepath',
                        help='path to the training data (input)')
    parser.add_argument('-d', '--dev_path', type=str, metavar='filepath',
                        help='path to the development data (input)')
    parser.add_argument('-m', '--mod_path', type=str, metavar='filepath', default=None,
                        help='path to the model data (output)')
    parser.add_argument('-l', '--log_path', type=str, metavar='filepath', default='logs',
                        help='path to the log data (output)')

    parser.add_argument('-vt', '--tsv_tok', type=int, metavar='int', default=0,
                        help='the column index of tokens in TSV')
    parser.add_argument('-vp', '--tsv_pos', type=int, metavar='int', default=1,
                        help='the column index of pos-tags in TSV')

    # lexicon
    parser.add_argument('-wv', '--word_vsm', type=str, metavar='filepath',
                        help='vector space model for word embeddings')
    parser.add_argument('-av', '--ambi_vsm', type=str, metavar='filepath', default=None,
                        help='vector space model for ambiguity classes')

    # configuration
    parser.add_argument('-nc', '--num_class', type=int, metavar='int', default=50,
                        help='number of classes')
    parser.add_argument('-cw', '--windows', type=int_tuple, metavar='int[,int]*',
                        default=(-3, -2, -1, 0, 1, 2, 3),
                        help='contextual windows for feature extraction')
    parser.add_argument('-nf', '--ngram_filters', type=int_tuple, metavar='int[,int]*',
                        default=(128, 128, 128, 128, 128),
                        help='number of filters for n-gram convolutions')
    parser.add_argument('-do', '--dropout', type=float, metavar='float', default=0.2,
                        help='dropout')

    parser.add_argument('-cx', '--ctx', type=context, metavar='[cg]\d', default=0,
                        help='device context')
    parser.add_argument('-ep', '--epoch', type=int, metavar='int', default=50,
                        help='number of epochs')
    parser.add_argument('-tb', '--trn_batch', type=int, metavar='int', default=64,
                        help='batch size for training')
    parser.add_argument('-db', '--dev_batch', type=int, metavar='int', default=1024,
                        help='batch size for evaluation')
    parser.add_argument('-lr', '--learning_rate', type=float, metavar='float', default=0.01,
                        help='learning rate')

    args = parser.parse_args()

    return args


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
