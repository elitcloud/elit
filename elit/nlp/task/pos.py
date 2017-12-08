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
import random
import time
from types import SimpleNamespace

import mxnet as mx
import numpy as np
from mxnet import gluon

from elit.nlp.component import ForwardState, NLPComponent, CNN2DModel, pkl, gln
from elit.nlp.lexicon import LabelMap, FastText, Word2Vec
from elit.nlp.metric import Accuracy
from elit.nlp.structure import TOKEN, POS
from elit.nlp.util import X_ANY, x_extract, read_tsv, get_embeddings, get_loc_embeddings

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
        if params.ambi_vsm: self.embs.append(get_embeddings(params.ambi_vsm, document))
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

    @property
    def x(self):
        """
        :return: the n x d matrix where n = # of windows and d = word_emb.dim + ambi_emb.dim + num_class + 2
        """
        t = len(self.document[self.sen_id])
        l = ([x_extract(self.tok_id, w, t, emb[self.sen_id], zero) for w in self.windows] for emb, zero in self.embs)
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
        ngram_conv = [SimpleNamespace(filters=f, kernel_row=i, activation='relu') for i, f in enumerate(params.ngram_filters, 1)]
        super().__init__(input_col, params.num_class, ngram_conv, params.dropout, **kwargs)


class POSTagger(NLPComponent):
    def __init__(self, ctx, word_vsm, ambi_vsm=None, num_class=50, windows=(-2, -1, 0, 1, 2),
                 ngram_filters=(128, 128, 128, 128, 128), dropout=0.2, label_map=None, model_path=None):
        """
        :param ctx: the context (e.g., CPU or GPU) to process this component.
        :type ctx: mxnet.context.Context
        :param word_vsm: the vector space model for word embeddings.
        :type word_vsm: elit.nlp.lexicon.VectorSpaceModel
        :param ambi_vsm: the vector space model for ambiguity classes.
        :type ambi_vsm: elit.nlp.lexicon.VectorSpaceModel
        :param num_class: the total number of classes to predict.
        :type num_class: int
        :param windows: the contextual windows for feature extraction.
        :type windows: tuple of int
        :param ngram_filters: the number of filters for n-gram convolutions.
        :type ngram_filters: tuple of int
        :param dropout: the dropout ratio.
        :type dropout: float
        :param label_map: the mapping between class labels and their unique IDs.
        :type label_map: elit.nlp.lexicon.LabelMap
        :param model_path: if not None, this component is initialized by objects saved in the model_path.
        :type model_path: str
        """
        if model_path:
            f = open(pkl(model_path), 'rb')
            label_map = pickle.load(f)
            num_class = pickle.load(f)
            windows = pickle.load(f)
            ngram_filters = pickle.load(f)
            dropout = pickle.load(f)
            f.close()

        self.params = self.create_params(word_vsm, ambi_vsm, num_class, windows, ngram_filters, dropout, label_map)
        super().__init__(ctx, POSModel(self.params))

        if model_path:
            self.model.load_params(gln(model_path), ctx=ctx)
        else:
            ini = mx.init.Xavier(magnitude=2.24, rnd_type='gaussian')
            self.model.collect_params().initialize(ini, ctx=ctx)

    def save(self, filepath):
        f = open(pkl(filepath), 'wb')
        pickle.dump(self.params.label_map, f)
        pickle.dump(self.params.num_class, f)
        pickle.dump(self.params.windows, f)
        pickle.dump(self.params.ngram_filters, f)
        pickle.dump(self.params.dropout, f)
        f.close()

        self.model.save_params(gln(filepath))

    def create_state(self, document):
        return POSState(document, self.params)

    @staticmethod
    def create_params(word_vsm, ambi_vsm, num_class, windows, ngram_filters, dropout, label_map):
        return SimpleNamespace(
            word_vsm=word_vsm,
            ambi_vsm=ambi_vsm,
            label_map=label_map or LabelMap(),
            num_class=num_class,
            windows=windows,
            ngram_filters=ngram_filters,
            dropout=dropout,
            zero_output=np.zeros(num_class).astype('float32'))


# ======================================== Train ========================================

def train_args():
    def int_tuple(s):
        return tuple(map(int, s.split(',')))

    def context(s):
        d = int(s[1:]) if len(s) > 1 else 0
        return mx.cpu(d) if s[0] == 'c' else mx.gpu(d)

    parser = argparse.ArgumentParser('Train: part-of-speech tagging')

    # data
    parser.add_argument('-t', '--trn_path', type=str, metavar='filepath', help='path to the training data (input)')
    parser.add_argument('-d', '--dev_path', type=str, metavar='filepath', help='path to the development data (input)')
    parser.add_argument('-m', '--mod_path', type=str, metavar='filepath', default=None, help='path to the model data (output)')
    parser.add_argument('-vt', '--tsv_tok', type=int, metavar='int', default=0, help='the column index of tokens in TSV')
    parser.add_argument('-vp', '--tsv_pos', type=int, metavar='int', default=1, help='the column index of pos-tags in TSV')

    # lexicon
    parser.add_argument('-wv', '--word_vsm', type=str, metavar='filepath', help='vector space model for word embeddings')
    parser.add_argument('-av', '--ambi_vsm', type=str, metavar='filepath', default=None, help='vector space model for ambiguity classes')

    # configuration
    parser.add_argument('-nc', '--num_class', type=int, metavar='int', default=50, help='number of classes')
    parser.add_argument('-cw', '--windows', type=int_tuple, metavar='int[,int]*', default=(-2, -1, 0, 1, 2), help='contextual windows for feature extraction')
    parser.add_argument('-nf', '--ngram_filters', type=int_tuple, metavar='int[,int]*', default=(128,128,128,128,128), help='number of filters for n-gram convolutions')
    parser.add_argument('-do', '--dropout', type=float, metavar='float', default=0.2, help='dropout')

    parser.add_argument('-cx', '--ctx', type=context, metavar='[cg]\d', default=0, help='device context')
    parser.add_argument('-ep', '--epoch', type=int, metavar='int', default=50, help='number of epochs')
    parser.add_argument('-tb', '--trn_batch', type=int, metavar='int', default=64, help='batch size for training')
    parser.add_argument('-db', '--dev_batch', type=int, metavar='int', default=1024, help='batch size for evaluation')
    parser.add_argument('-lr', '--learning_rate', type=float, metavar='float', default=0.01, help='learning rate')

    args = parser.parse_args()

    log = ['Configuration',
           '- train batch    : %d' % args.trn_batch,
           '- learning rate  : %f' % args.learning_rate,
           '- dropout ratio  : %f' % args.dropout,
           '- n-gram filters : %s' % str(args.ngram_filters),
           '- num of classes : %d' % args.num_class,
           '- windows        : %s' % str(args.windows)]

    logging.info('\n'.join(log))
    return args


def train():
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    mx.random.seed(11)
    random.seed(11)

    # processor
    args = train_args()
    word_vsm = FastText(args.word_vsm)
    ambi_vsm = Word2Vec(args.ambi_vsm) if args.ambi_vsm else None
    comp = POSTagger(args.ctx, word_vsm, ambi_vsm, args.num_class, args.windows, args.ngram_filters, args.dropout)

    # states
    cols = {TOKEN: args.tsv_tok, POS: args.tsv_pos}
    trn_states = read_tsv(args.trn_path, cols, comp.create_state)
    dev_states = read_tsv(args.dev_path, cols, comp.create_state)

    # optimizer
    loss_func = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(comp.model.collect_params(), 'adagrad', {'learning_rate': args.learning_rate})

    # train
    best_e, best_eval = -1, -1
    trn_metric = Accuracy()
    dev_metric = Accuracy()

    for e in range(args.epoch):
        trn_metric.reset()
        dev_metric.reset()

        st = time.time()
        trn_eval = comp.train(trn_states, args.trn_batch, trainer, loss_func, trn_metric)
        mt = time.time()
        dev_eval = comp.evaluate(dev_states, args.dev_batch, dev_metric)
        et = time.time()

        if best_eval < dev_eval:
            best_e, best_eval = e, dev_eval
            if args.mod_path: comp.save(args.mod_path+'.'+str(e))

        logging.info(
            '%4d: trn-time: %d, dev-time: %d, trn-acc: %5.2f (%d), dev-acc: %5.2f (%d), num-class: %d, best-acc: %5.2f @%4d' %
            (e, mt-st, et-mt, trn_eval, trn_metric.total, dev_eval, dev_metric.total, len(comp.params.label_map), best_eval, best_e))


if __name__ == '__main__':
    train()
