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
from types import SimpleNamespace

import mxnet as mx
import numpy as np
from mxnet import gluon

from elit.component import FFNNModel, LSTMModel, NLPComponent, ForwardState
from elit.utils import TOKEN, NER, F1
from elit.lexicon import LabelMap, FastText, Word2Vec, X_ANY, get_loc_embeddings, get_embeddings, x_extract
from elit.utils.bilou import BILOU
from elit.utils.file import read_tsv
from elit.util import pkl, gln

__author__ = 'Jinho D. Choi'


class NERState(ForwardState):
    def __init__(self, document, params):
        """
        NERState inherits the one-pass, left-to-right tagging strategy from ForwardState.
        :param document: the input document.
        :type document: elit.nlp.structure.Document
        :param params: parameters created by NERecognizer.create_params()
        :type params: SimpleNamespace
        """
        super().__init__(document, params.label_map, params.zero_output, NER)
        self.windows = params.windows
        self.embs = [get_loc_embeddings(document), get_embeddings(params.word_vsm, document)]
        if params.name_vsm:
            self.embs.append(get_embeddings(params.name_vsm, document))
        self.embs.append((self.output, self.zero_output))

    def eval(self, metric):
        """
        :type metric: elit.nlp.metric.F1
        """
        preds = self.labels

        for i, sentence in enumerate(self.document):
            gold = sentence[NER]
            pred = preds[i]

            gold = BILOU.collect(gold)
            auto = BILOU.collect(pred)

            metric.correct += len([1 for k, v in gold.items() if v == auto.get(k, None)])
            metric.p_total += len(gold)
            metric.r_total += len(auto)

    @property
    def x(self):
        """
        :return: the n x d matrix where n = # of feature_windows and d = 2 + word_emb.dim + name_emb.dim + num_class
        """
        t = len(self.document[self.sen_id])
        l = ([x_extract(self.tok_id, w, t, emb[self.sen_id], zero) for w in self.windows] for
             emb, zero in self.embs)
        return np.column_stack(l)


class NERModel(FFNNModel):
    def __init__(self, num_class, input_size):
        """
        :param kwargs: parameters to initialize gluon.Block.
        :type kwargs: dict
        """
        loc_dim = len(X_ANY)
        word_dim = params.word_vsm.dim
        name_dim = params.name_vsm.dim if params.name_vsm else 0

        input_col = loc_dim + word_dim + name_dim + params.num_class
        ngram_conv = [SimpleNamespace(filters=f, kernel_row=i, activation='relu') for i, f in
                      enumerate(params.ngram_filters, 1)]
        super().__init__(params.num_class, input_col, ngram_conv, params.dropout, **kwargs)


class NERModelLSTM(LSTMModel):
    def __init__(self, params, **kwargs):
        """
        :param params: parameters to initialize POSModel.
        :type params: SimpleNamespace
        :param kwargs: parameters to initialize gluon.Block.
        :type kwargs: dict
        """
        loc_dim = len(X_ANY)
        word_dim = params.word_vsm.dim
        name_dim = params.name_vsm.dim if params.name_vsm else 0

        input_col = loc_dim + word_dim + name_dim
        n_hidden = 128
        super().__init__(input_col, params.num_class, n_hidden, params.dropout, **kwargs)


class NERModelLR(gluon.Block):
    def __init__(self, params, **kwargs):
        """
        :param params: parameters to initialize POSModel.
        :type params: SimpleNamespace
        :param kwargs: parameters to initialize gluon.Block.
        :type kwargs: dict
        """
        super().__init__(**kwargs)
        self.dropout = gluon.nn.Dropout(0.2)
        self.out = gluon.nn.Dense(params.num_class)

    def forward(self, x):
        x = self.dropout(x)
        x = self.out(x)
        return x


class NERecognizer(NLPComponent):
    # def __init__(self, ctx, word_vsm, name_vsm=None, num_class=17, feature_windows=(-2, -1, 0, 1, 2),
    #              ngram_filters=(128, 128, 128, 128, 128), dropout=0.2, label_map=None, model_path=None):
    """
    :param ctx: the context (e.g., CPU or GPU) to process this component.
    :type ctx: mxnet.context.Context
    :param word_vsm: the vector space model for word embeddings.
    :type word_vsm: elit.nlp.lexicon.VectorSpaceModel
    :param name_vsm: the vector space model for ambiguity classes.
    :type name_vsm: elit.nlp.lexicon.VectorSpaceModel
    :param num_class: the total number of classes to predict.
    :type num_class: int
    :param feature_windows: the contextual feature_windows for feature extraction.
    :type feature_windows: tuple of int
    :param ngram_filters: the number of filters for n-gram conv2d.
    :type ngram_filters: tuple of int
    :param dropout: the dropout ratio.
    :type dropout: float
    :param label_map: the mapping between class labels and their unique IDs.
    :type label_map: elit.nlp.lexicon.LabelMap
    :param model_path: if not None, this component is initialized by objects saved in the model_path.
    :type model_path: str
    """

    # self.params = self.create_params(word_vsm, name_vsm, num_class, feature_windows, ngram_filters, dropout, label_map)
    # super().__init__(ctx, NERModel(self.params))

    # override
    def load(self, model_path, *args, **kwargs):
        ctx, word_vsm, name_vsm, num_class, windows, ngram_filters, dropout = args
        label_map = None
        if model_path:
            with open(pkl(model_path), 'rb') as f:
                label_map = pickle.load(f)
                num_class = pickle.load(f)
                windows = pickle.load(f)
                ngram_filters = pickle.load(f)
                dropout = pickle.load(f)

        self.params = self.create_params(word_vsm, name_vsm, num_class, windows, ngram_filters,
                                         dropout, label_map)
        super().__init__(ctx=ctx, model=NERModel(, self.params,)

        if model_path:
            self.model.load_params(gln(model_path), ctx=self.ctx)
        else:
            ini = mx.init.Xavier(magnitude=2.24, rnd_type='gaussian')
            self.model.collect_params().initialize(ini, ctx=self.ctx)

    # override
    def save(self, filepath, *args, **kwargs):
        with open(pkl(filepath), 'wb') as f:
            pickle.dump(self.params.label_map, f)
            pickle.dump(self.params.num_class, f)
            pickle.dump(self.params.windows, f)
            pickle.dump(self.params.ngram_filters, f)
            pickle.dump(self.params.dropout, f)

        self.model.save_params(gln(filepath))

    def create_state(self, document):
        return NERState(document, self.params)



# ======================================== Train ========================================

def train_args():
    def int_tuple(s):
        return tuple(map(int, s.split(',')))

    def context(s):
        d = int(s[1:]) if len(s) > 1 else 0
        return mx.cpu(d) if s[0] == 'c' else mx.gpu(d)

    parser = argparse.ArgumentParser('Train: named entity recognition')

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
    parser.add_argument('-vp', '--tsv_ner', type=int, metavar='int', default=4,
                        help='the column index of pos-tags in TSV')

    # lexicon
    parser.add_argument('-wv', '--word_vsm', type=str, metavar='filepath',
                        help='vector space model for word embeddings')
    parser.add_argument('-nv', '--name_vsm', type=str, metavar='filepath', default=None,
                        help='vector space model for named entity gazetteers')

    # configuration
    parser.add_argument('-nc', '--num_class', type=int, metavar='int', default=50,
                        help='number of classes')
    parser.add_argument('-cw', '--feature_windows', type=int_tuple, metavar='int[,int]*',
                        default=(-3, -2, -1, 0, 1, 2, 3),
                        help='contextual feature_windows for feature extraction')
    parser.add_argument('-nf', '--ngram_filters', type=int_tuple, metavar='int[,int]*',
                        default=(128, 128, 128, 128, 128),
                        help='number of filters for n-gram conv2d')
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
    args = train_args()
    if not args.trn_path:
        predict(args)
        return

    logging.basicConfig(filename=args.log_path + '.log', filemode='w', format='%(message)s',
                        level=logging.INFO)

    log = ['Configuration',
           '- train batch    : %d' % args.trn_batch,
           '- learning rate  : %f' % args.learning_rate,
           '- dropout ratio  : %f' % args.dropout,
           '- n-gram filters : %s' % str(args.ngram_filters),
           '- num of classes : %d' % args.num_class,
           '- feature_windows        : %s' % str(args.windows)]

    logging.info('\n'.join(log))
    # mx.random.seed(11)
    # random.seed(11)
    # processor

    word_vsm = FastText(args.word_vsm)
    name_vsm = Word2Vec(args.name_vsm) if args.name_vsm else None
    # comp = NERecognizer(args.ctx, word_vsm, name_vsm, args.num_class, args.feature_windows, args.ngram_filters, args.dropout)
    comp = NERecognizer()
    comp.load(args.mod_path, args.ctx, word_vsm, name_vsm, args.num_class, args.windows,
              args.ngram_filters, args.dropout)

    # states
    cols = {TOKEN: args.tsv_tok, NER: args.tsv_ner}
    trn_states = read_tsv(args.trn_path, cols, comp.create_state)
    dev_states = read_tsv(args.dev_path, cols, comp.create_state)

    # optimizer
    loss_func = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(comp.model.collect_params(), 'adagrad',
                            {'learning_rate': args.learning_rate})

    # train
    best_e, best_eval = -1, -1
    trn_metric = F1()
    dev_metric = F1()

    for e in range(args.epoch):
        trn_metric.reset()
        dev_metric.reset()

        # call the overridden method instead

        # st = time.time()
        # trn_eval = comp.train(trn_states, args.trn_batch, trainer, loss_func, trn_metric)
        # mt = time.time()
        # dev_eval = comp.evaluate(dev_states, args.dev_batch, dev_metric)
        # et = time.time()

        st, mt, et, trn_eval, dev_eval = comp.train(trn_states, dev_states, args.trn_batch, trainer,
                                                    loss_func, trn_metric, args.dev_batch,
                                                    dev_metric)

        if best_eval < dev_eval[0]:
            best_e, best_eval = e, dev_eval[0]
            # if args.mod_path: comp.save(args.mod_path+'.'+str(e))
            if args.mod_path: comp.save(args.mod_path)

        logging.info(
            '%4d: trn-time: %d, dev-time: %d, trn-f1: %5.2f, dev-f1: %5.2f, num-class: %d, best-acc: %5.2f @%4d' %
            (e, mt - st, et - mt, trn_eval[0], dev_eval[0], len(comp.params.label_map), best_eval,
             best_e))


def predict(args):
    word_vsm = FastText(args.word_vsm)
    name_vsm = Word2Vec(args.name_vsm) if args.name_vsm else None
    comp = NERecognizer()
    comp.load(args.mod_path, args.ctx, word_vsm, name_vsm, args.num_class, args.windows,
              args.ngram_filters, args.dropout)
    cols = {TOKEN: args.tsv_tok, NER: args.tsv_ner}
    dev_states = read_tsv(args.dev_path, cols, comp.create_state)
    # train
    # best_e, best_eval = -1, -1
    # trn_metric = Accuracy()
    dev_metric = F1()
    dev_eval = comp.evaluate(dev_states, args.dev_batch, dev_metric)
    print("Test accuracy for %s is %5.2f" % (args.mod_path, dev_eval))


if __name__ == '__main__':
    train()
