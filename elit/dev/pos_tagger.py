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

import sys
sys.path.append("/Users/tlee/Desktop/elit")

import argparse
import logging
from typing import Tuple, List

import mxnet as mx
import numpy as np
import fasttext
from fasttext.model import WordVectorModel
from gensim.models import KeyedVectors

from elit.dev.template.lexicon import NLPLexiconMapper, NLPEmbedding
from elit.dev.template.model import NLPModel
from elit.dev.template.state import NLPState
from elit.dev.template.util import argparse_ffnn, argparse_model, argparse_data, read_graphs, argparse_lexicon, conv_pool
from elit.dev.reader import TSVReader
from elit.dev.structure import NLPGraph, NLPToken

__author__ = 'Jinho D. Choi'


class POSLexicon(NLPLexiconMapper):
    def __init__(self, w2v: KeyedVectors=None, f2v: WordVectorModel=None, a2v: KeyedVectors=None,
                 output_size: int=50):
        """
        :param w2v: word embeddings from word2vec.
        :param f2v: word embeddings from fasttext.
        :param a2v: a2v classes.
        :param output_size: the number of part-of-speech tags to predict.
        """
        super().__init__(w2v, f2v)
        self.a2v: NLPEmbedding = NLPEmbedding(a2v, 'word', 'a2v') if a2v else None
        self.pos_zeros = np.zeros((output_size,)).astype('float32')


class POSState(NLPState):
    def __init__(self, graph: NLPGraph, lexicon: POSLexicon, save_gold=False):
        super().__init__(graph)
        self.lex: POSLexicon = lexicon

        # reset
        self.golds = [node.set_pos(None) for node in self.graph] if save_gold else None
        for node in self.graph: node.pos_scores = lexicon.pos_zeros
        self.idx_curr: int = 1

    def reset(self):
        for node in self.graph:
            node.pos = None
            node.pos_scores = self.lex.pos_zeros

        self.idx_curr = 1
        self.reset_count += 1

    # ============================== Oracle ==============================

    @property
    def gold(self) -> str:
        return self.golds[self.idx_curr - 1] if self.golds else None

    def eval(self, stats: np.array) -> float:
        if self.golds is None: return 0

        stats[0] += len(self.graph)
        for i, node in enumerate(self.graph):
            if node.pos == self.golds[i]:
                stats[1] += 1

        return stats[1] / stats[0]

    # ============================== Transition ==============================

    def process(self, label: str, scores: np.array=None):
        node: NLPToken = self.graph.nodes[self.idx_curr]
        if scores is not None: node.pos_scores = scores
        node.pos = label
        self.idx_curr += 1

    @property
    def terminate(self) -> bool:
        return self.idx_curr >= len(self.graph.nodes)

    # ============================== Feature ==============================

    def features(self, node: NLPToken) -> List[np.array]:
        fs = [node.pos_scores if node else self.lex.pos_zeros]
        if self.lex.w2v: fs.append(self.lex.w2v.score())
        if self.lex.f2v: fs.append(self.lex.f2v.score())
        if self.lex.a2v: fs.append(self.lex.a2v.get(node))
        return fs


class POSModel(NLPModel):
    def __init__(self, batch_size=64, num_label: int=50, feature_context: Tuple = (-2, -1, 0, 1, 2),
                 context: mx.context.Context=mx.cpu(0), w2v_dim=100,
                 ngram_filter_list=(1, 2, 3), ngram_filter: int=32):
        super().__init__(POSState, batch_size)
        self.mxmod: mx.module.Module = self.init_mxmod(batch_size=batch_size,
                                                       num_label=num_label,
                                                       num_feature=len(feature_context),
                                                       context=context,
                                                       w2v_dim=w2v_dim,
                                                       ngram_filter_list=ngram_filter_list,
                                                       ngram_filter=ngram_filter)
        self.feature_context: Tuple[int] = feature_context

    # ============================== Feature ==============================

    def x(self, state: POSState) -> np.array:
        vectors_pos_score = [state.features(state.get_node(state.idx_curr, window))[0] for window in self.feature_context]
        vectors_f2v = [state.features(state.get_node(state.idx_curr, window))[1] for window in self.feature_context]
        vectors_a2v = [state.features(state.get_node(state.idx_curr, window))[2] for window in self.feature_context]

        out_f2v = np.asarray(vectors_f2v)
        out_a2v = np.asarray(vectors_a2v)
        out = (out_f2v, out_a2v)
        return out

    # ============================== Module ==============================

    def init_mxmod(self, batch_size: int, num_label: int, num_feature: int, context: mx.context.Context, w2v_dim: int,
                   ngram_filter_list: Tuple, ngram_filter: int) -> mx.module.Module:
        batch_size = -1
        # n-gram convolution for f2v and a2v
        input_f2v = mx.sym.Variable('data_f2v')
        input_a2v = mx.sym.Variable('data_a2v')
        input_pool2v = mx.sym.Variable('data_pool2v')

        print ("NUMBER FEATURE: ", num_feature)
        conv_input_f2v = mx.sym.Reshape(data=input_f2v, shape=(batch_size, 1, num_feature, w2v_dim))
        conv_input_a2v = mx.sym.Reshape(data=input_a2v, shape=(batch_size, 1, num_feature, 50))
        conv_input_pool2v = mx.sym.Reshape(data=input_pool2v, shape=(batch_size, 1, 1, 192))


        pooled_1 = [conv_pool(conv_input_f2v, conv_kernel=(filter, w2v_dim), num_filter=ngram_filter, act_type='relu',
                            pool_kernel=(num_feature - filter + 1, 1), pool_stride=(1, 1))
                  for filter in ngram_filter_list]

        pooled_2 = [conv_pool(conv_input_a2v, conv_kernel=(filter, 50), num_filter=ngram_filter, act_type='relu',
                            pool_kernel=(num_feature - filter + 1, 1), pool_stride=(1, 1))
                  for filter in ngram_filter_list]


        # concatenate pooled features from f2v and a2v
        pooled = pooled_1 + pooled_2
        concat = mx.sym.Concat(*pooled, dim=1)

        h_pool = mx.sym.Reshape(name="concat_pooling", data=concat, shape=(batch_size, 2 * ngram_filter * len(ngram_filter_list)))
      # h_pool = mx.sym.Dropout(x=h_pool, p=dropouts[0]) if dropouts[0] > 0.0 else h_pool

        # block gradient
        h_pool = mx.sym.BlockGrad(h_pool, name="concat_pooling")

        # fully connected
        fc_weight = mx.sym.Variable('fc_weight')
        fc_bias = mx.sym.Variable('fc_bias')
        fc = mx.sym.FullyConnected(data=h_pool, weight=fc_weight, bias=fc_bias, num_hidden=num_label)

        output = mx.sym.Variable('softmax_label')
        sm = mx.sym.SoftmaxOutput(data=fc, label=output, name='softmax')

        # mx module now contains softmax and pool output
        final = mx.sym.Group([sm, h_pool])
        return mx.mod.Module(symbol=final, data_names=('data_f2v', 'data_a2v'), context=context)


def parse_args():
    parser = argparse.ArgumentParser('Train a part-of-speech tagger')

    # x
    args= argparse_data(parser, tsv=lambda t: TSVReader(word_index=t[0], pos_index=t[1]))
    args.add_argument('--log', type=str, metavar='filepath', help='path to the logging file')

    # lexicon
    args = argparse_lexicon(parser)
    args.add_argument('--a2v', type=str, metavar='filepath', help='path to the ambiguity class bin file')

    # model
    def feature_context(s: str):
        return tuple(map(int, s.split(',')))

    argparse_ffnn(parser)
    args = argparse_model(parser)
    args.add_argument('--feature_context', type=feature_context, metavar='int,int*', default=[-2, -1, 0, 1, 2],
                      help='context window for feature extraction')

    return parser.parse_args()


def main():
    # arguments
    args = parse_args()
    if args.log: logging.basicConfig(filename=args.log, format='%(message)s', level=logging.INFO)
    else: logging.basicConfig(format='%(message)s', level=logging.INFO)

    # x
    trn_graphs = read_graphs(args.tsv, args.trn_data)
    dev_graphs = read_graphs(args.tsv, args.dev_data)

    # lexicon
    w2v = KeyedVectors.load_word2vec_format(args.w2v, binary=True) if args.w2v else None
    f2v = fasttext.load_model(args.f2v) if args.f2v else None
    a2v = KeyedVectors.load_word2vec_format(args.a2v, binary=True) if args.a2v else None

    lexicon = POSLexicon(w2v=w2v, f2v=f2v, a2v=a2v, output_size=args.output_size)

    # model
    model = POSModel(feature_context=args.feature_context, batch_size=64, w2v_dim=100)
    model.train(trn_graphs, dev_graphs, lexicon, num_steps=args.num_steps,
                bagging_ratio=args.bagging_ratio, optimizer=args.optimizer, force_init=True)


if __name__ == '__main__':
    main()
