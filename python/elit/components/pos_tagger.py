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
from typing import Tuple, List

import mxnet as mx
import numpy as np
from fasttext.model import WordVectorModel
from gensim.models import KeyedVectors

from elit.components.template.lexicon import NLPLexicon, NLPEmbedding
from elit.components.template.model import NLPModel
from elit.components.template.state import NLPState
from elit.components.template.util import argparse_ffnn, argparse_model, argparse_data, read_graphs, create_ffnn
from elit.reader import TSVReader
from elit.structure import NLPGraph, NLPNode

__author__ = 'Jinho D. Choi'


class POSLexicon(NLPLexicon):
    def __init__(self, word2vec: KeyedVectors=None, fasttext: WordVectorModel=None, ambiguity: KeyedVectors=None,
                 output_size: int=50):
        """
        :param word2vec: word embeddings.
        :param fasttext: word embedding with character-based model.
        :param ambiguity: ambiguity classes.
        :param output_size: the number of part-of-speech tags to predict.
        """
        super().__init__(word2vec, fasttext)
        self.ambiguity: NLPEmbedding = NLPEmbedding(ambiguity, 'word', 'ambiguity') if ambiguity else None
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
        node: NLPNode = self.graph.nodes[self.idx_curr]
        if scores is not None: node.pos_scores = scores
        node.pos = label
        self.idx_curr += 1

    @property
    def terminate(self) -> bool:
        return self.idx_curr >= len(self.graph.nodes)

    # ============================== Feature ==============================

    def features(self, node: NLPNode) -> List[np.array]:
        fs = [node.pos_scores if node else self.lex.pos_zeros]
        if self.lex.word2vec:  fs.append(self.lex.word2vec.get(node))
        if self.lex.fasttext:  fs.append(self.lex.fasttext.get(node))
        if self.lex.ambiguity: fs.append(self.lex.ambiguity.get(node))
        return fs


class POSModel(NLPModel):
    def __init__(self, mxmod: mx.module.Module, feature_windows: Tuple[int]=None):
        super().__init__(mxmod, POSState)
        if feature_windows is None: feature_windows = list(range(-3, 4))
        self.feature_windows: Tuple[int] = feature_windows

    # ============================== Feature ==============================

    def x(self, state: POSState) -> np.array:
        vectors = [feature for window in self.feature_windows
                   for feature in state.features(state.get_node(state.idx_curr, window))]
        return np.concatenate(vectors, axis=0)


def parse_args():
    parser = argparse.ArgumentParser('Train a part-of-speech tagger')

    # data
    data = argparse_data(parser, tsv=lambda t: TSVReader(word_index=t[0], pos_index=t[1]))
    data.add_argument('--word2vec', type=str, metavar='path', help='path to the word embedding bin file')
    data.add_argument('--ambiguity', type=str, metavar='path', help='path to the ambiguity class bin file')

    # model
    def feature_windows(s: str):
        return tuple(map(int, s.split(',')))

    model = argparse_model(parser)
    model.add_argument('--feature_windows', type=feature_windows, metavar='int,int*', default=[-3,-2,-1,0,1,2,3],
                       help='window of left context')
    argparse_ffnn(parser)

    return parser.parse_args()


def main():
    # arguments
    logging.basicConfig(filename='log.pos', format='%(message)s', level=logging.INFO)
    args = parse_args()

    # data
    trn_graphs = read_graphs(args.tsv, args.trn_data)
    dev_graphs = read_graphs(args.tsv, args.dev_data)

    # lex
    word2vec = KeyedVectors.load_word2vec_format(args.word2vec, binary=True)
    ambiguity = KeyedVectors.load_word2vec_format(args.ambiguity, binary=True)
    lexicon = POSLexicon(word2vec=word2vec, ambiguity=ambiguity, output_size=args.output_size)

    # model
    mxmod = create_ffnn(args.hidden, args.input_dropout, args.output_size, args.context)
    model = POSModel(mxmod, feature_windows=args.feature_windows)
    model.train(lexicon, trn_graphs, dev_graphs, num_steps=3000, batch_size=args.batch_size)


if __name__ == '__main__':
    main()
