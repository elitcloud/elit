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
from gensim.models import KeyedVectors

from elit.components.template.lexicon import NLPLexicon
from elit.components.template.model import NLPModel
from elit.components.template.state import NLPState
from elit.components.template.util import argparse_ffnn, argparse_model, argparse_data, read_graphs
from elit.reader import TSVReader
from elit.structure import NLPGraph, NLPNode

__author__ = 'Jinho D. Choi'


class POSLexicon(NLPLexicon):
    def __init__(self, word_embeddings: KeyedVectors, ambiguity_classes: KeyedVectors):
        super().__init__(word_embeddings)
        self.ambiguity_classes: KeyedVectors = self._init_vectors(ambiguity_classes)

    # ============================== Initialization ==============================

    def init(self, graph: NLPGraph):
        self._init(graph, self.word_embeddings,   'word', 'word_embedding')
        self._init(graph, self.ambiguity_classes, 'word', 'ambiguity_class')
        self.init_pos_embeddings(graph)

    def init_pos_embeddings(self, graph: NLPGraph):
        default = self.get_default_vector(self.ambiguity_classes)
        for node in graph: node.pos_embedding = default


class POSState(NLPState):
    def __init__(self, lexicon: POSLexicon, graph: NLPGraph, save_gold=False):
        super().__init__(lexicon, graph)
        self.golds = [node.set_pos(None) for node in self.graph] if save_gold else None
        self.idx_curr: int = 1

    def reset(self):
        self.lexicon.init_pos_embeddings(self.graph)
        for node in self.graph: node.pos = None
        self.idx_curr = 1

    # ============================== Oracle ==============================

    @property
    def gold_label(self) -> str:
        return self.golds[self.idx_curr - 1] if self.golds else None

    def eval_counts(self) -> np.array:
        if self.golds is None: return np.array([])

        correct = 0
        for i, node in enumerate(self.graph):
            if node.pos == self.golds[i]:
                correct += 1

        return np.array([len(self.graph), correct])

    # ============================== Transition ==============================

    def process(self, label: str, scores: np.array = None):
        curr: NLPNode = self.graph.nodes[self.idx_curr]
        if scores: curr.pos_embedding = scores

        curr.pos = label
        self.idx_curr += 1

    @property
    def terminate(self) -> bool:
        return self.idx_curr >= len(self.graph.nodes)

    # ============================== Feature ==============================

    def features(self, node: NLPNode) -> List[np.array]:
        word_emb = node.word_embedding if node else self.lexicon.get_default_vector(self.lexicon.word_embeddings)
        ambi_cls = node.ambiguity_class if node else self.lexicon.get_default_vector(self.lexicon.ambiguity_classes)
        # pos_emb = node.pos_embedding if node else self._get_default_vector(self.ambiguity_classes)
        return [word_emb, ambi_cls]


class POSModel(NLPModel):
    def __init__(self, lexicon: POSLexicon,
                 context: mx.context.Context = mx.cpu(),
                 hidden: List[Tuple[int, str, float]]=None, input_dropout: float=0, output_size: int=50,
                 feature_left_window=-3, feature_right_window=3):
        super().__init__(lexicon)
        if hidden is None: hidden = []
        self.mxmod = self.create_ffnn(hidden, input_dropout, output_size, context)
        self.feature_windows = range(feature_left_window, feature_right_window+1)

    # ============================== State ==============================

    def create_state(self, graph: NLPGraph, save_gold: bool=False) -> POSState:
        return POSState(self.lexicon, graph, save_gold)

    # ============================== Feature ==============================

    def x(self, state: POSState) -> np.array:
        vectors = [feature for window in self.feature_windows
                   for feature in state.features(state.get_node(state.idx_curr, window))]
        return np.concatenate(vectors, axis=0)


def parse_args():
    parser = argparse.ArgumentParser('Train a part-of-speech tagger')

    # data
    data = argparse_data(parser, tsv=lambda t: TSVReader(word_index=t[0], pos_index=t[1]))
    data.add_argument('--word_embeddings', type=str, metavar='path', help='path to the word embedding bin file')
    data.add_argument('--ambiguity_classes', type=str, metavar='path', help='path to the ambiguity class bin file')

    # model
    model = argparse_model(parser)
    model.add_argument('--feature_left_window', type=int, metavar='int', default=-3, help='window of left context')
    model.add_argument('--feature_right_window', type=int, metavar='int', default=3, help='window of right context')
    argparse_ffnn(parser)

    return parser.parse_args()


def main():
    # arguments
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    args = parse_args()

    # data
    trn_graphs = read_graphs(args.tsv, args.trn_data)
    dev_graphs = read_graphs(args.tsv, args.dev_data)

    # lexicon
    word_embeddings = KeyedVectors.load_word2vec_format(args.word_embeddings, binary=True)
    ambiguity_classes = KeyedVectors.load_word2vec_format(args.ambiguity_classes, binary=True)
    lexicon = POSLexicon(word_embeddings=word_embeddings, ambiguity_classes=ambiguity_classes)

    # model
    model = POSModel(lexicon, context=args.context,
                     hidden=args.hidden, input_dropout=args.input_dropout, output_size=args.output_size,
                     feature_left_window=args.feature_left_window, feature_right_window=args.feature_right_window)

    model.train(trn_graphs=trn_graphs, dev_graphs=dev_graphs, num_steps=50, batch_size=args.batch_size)


if __name__ == '__main__':
    main()
