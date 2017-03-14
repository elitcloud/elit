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
from elit.components.template.lexicon import *
from elit.components.template.state import *
from elit.components.template.model import *

__author__ = 'Jinho D. Choi'


class POSLexicon(NLPLexicon):
    def __init__(self, word_embeddings: KeyedVectors, ambiguity_classes: KeyedVectors,
                 left_window: int=-3, right_window: int=3):
        """
        :param word_embeddings:
        :param ambiguity_classes:
        :param left_window: the leftmost contextual node to extract features from (inclusive).
        :param right_window: the rightmost contextual node to extract features from (inclusive).
        """
        super().__init__(word_embeddings)
        self.ambiguity_classes: KeyedVectors = self._init_vectors(ambiguity_classes)
        self.windows = range(left_window, right_window + 1)

    # ============================== Initialization ==============================

    def init(self, graph: NLPGraph):
        self._init(graph, self.word_embeddings,   'word', 'word_embedding')
        self._init(graph, self.ambiguity_classes, 'word', 'ambiguity_class')
        self.init_pos_embeddings(graph)

    def init_pos_embeddings(self, graph: NLPGraph):
        default = self._get_default_vector(self.ambiguity_classes)
        for node in graph: node.pos_embedding = default

    # ============================== Feature ==============================

    def features(self, node: NLPNode) -> filter:
        if node is None:
            word_emb = self._get_default_vector(self.word_embeddings) if self.word_embeddings else None
            ambi_cls = self._get_default_vector(self.ambiguity_classes) if self.ambiguity_classes else None
        else:
            word_emb = node.word_embedding if self.word_embeddings else None
            ambi_cls = node.ambiguity_class if self.ambiguity_classes else None

        return filter(None, [word_emb, ambi_cls])


class POSState(NLPState):
    def __init__(self, lexicon: POSLexicon, graph: NLPGraph, train=False):
        super().__init__(lexicon, graph)
        self.gold_tags = [node.set_pos(None) for node in self.graph] if train else None
        self.idx_curr: int = 1

    def reset(self):
        self.lexicon.init_pos_embeddings(self.graph)
        for node in self.graph: node.pos = None
        self.idx_curr = 1

    # ============================== Oracle ==============================

    @property
    def gold_label(self) -> str:
        return self.gold_tags[self.idx_curr - 1] if self.gold_tags else None

    def eval_counts(self) -> Union[np.array, None]:
        if self.gold_tags is None: return None

        correct = 0
        for i, node in enumerate(self.graph):
            if node.pos == self.gold_tags[i]:
                correct += 1

        return np.array([len(self.graph), correct])

    # ============================== Transition ==============================

    def process(self, label: Union[str, Tuple[np.array, NLPModel]]):
        curr = self.graph.nodes[self.idx_curr]

        if type(label) is not str:
            ys: np.array = label[0]
            model: NLPModel = label[1]
            curr.pos_output = ys
            label = model.get_label(np.argmax(ys))

        curr.pos = label
        curr = label
        self.idx_curr += 1

    @property
    def terminate(self) -> bool:
        return self.idx_curr >= len(self.graph.nodes)

    @property
    def x(self) -> np.array:
        vectors = [feature for window in self.lexicon.windows
                   for feature in self.lexicon.features(self.get_node(self.idx_curr, window))]
        return np.concatenate(vectors, axis=0)


class POSModel(NLPModel):
    def __init__(self, hidden_layers: List[Tuple[int, str]], output_layer: Tuple[int, str],
                 context: mx.context.Context):
        super().__init__()
        self.model = FeedForwardNeuralNetwork(hidden_layers, output_layer, context)


