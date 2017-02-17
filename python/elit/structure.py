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
import functools
from itertools import islice
from typing import Dict
from typing import List
from typing import Union

__author__ = 'Jinho D. Choi'

# fields
BLANK = '_'
ROOT_TAG = '@#r$%'

# delimiters
DELIM_FEAT    = '|'
DELIM_FEAT_KV = '='
DELIM_ARC     = ';'
DELIM_ARC_KV  = ':'


class NLPArc:
    """
    :param node:
    :param label:
    """
    def __init__(self, node=None, label: str=None):
        """
        :param node:
        :type  node: NLPNode
        :param label:
        """
        self.node  = node
        self.label = label

    def __str__(self):
        return DELIM_ARC_KV.join((str(self.node.node_id), self.label))


@functools.total_ordering
class NLPNode:
    """
    :param node_id: node id.
    :param word: word form.
    :param lemma: lemma.
    :param pos: part-of-speech tag.
    :param nament: named entity tag.
    :param feats: extra features.
    """
    def __init__(self, node_id: int=-1, word: str=None, lemma: str=None, pos: str=None, nament: str=None,
                 feats: Dict[str, str]=None):
        # fields
        self.node_id: int = node_id
        self.word: str = word
        self.lemma: str = lemma
        self.pos: str = pos
        self.nament: str = nament
        self.feats: Dict[str, str] = feats or {}

        # dependencies
        self._parent: NLPArc  = None
        self.children: List[NLPArc] = []
        self.secondary_parents: List[NLPArc] = []
        self.secondary_children: List[NLPArc] = []

    def __hash__(self):
        return hash(id(self))

    def __lt__(self, other):
        return self.node_id < other.node_id

    def __eq__(self, other):
        return id(self) == id(other)

    def __str__(self):
        node_id = str(self.node_id)
        word    = self.word if self.word else BLANK
        lemma   = self.lemma if self.lemma else BLANK
        pos     = self.pos if self.pos else BLANK
        nament  = self.nament if self.nament else BLANK
        feats   = DELIM_FEAT.join((DELIM_FEAT_KV.join((k, v)) for k, v in self.feats.items())) if self.feats else BLANK
        head_id = str(self._parent.node.node_id) if self._parent else BLANK
        deprel  = self._parent.label if self._parent and self._parent.label else BLANK
        sheads  = DELIM_ARC.join(str(arc) for arc in self.secondary_parents) if self.secondary_parents else BLANK
        return '\t'.join((node_id, word, lemma, pos, feats, head_id, deprel, sheads, nament))

    @property
    def parent(self) -> Union[NLPArc, None]:
        """
        :return: the arc indicating the primary head and the dependency label if exists; otherwise, None.
        """
        return self._parent

    @parent.setter
    def parent(self, arc: Union[NLPArc, None]):
        """
        :param arc: an arc of (node, label) to be set as the primary parent.
        """
        prev_head = self._parent
        self._parent = arc
        if prev_head: bisect_remove(prev_head.node.children, self)
        if arc: insort_right(arc.node.children, NLPArc(self, arc.label))

    def parent_of(self, node) -> bool:
        """
        :param node:
        :type  node: NLPNode
        :return: True if the node is the parent of this node; otherwise, False.
        """
        return self._parent and self._parent == node

    def add_secondary_parent(self, arc: NLPArc):
        """
        :param arc: an arc of (node, label) to be added as a secondary head.
        """
        insort_right(self.secondary_parents, arc)
        insort_right(arc.node.secondary_children, NLPArc(self, arc.label))

    def remove_secondary_parent(self, node) -> bool:
        """
        :param node: a node to be removed from the secondary head list.
        :type  node: NLPNode
        :return: True if the node is removed from the secondary head list; otherwise, False.
        """
        idx = bisect_index(self.secondary_parents, node)
        if idx >= 0:
            del self.secondary_parents[idx]
            return True
        return False

    @property
    def leftmost_child(self) -> Union[NLPArc, None]:
        """
        :return: the leftmost primary dependent whose token position is on the left-hand side of this node if exists;
                 otherwise, None.
        """
        return self.children[0] if self.children and self.children[0].node < self else None

    @property
    def rightmost_child(self) -> Union[NLPArc, None]:
        """
        :return: the rightmost primary dependent whose token position is on the right-hand side of this node if exists;
                 otherwise, None.
        """
        return self.children[-1] if self.children and self.children[-1].node > self else None

    @property
    def left_nearest_child(self) -> Union[NLPArc, None]:
        """
        :return: the left-nearest primary dependent whose token position is on the left-hand side of this node
                 if exists; otherwise, None.
        """
        idx = bisect_left(self.children, self) - 1
        return self.children[idx] if idx >= 0 else None

    @property
    def right_nearest_child(self) -> Union[NLPArc, None]:
        """
        :return: the right-nearest primary dependent whose token position is on the right-hand side of this node
                 if exists; otherwise, None.
        """
        idx = bisect_right(self.children, self)
        return self.children[idx] if idx < len(self.children) else None

    @property
    def leftmost_sibling(self) -> Union[NLPArc, None]:
        """
        :return: the leftmost primary sibling whose token position is on the left-hand side of this node if exists;
                 otherwise, None.
        """
        return self.head.dependents[0] if self.head and self.head.dependents[0] < self else None

    @property
    def rightmost_sibling(self) -> Union[NLPArc, None]:
        """
        :return: the rightmost primary sibling whose token position is on the right-hand side of this node if exists;
                 otherwise, None.
        """
        return self.head.dependents[-1] if self.head and self.head.dependents[-1] > self else None

    @property
    def left_nearest_sibling(self) -> Union[NLPArc, None]:
        """
        :return: the left-nearest primary sibling whose token position is on the left-hand side of this node if exists;
                 otherwise, None.
        """
        if self.head:
            idx = self.head.dependents.bisect(self) - 1
            return self.head.dependents[idx] if idx >= 0 else None
        return None

    @property
    def right_nearest_sibling(self) -> Union[NLPArc, None]:
        """
        :return: the right-nearest primary sibling whose token position is on the right-hand side of this node
                 if exists; otherwise, None.
        """
        if self.head:
            idx = self.head.dependents.bisect(self) + 1
            return self.head.dependents[idx] if idx < len(self.head.dependents) else None
        return None


class NLPGraph:
    """
    :param nodes: a list of NLP nodes whose heads are not initialized.
    :type  nodes: List[NLPNode]
      An artificial root is automatically added to the front of the node list.
    """
    def __init__(self, nodes: List[NLPNode]=None):
        self.nodes = [NLPNode(node_id=0, word=ROOT_TAG, lemma=ROOT_TAG, pos=ROOT_TAG, nament=ROOT_TAG)]
        if nodes: self.nodes.extend(nodes)

    def __next__(self):
        try: return next(self._iter)
        except StopIteration: raise StopIteration

    def __iter__(self):
        self._iter = islice(self.nodes, 1, len(self.nodes))
        return self

    def __str__(self):
        return '\n'.join(map(str, self))

    def __len__(self):
        return len(self.nodes) - 1


def bisect_left(arcs: List[NLPArc], node: NLPNode, lo: int=0, hi: int=None) -> int:
    """
    :param arcs: a sorted list of arcs.
    :param node: the node to search for.
    :param lo: the lower-bound for search (inclusive).
    :param hi: the upper-bound for search (exclusive).
    :return: the index where to insert the node in the sorted list.
    """
    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(arcs)
    while lo < hi:
        mid = (lo + hi) // 2
        if arcs[mid].node < node: lo = mid + 1
        else: hi = mid
    return lo


def bisect_right(arcs: List[NLPArc], node: NLPNode, lo: int=0, hi: int=None) -> int:
    """
    :param arcs: a sorted list of arcs.
    :param node: the node to search for.
    :param lo: the lower-bound for search (inclusive).
    :param hi: the upper-bound for search (exclusive).
    :return: the index where to insert the node in the sorted list.
    """
    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(arcs)
    while lo < hi:
        mid = (lo + hi) // 2
        if node < arcs[mid].node: hi = mid
        else: lo = mid + 1
    return lo


def bisect_index(arcs: List[NLPArc], node: NLPNode, lo: int=0, hi: int=None) -> int:
    """
    :param arcs: a sorted list of arcs.
    :param node: the node to search for.
    :param lo: the lower-bound for search (inclusive).
    :param hi: the upper-bound for search (exclusive).
    :return: the index of the node in the sorted list of tuples if exists; otherwise, -1.
    """
    idx = bisect_left(arcs, node, lo, hi)
    if idx != len(arcs) and arcs[idx].node == node: return idx
    return -1


def bisect_remove(arcs: List[NLPArc], node: NLPNode, lo: int=0, hi: int=None) -> int:
    """
    :param arcs: a sorted list of arcs.
    :param node: the node to search for.
    :param lo: the lower-bound for search (inclusive).
    :param hi: the upper-bound for search (exclusive).
    :return: the index of the removed item in the sorted list of tuples if exists; otherwise, -1.
    """
    idx = bisect_index(arcs, node, lo, hi)
    if idx >= 0: del arcs[idx]
    return idx


def insort_left(arcs: List[NLPArc], arc: NLPArc, lo: int=0, hi: int=None):
    """
    :param arcs: arcs sorted list of arcs.
    :param arc: the node to be inserted.
    :param lo: the lower-bound for search (inclusive).
    :param hi: the upper-bound for search (exclusive).
    """
    idx = bisect_left(arcs, arc.node, lo, hi)
    arcs.insert(idx, arc)


def insort_right(arcs: List[NLPArc], arc: NLPArc, lo: int=0, hi: int=None):
    """
    :param arcs: arcs sorted list of arcs.
    :param arc: the node to be inserted.
    :param lo: the lower-bound for search (inclusive).
    :param hi: the upper-bound for search (exclusive).
    """
    idx = bisect_right(arcs, arc.node, lo, hi)
    arcs.insert(idx, arc)
