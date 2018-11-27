# ========================================================================
# Copyright 2018 ELIT
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
import itertools
from typing import List, Tuple, Sequence

__author__ = 'Jinho D. Choi'

DOC_ID = 'doc_id'  # document ID
SENS = 'sens'  # sentences
SID = 'sid'  # sentence ID
TOK = 'tok'
OFF = 'off'
LEM = 'lem'
POS = 'pos'
MORPH = 'morph'
NER = 'ner'
DEP = 'dep'
COREF = 'coref'
SENTI = 'senti'

# sentiment analysis labels
SENTI_POSITIVE = 'p'
SENTI_NEGATIVE = 'n'
SENTI_NEUTRAL = '0'


class Sentence(dict):
    def __init__(self, d=None, **kwargs):
        """
        :param d: a dictionary, if not None, all of whose fields are added to this sentence.
        :type d: dict
        :param kwargs: additional fields to be added; if keys already exist; the values are overwritten with these.
        """
        super().__init__()
        self._iter = -1

        if d is not None:
            self.update(d)
        self.update(kwargs)
        self._tokens = self.setdefault(TOK, [])

    def __len__(self):
        """
        :return: the number of tokens in the sentence.
        """
        return len(self.tokens)

    def __iter__(self):
        self._iter = -1
        return self

    def __next__(self):
        self._iter += 1
        if self._iter >= len(self.tokens):
            raise StopIteration
        return self._tokens[self._iter]

    @property
    def tokens(self):
        """
        :return: the list of tokens in the sentence.
        :rtype: list of str
        """
        return self._tokens

    @property
    def part_of_speech_tags(self):
        """
        :return: the list of part-of-speech tags corresponding to the tokens in this sentence.
        :rtype: list of str
        """
        return self[POS]


class Document(dict):
    def __init__(self, d=None, **kwargs):
        """
        :param d: a dictionary, if not None, all of whose fields are added to this document.
        :type d: dict
        :param kwargs: additional fields to be added; if keys already exist; the values are overwritten with these.
        """
        super().__init__()
        self._iter = -1

        if d is not None:
            self.update(d)
        self.update(kwargs)
        self._sentences = self.setdefault(SENS, [])

    def __len__(self):
        """
        :return: the number of sentences in the document.
        :rtype: int
        """
        return len(self.sentences)

    def __iter__(self):
        self._iter = -1
        return self

    def __next__(self):
        self._iter += 1
        if self._iter >= len(self.sentences):
            raise StopIteration
        return self._sentences[self._iter]

    @property
    def sentences(self) -> List[Sentence]:
        """
        :return: the list of sentences in the document.
        :rtype: list of Sentence
        """
        return self._sentences

    @property
    def tokens(self):
        """
        :return: list of tokens across all sentences.
        :rtype: list of str
        """
        tokens = []
        for sentence in self.sentences:
            tokens.extend(sentence.tokens)
        return tokens

    def add_sentence(self, sentence: Sentence):
        self.sentences.append(sentence)

    def add_sentences(self, sentences: Sequence[Sentence]):
        self.sentences.extend(sentences)


def to_gold(key: str) -> str:
    return key + '-gold'


def to_out(key: str) -> str:
    return key + '-out'


class BILOU(object):
    B = 'B'  # beginning
    I = 'I'  # inside
    L = 'L'  # last
    O = 'O'  # outside
    U = 'U'  # unit

    @classmethod
    def to_chunks(cls, tags: List[str], fix: bool = False) -> List[Tuple[int, int, str]]:
        """
        :param tags: a list of tags encoded by BILOU.
        :param fix: if True, fixes potential mismatches in BILOU (see :meth:`heuristic_fix`).
        :return: a list of tuples where each tuple contains (begin index (inclusive), end index (exclusive), label) of the chunk.
        """
        if fix:
            cls.heuristic_fix(tags)
        chunks = []
        begin = -1

        for i, tag in enumerate(tags):
            t = tag[0]

            if t == cls.B:
                begin = i
            elif t == cls.I:
                pass
            elif t == cls.L:
                if begin >= 0:
                    chunks.append((begin, i + 1, tag[2:]))
                begin = -1
            elif t == cls.O:
                begin = -1
            elif t == cls.U:
                chunks.append((i, i + 1, tag[2:]))
                begin = -1

        return chunks

    @classmethod
    def heuristic_fix(cls, tags):
        """
        Use heuristics to fix potential mismatches in BILOU.
        :param tags: a list of tags encoded by BLIOU.
        """

        def fix(i, pt, ct, t1, t2):
            if pt == ct:
                tags[i][0] = t1
            else:
                tags[i - 1][0] = t2

        def aux(i):
            p = tags[i - 1][0]
            c = tags[i][0]
            pt = tags[i - 1][1:]
            ct = tags[i][1:]

            if p == cls.B:
                if c == cls.B:
                    fix(i, pt, ct, cls.I, cls.U)  # BB -> BI or UB
                elif c == cls.U:
                    fix(i, pt, ct, cls.L, cls.U)  # BU -> BL or UU
                elif c == cls.O:
                    tags[i - 1][0] = cls.U  # BO -> UO
            elif p == cls.I:
                if c == cls.B:
                    fix(i, pt, ct, cls.I, cls.L)  # IB -> II or LB
                elif c == cls.U:
                    fix(i, pt, ct, cls.I, cls.L)  # IU -> II or LU
                elif c == cls.O:
                    tags[i - 1][0] = cls.L  # IO -> LO
            elif p == cls.L:
                if c == cls.I:
                    fix(i, pt, ct, cls.I, cls.B)  # LI -> II or LB
                elif c == cls.L:
                    fix(i, pt, ct, cls.I, cls.B)  # LL -> IL or LB
            elif p == cls.O:
                if c == cls.I:
                    tags[i][0] = cls.B  # OI -> OB
                elif c == cls.L:
                    tags[i][0] = cls.B  # OL -> OB
            elif p == cls.U:
                if c == cls.I:
                    fix(i, pt, ct, cls.B, cls.B)  # UI -> BI or UB
                elif c == cls.L:
                    fix(i, pt, ct, cls.B, cls.B)  # UL -> BL or UB

        for idx in range(1, len(tags)):
            aux(idx)
        prev = tags[-1][0]

        if prev == cls.B:
            tags[-1][0] = cls.U
        elif prev == cls.I:
            tags[-1][0] = cls.L


class Node(object):
    """
    Node is used for constructing the NLP structure.

    """

    def __init__(self):
        self._parent = None
        self._left_sibling = None
        self._right_sibling = None
        self._children = []

    @property
    def parent(self):
        """

        :return: parent of current node
        :rtype: Node
        """
        return self._parent

    @parent.setter
    def parent(self, node):
        if not isinstance(node, Node):
            raise ValueError('parent must be an Node')
        if self.has_parent():
            # not sure what's going on here?
            self.parent.remove_child(self)
        else:
            node.add_child(self)

    @property
    def left_sibling(self):
        """

        :return: left libling of current node
        :rtype: Node
        """
        return self._left_sibling

    @left_sibling.setter
    def left_sibling(self, node):
        if not isinstance(node, Node):
            raise ValueError('sibling must be an Node')
        if node is not None:
            self._left_sibling = node

    @property
    def right_sibling(self):
        """

        :return: right libling of current node
        :rtype: Node
        """
        return self.right_sibling

    @right_sibling.setter
    def right_sibling(self, node):
        if not isinstance(node, Node):
            raise ValueError('sibling must be an Node')
        if node is not None:
            self._right_sibling = node

    @property
    def children(self):
        """
        the list of children

        :return: the list of children
        :rtype: list
        """
        return self._children

    def index_of(self, child):
        """
        the index of the child

        :type child: Node
        :param child: a child node
        :return: the index of the child node. return None if the given node is not a child
        :rtype: int or None
        """
        try:
            return self.children.index(child)
        except ValueError:
            return None

    # Child

    def child(self, index):
        """


        :type index: int
        :param index: index of a child node
        :return: the index'th child of this node if exists; otherwise, None
        :rtype: Node or None
        """
        try:
            return self.children[index]
        except IndexError:
            return None

    def first_child(self):
        """

        :return: the first child of this node if exists; otherwise, None
        :rtype: Node or None
        """
        # return self.child(0)
        return self.find_child_by(index=0)

    def last_child(self):
        """

        :return: the last child of this node if exists; otherwise, None
        :rtype: Node or None
        """
        # return self.child(-1)
        return self.find_child_by(index=-1)

    def find_child_by(self, matcher=lambda n: n, index=0):
        """

        :type matcher: lambda
        :type index: int
        :param matcher: the condition
        :param index: displacement (0: 1st, 1: 2nd, etc.)
        :return: the order'th child matching the specific condition of this node if exists; otherwise, None
        :rtype: Node or None
        """
        return self.find_children_by(matcher)[index]

    def add_child(self, node, index=None):
        """
        Adds a node as the index'th child of this node if it is not already a child of this node.

        :type node: Node
        :type index: int
        :param node: node is added to children
        :param index: add child to this position
        :return: if the node is added successfully, retun True; otherwise, False
        :rtype: bool
        """
        if index is None:
            index = self.index_of(node)
        if self.is_parent_of(node):
            return False
        else:
            if node.has_parent():
                node.parent.remove_child(node)
            node.parent = self
            self.children.insert(index, node)
            self._siblings(self.child(index - 1), node)
            self._siblings(node, self.child(index + 1))
            return True

    def set_child(self, node, index):
        """
        Sets a node as the index'th child of this node if it is not already a child of this node.

        :type node: Node
        :type index: int
        :param node: node is set at the index position
        :param index: set a child to this position
        :return: the previously index'th node if added; otherwise, None
        :rtype: Node or None
        """
        if not self.is_parent_of(node):
            if node.has_parent():
                node.parent.remove_child(node)
            node.parent = self
            old_node = self.children[index]
            self.children[index] = node
            self._siblings(self.child(index - 1), node)
            self._siblings(node, self.child(index + 1))
            old_node.isolate()
            return old_node
        else:
            return None

    def remove_child(self, node):
        """
        Removes a child from this node.

        :type node: Node
        :param node: remove node
        :return: True if the node exist; otherwise, False
        :rtype: bool
        """
        index = self.index_of(node)
        if index is not None:
            self._siblings(self.child(index - 1), self.child(index + 1))
            self.children.remove(node)  #
            node.isolate()  # is isolation necessary ?
            return True
        else:
            return False

    def replace_child_from(self, old_node, new_node):
        """
        Replaces the child from old_node to new_node

        :type old_node: Node
        :type new_node: Node
        :param old_node: old child
        :param new_node: new child
        :return: True if the child is replaced successfully; otherwise, False
        :rtype: bool
        """

        index = self.index_of(old_node)
        if index is not None:
            self.children[index] = new_node
            self._siblings(self.child(index - 1), new_node)
            self._siblings(new_node, self.child(index + 1))
            old_node.isolate()
            return True
        else:
            return False

    def remove(self):
        """
        Removes this node from its parent.
        If this is the only child, removeToken its parent from its grandparent, and applies
        this logic recursively to the ancestors.

        """
        node = self
        while node.has_parent():
            parent = node.parent
            parent.remove_child(node)
            if parent.has_child():
                break
            node = parent

    def has_child(self):
        """
        Whether current node has any child or not

        :return: True if this node has any child; otherwise, False
        :rtype: bool
        """
        return True if self.children else False

    def is_child_of(self, node):
        """
        Whether current node is a child of the node or not

        :param node: Node
        :param node: Node
        :return: True if this node is a child of the specific node; otherwise, False
        :rtype: bool
        """
        return node is not None and self.parent is node

    def contains_child(self, matcher=lambda n: n):
        """
        Whether current node contains a child with the matcher or not

        :type matcher: lambda
        :param matcher: The matcher of condition
        :return: True if this node contains a child of the matcher; otherwise, False
        :rtype: bool
        """
        return True if list(filter(matcher, self.children)) else False

    # Descendants

    def num_of_children(self):
        """

        :return: numbers of children
        :rtype: int
        """
        return len(self.children)

    def find_children_in(self, first_child_index=0, last_child_index=None):
        """
        The sublist begins at the specific position and extends to the end.

        :type first_child_index: int
        :type last_child_index: int
        :param first_child_index: the ID of the first child (inclusive)
        :param last_child_index: the ID of the last child (exclusive)
        :return: an immutable list of sub-children
        :rtype: list
        """
        return self.children[first_child_index:last_child_index]

    def find_children_by(self, matcher=lambda n: n):
        """

        :type matcher: lambda
        :param matcher: the condition
        :return: the list of children matching the specific condition.
        :rtype: list
        """
        return list(filter(matcher, self.children))

    def grand_children(self):
        """

        :return: the list of grand-children
        :rtype: list
        """
        return self.second_order('children')

    def find_first_descendant_by(self, matcher=lambda n: True):
        """

        :type matcher: lambda
        :param matcher: the condition
        :return: the first descendant matching the specific condition
        :rtype: Node or None
        """
        for node in self.children:
            if matcher(node):
                return node

            node = self.find_first_descendant_by(node.children, matcher)
            if node is not None:
                return node
        return None

    def find_first_lowest_chained_descendant_by(self, matcher):
        """

        :type matcher: lambda
        :param matcher: the condition
        :return: the first lowest descendant whose intermediate ancestors to this node all match the specific condition
        :rtype: Node or None
        """
        node = self.find_child_by(matcher)
        descendant = None

        while node is not None:
            descendant = node
            node = node.find_child_by(matcher)

        return descendant

    def descendants(self):
        """

        :return: the list of all descendents (excluding this node).
        :rtype: list
        """
        result = []
        if not self.has_child():
            return [self]
        else:
            for child in self.children:
                result = result + child.descendants
            return result

    def find_descendants_in(self, depth):
        """

        :type depth: int
        :param depth: the level of the descendents to be retrieved (1: children, 2: childreNLPNode + grand-children, etc.).
        :return: the list of descendents (excluding this node).
        :rtype: list
        """
        result = []
        if depth <= 0:
            return result
        else:
            for child in self.children:
                result = result + child.find_descendants_in(depth - 1)
            return result

    def find_single_chained_by(self, matcher=lambda n: True):
        """

        :type matcher: lambda
        :param matcher: the condition
        :return: a node in this node's subtree (including self) matching the specific condition if it is single-chained.
        :rtype: Node or None
        """
        node = self
        while node is not None:
            if matcher(node):
                return node
            if node.num_of_children() is 1:
                node = node.first_child()
            else:
                break
        return None

    def adapt_descendants_from(self, node):
        """

        :type node: Node
        :param node:
        """
        for child in node.children:
            child.parent = self

    def is_descendants_of(self, node):
        """

        :type node: Node
        :param node: the node
        :return: True if the node is a descendant of the specific node
        :rtype: bool
        """
        return self.find_nearest_node_by(
            getter='parent', matcher=lambda n: n is node) is not None

    # Ancestors

    def grand_parent(self):
        """

        :return: the grandparent of this node if exists; otherwise, None
        :rtype: Node or None
        """
        return self.ancestor(height=2)

    def ancestor(self, height):
        """

        :type height: int
        :param height:  height of the ancestor from this node (1: parent, 2: grandparent, etc.).
        :return: the lowest ancestor matching the specific condition
        :rtype: Node
        """
        return self.find_node_in(order=height, getter='parent')

    def find_lowest_ancestor_by(self, matcher=lambda n: True):
        """

        :type matcher: lambda
        :param matcher: the condition
        :return: the lowest ancestor matching the specific condition
        :rtype: Node
        """
        return self.find_nearest_node_by('parent', matcher)

    def find_highest_ancestor_by(self, matcher=lambda n: True):
        """

        :type matcher: lambda
        :param matcher: the condition
        :return: the highest ancestor where all the intermediate ancestors match the specific condition
        :rtype: Node
        """
        node = self.parent
        ancestor = None
        while node is not None:
            if matcher(node):
                ancestor = node
            else:
                break
            node = node.parent
        return ancestor

    def lowest_common_ancestor(self, node):
        """

        :type node: Node
        :param node: the node
        :return: the lowest common ancestor of this node and the specified node
        :rtype: Node or None
        """
        ancestor_set = self.ancestor_set()
        ancestor_set.add(self)

        while node is not None:
            if node in ancestor_set:
                return node
            node = node.parent

        return None

    def ancestor_set(self):
        """

        :return: the setToken of all ancestors of this node
        :rtype: set
        """
        ancestor_set = set()
        node = self.parent
        while node is not Node:
            ancestor_set.add(node)
            node = node.parent
        return ancestor_set

    def is_parent_of(self, node):
        """

        :type node: Node
        :param node: the node
        :return: True if this node is the parent of the specific node; otherwise, False
        :rtype: bool
        """
        return node.is_child_of(self)

    def is_ancestor_of(self, node):
        """

        :type node: Node
        :param node: the node
        :return: True if the node is a descendant of the specific node; otherwise, False
        :rtype: bool
        """
        return node.is_descendants_of(self)

    def has_parent(self, matcher=lambda n: True):
        """

        :param matcher: the condition
        :return: True if this node has a parent; otherwise, False
        :rtype: bool
        """
        return self.parent is not None and matcher(self)

    def has_grand_parent(self):
        """

        :return: True if this node has a grand parent; otherwise, False
        :rtype: bool
        """
        return self.grand_parent() is not None

    # Siblings

    def siblings(self):
        """

        :return: the list of all siblings
        :rtype: list
        """
        return list(
            filter(
                lambda child: child is not self,
                self.parent.children)) if self.has_parent() else []

    def left_nearest_sibling(self, order=0):
        """

        :type order: int
        :param order: displacement (0: left-nearest, 1: 2nd left-nearest, etc.)
        :return: the order'th left-nearest sibling of this node if exists; otherwise, None
        :rtype: Node or None
        """
        return self.find_node_in(
            order + 1, 'left_sibling') if order >= 0 else None

    def find_left_nearest_sibling_by(self, matcher=lambda x: True):
        """

        :type matcher: lambda
        :param matcher: the condition
        :return: the left nearest sibling that matches the specific condition
        :rtype: Node
        """
        return self.find_nearest_node_by('left_sibling', matcher)

    def right_nearest_sibling(self, order=0):
        """

        :type order: int
        :param order: displacement (0: left-nearest, 1: 2nd left-nearest, etc.)
        :return: the order'th right-nearest sibling of this node if exists; otherwise, None
        :rtype: Node or None
        """
        return self.find_node_in(
            order + 1, 'right_sibling') if order >= 0 else None

    def find_right_nearest_sibling_by(self, matcher=lambda x: True):
        """

        :type matcher: lambda
        :param matcher: the condition
        :return: the right nearest sibling that matches the specific condition
        :rtype: Node
        """
        return self.find_nearest_node_by('right_sibling', matcher)

    def has_left_sibling(self, matcher=lambda x: True):
        """

        :type matcher: lambda
        :param matcher: the condition
        :return: Trur if this node has a left sibling; otherwise, False.
        """
        return self.left_sibling is not None or self.find_left_nearest_sibling_by(
            matcher)

    def has_right_sibling(self, matcher=lambda x: True):
        """

        :type matcher: lambda
        :param matcher: the condition
        :return: Trur if this node has a right sibling; otherwise, False.
        """
        return self.right_sibling is not None or self.find_right_nearest_sibling_by(
            matcher)

    def is_sibling_of(self, node):
        """

        :type node: Node
        :param node: the node
        :return: True if this node is a sibling of the specific node.
        :rtype: bool
        """
        return node.is_child_of(self.parent)

    def is_left_sibling_of(self, node):
        """

        :type node: Node
        :param node:
        :return: True if this node is a left sibling of the specific node
        :rtype: bool
        """
        return node is not None and self.parent is node.parent and self.find_nearest_node_by(
            'right_sibling', lambda n: n is node) is not None

    def is_right_sibling_of(self, node):
        """

        :type node: Node
        :param node:
        :return: True if this node is a right sibling of the specific node
        :rtype: bool
        """
        return node.is_left_sibling_of(self)

    # Helpers

    def find_node_in(self, order, getter):
        """

        :type order: int
        :type getter: str
        :param order: 0: self, 1: nearest, 2: second nearest, etc.
        :param getter: takes a node and returns a node.
        :return: the order'th node with respect to the getter.
        :rtype: Node
        """
        node = self
        for _ in range(order):
            if node is None:
                return None
            node = getattr(node, getter)
        return node

    def find_nearest_node_by(self, getter, matcher=lambda n: True):
        """

        :type getter: str
        :type matcher: lambda
        :param getter: takes a node and returns a node.
        :param matcher: takes a node and the supplement, and returns true if its field matches to the specific predicate.
        :return: the first node matching the specific condition
        :rtype: Node or None
        """
        node = getattr(self, getter)

        while node is not None:
            if matcher(node):
                return node
            node = getattr(node, getter)
        return None

    def isolate(self):
        """
        Isolates this node from its parent, children, and siblings
        """
        self._parent = None
        self._left_sibling = None
        self._right_sibling = None

    def second_order(self, getter):
        """

        :type getter: str
        :param getter: takes a node and returns a list of nodes.
        :return: the list of second order elements according to the getter.
        :rtype: list
        """
        return list(itertools.chain.from_iterable(
            list(filter(lambda n: getattr(n, getter) if n is not self else None, getattr(self, getter)))))

    def _siblings(self, left_node, right_node):
        """
        Sets two nodes siblings of each other.

        :type left_node: Node
        :type right_node: Node
        :param left_node:
        :param right_node:
        """
        right_node.left_sibling = left_node
        left_node.right_sibling = right_node
