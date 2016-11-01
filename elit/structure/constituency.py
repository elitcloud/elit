# ========================================================================
# Copyright 2016 Emory University
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
from typing import List
from typing import Pattern

from elit.utils.language import *
from elit.structure.ptb import *

__author__ = 'Jinho D. Choi'


# ==================================== ConstituencyNode ====================================

class ConstituencyNode:
    _DELIM_TAGS = re.compile('([-=])')

    def __init__(self, tags: str, word_form: str=None):
        # fields
        self._word_form = None         # str
        self.syntactic_tag = None      # str
        self.function_tagset = set()   # Set[str]
        self.token_id = -1             # int
        self.terminal_id = -1          # int
        self.co_index = -1             # int
        self.gap_index = -1            # int

        # structure
        self.children_list = list()    # list[ConstituencyNode]
        self.parent = None             # ConstituencyNode
        self.left_sibling = None       # ConstituencyNode
        self.right_sibling = None      # ConstituencyNode
        self.antecedent = None         # ConstituencyNode

        # initialize constituency tag, function tags, co-index, gap-index
        self.tags = tags
        self.word_form = word_form

    def __str__(self):
        return self.str_parse_tree(single_line=True)

    # ==================================== ConstituencyNode: fields ====================================

    @property
    def tags(self) -> str:
        """
        :return: all tags (e.g., 'NP-LOC-PRD-1=2').
        """
        ls = [str(self.syntactic_tag)]

        for tag in sorted(list(self.function_tagset)):
            ls.append('-')
            ls.append(tag)

        if self.has_co_index:
            ls.append('-')
            ls.append(str(self.co_index))

        if self.has_gap_index:
            ls.append('=')
            ls.append(str(self.gap_index))

        return ''.join(ls)

    @tags.setter
    def tags(self, tags: str):
        """
        :param tags: all tags (e.g., 'NP-LOC-PRD-1=2').
        """
        if tags is None or tags.startswith('-'):
            self.syntactic_tag = tags
            return

        ls = self._DELIM_TAGS.split(tags)
        self.syntactic_tag = ls[0]

        for i in range(2, len(ls), 2):
            t = ls[i - 1]
            v = ls[i]

            if v.isdigit():
                if t == '-':
                    self.co_index = int(v)
                else:
                    self.gap_index = int(v)
            else:
                self.function_tagset.add(v)

    @property
    def word_form(self) -> str:
        return self._word_form

    @word_form.setter
    def word_form(self, form: str):
        self._word_form = form

        # init co-index of an empty category if applicable
        if self.is_empty_category and form:
            index = form[form.rfind('-') + 1:]
            if index.isdigit():
                self.co_index = int(index)

    @property
    def terminal_list(self) -> List['ConstituencyNode']:
        """
        :return: the list of terminals under the subtree of this node.
        """
        return _traverse(self, lambda node: node.is_terminal, list())

    @property
    def first_terminal(self) -> 'ConstituencyNode':
        """
        :return: the first terminal under the subtree of this node if exists; otherwise, None.
        """
        return _traverse(self, lambda node: node.is_terminal)

    @property
    def is_terminal(self) -> bool:
        """
        :return: True if this node is a terminal : Boolean.
        """
        return not self.children_list

    @property
    def is_empty_category(self) -> bool:
        """
        :return: True if this is an empty category.
        """
        return self.syntactic_tag == PTB.NONE

    @property
    def is_empty_category_deep(self) -> bool:
        """
        :return: True if this node is a phrase or a terminal containing only an empty category as a terminal.
        """
        def aux(node):
            if node.is_terminal:
                return node.is_empty_category
            if len(node.children_list) > 1:
                return False
            return aux(node.child(0))

        return aux(self)

    @property
    def has_co_index(self) -> bool:
        """
        :return: True if this node has a co-index to an empty category.
        """
        return self.co_index >= 0

    @property
    def has_gap_index(self) -> bool:
        """
        :return: True if this node has a co-index to a gapping relation.
        """
        return self.gap_index >= 0

    def first_empty_category(self, regex: Pattern=None) -> 'ConstituencyNode':
        """
        :param regex: regular expression (e.g., '\*ICH\*')
        :return: the first empty category matching the regular expression in the subtree of this node
        """
        return _traverse(self, lambda node: node.is_empty_category and (regex is None or regex.match(node.word_form)))

    def child(self, index: int) -> 'ConstituencyNode':
        """
        :return: the index'th child of this node if exists; otehrwise, None.
        """
        return self.children_list[index] if 0 <= index < len(self.children_list) else None

    def add_child(self, node: 'ConstituencyNode', index: int or None=None):
        """
        adds a child to the index'th position of this node. If index is not provided, the child is added to the end.
        """
        if index is None:
            index = len(self.children_list)

        node.parent = self
        self.children_list.insert(index, node)
        _set_siblings(self.child(index - 1), node)
        _set_siblings(node, self.child(index + 1))

    def match(self, **kwargs) -> bool:
        """
        :param kwargs:
         - crex: regular expression of syntactic tags : Pattern
         - ctag: syntactic tag : str
         - cset: set of syntactic tags : set of str
         - ftag: function tag : str
         - fset: set of function tags : set of str
        :return: True if this node matches all constraints in kwargs; otherwise False.
        """
        return ('crex' not in kwargs or kwargs['crex'].match(self.syntactic_tag)) and \
               ('ctag' not in kwargs or kwargs['ctag'] == self.syntactic_tag) and \
               ('cset' not in kwargs or self.syntactic_tag in kwargs['cset']) and \
               ('ftag' not in kwargs or kwargs['ftag'] in self.function_tagset) and \
               ('fset' not in kwargs or kwargs['fset'].intersection(self.function_tagset))

    def first_child(self, **kwargs) -> 'ConstituencyNode':
        """
        :param kwargs: see the descriptions in self.match().
        :return: the last child matching kwargs.
        """
        return next((child for child in self.children_list if child.match(**kwargs)), None)

    def last_child(self, **kwargs) -> 'ConstituencyNode':
        """
        :param kwargs: see the descriptions in self.match().
        :return: the last child matching kwargs.
        """
        return next((child for child in reversed(self.children_list) if child.match(**kwargs)), None)

    def left_nearest_sibling(self, **kwargs) -> 'ConstituencyNode':
        """
        :param kwargs: see the descriptions in self.match().
        :return: the left-nearest sibling matching kwargs.
        """
        return _node_iterator(self, lambda node: node.left_sibling, **kwargs)

    def right_nearest_sibling(self, **kwargs) -> 'ConstituencyNode':
        """
        :param kwargs: see the descriptions in self.match().
        :return: the right-nearest sibling matching kwargs.
        """
        return _node_iterator(self, lambda node: node.right_sibling, **kwargs)

    def nearest_ancestor(self, **kwargs) -> 'ConstituencyNode':
        """
        :param kwargs: see the descriptions in self.match().
        :return: the nearest ancestor matching kwargs.
        """
        return _node_iterator(self, lambda node: node.parent, **kwargs)

    def str_parse_tree(self, **kwargs) -> str:
        """
        :param kwargs:
         - pre_tags: pre-defined tags (e.g., TOP) : str
         - single_line: format in a single line : bool
         - numbered: include line numbers : bool
        :return: this node in the Penn Treebank word_format.
        """
        def aux(curr, lst, tags, single_line, numbered):
            if curr.is_terminal:
                ls = list()
                if not single_line and numbered:
                    ls.append('%4d: ' % (len(lst)))

                ls.append(tags)
                ls.append('(')
                ls.append(curr.tags)
                ls.append(' ')
                ls.append(curr.word_form)
                ls.append(')')

                lst.append(''.join(ls))
            else:
                tags = ''.join([tags, '(', curr.tags, ' '])
                indent = ' ' * len(tags) if not single_line else ''

                aux(curr.child(0), lst, tags, single_line, numbered)

                for child in curr.children_list[1:]:
                    aux(child, lst, indent, single_line, numbered)

                lst[-1] += ')'
            return lst

        pre_tags = kwargs.get('pre_tags', '')
        single = kwargs.get('single_line', False)
        number = kwargs.get('numbered', False)
        l = aux(self, list(), pre_tags, single, number)
        return ' '.join(l) if single else '\n'.join(l)


# ==================================== ConstituencyTree ====================================

class ConstituencyTree:
    def __init__(self, root: ConstituencyNode):
        self.root = root              # ConstituencyNode
        self.terminal_list = list()   # List[ConstituencyNode]
        self.token_list = list()      # List[ConstituencyNode]

    def __str__(self):
        return self.str_parse_tree(single_line=True)

    # ==================================== ConstituencyTree: Basic ====================================

    def terminal(self, index: int) -> 'ConstituencyNode':
        """
        :return: the index'th terminal.
        """
        return self.terminal_list[index]

    def token(self, index: int) -> 'ConstituencyNode':
        """
        :return: the index'th token.
        """
        return self.token_list[index]

    def node(self, terminal_id: int, height: int=0) -> 'ConstituencyNode':
        """
        :return: the height's ancestor from the terminal_id'th terminal.
        """
        n = self.terminal(terminal_id)
        for _ in range(height):
            n = n.parent
        return n

    def add_terminal(self, node: 'ConstituencyNode'):
        """
        adds a node to self.terminal_list and self.token_list if it is an empty category.
        """
        node.terminal_id = len(self.terminal_list)
        self.terminal_list.append(node)

        if not node.is_empty_category:
            node.token_id = len(self.token_list)
            self.token_list.append(node)

    def str_parse_tree(self, **kwargs) -> str:
        """
        :param kwargs: see the descriptions in ConstituencyNode.str_parse_tree.
        :return: this tree in the Penn Treebank word_format.
        """
        return self.root.str_parse_tree(**kwargs)

    def str_word_forms(self, delim=' ', empty_category=False) -> str:
        l = self.terminal_list if empty_category else self.token_list
        return delim.join([node.word_form for node in l])


# ==================================== ConstituencyReader ====================================

class ConstituencyReader:
    _DELIM = re.compile('([()\s])')

    def __init__(self, **kwargs):
        """
        :param kwargs:
         - filename: name of the file containing constituency trees : str
         - language: language (e.g., language.LANG_EN) : str
        """
        self.language = Language(kwargs.get('language')) if 'language' in kwargs else Language.english
        self.fin = kwargs['inputstream'] if 'inputstream' in kwargs else None
        self.tokens = None

    def __next__(self):
        """
        :return: the next tree if exists; otherwise, raise StopIteration.
        """
        tree = self.next_tree
        if tree:
            return tree
        else:
            raise StopIteration

    def __iter__(self):
        return self

    def open(self, inputstream):
        """
        :param inputstream: containing constituency trees.
        """
        if self.fin:
            self.close()
        self.fin = inputstream

    def close(self):
        self.fin.close()

    @property
    def next_tree(self):
        """
        :return: the next tree if exists; otherwise, None.
        """
        while True:
            token = self._next_token()
            if not token:
                return None  # end of the file
            elif token == '(':
                break  # loop until '(' is found

        root = ConstituencyNode(PTB.TOP)  # dummy head
        tree = ConstituencyTree(root)
        brackets = 1
        curr = root

        while True:
            token = self._next_token()
            if brackets == 1 and token == PTB.TOP:
                continue

            if token == '(':  # tokens[0] = '(', tokens[1] = 'tags'
                brackets += 1
                tags = self._next_token()
                node = ConstituencyNode(tags)
                curr.add_child(node)
                curr = node
            elif token == ')':  # tokens[0] = ')'
                brackets -= 1
                curr = curr.parent
            else:  # tokens[0] = 'word_form'
                curr.word_form = token
                tree.add_terminal(curr)

            if brackets == 0:  # the end of the current tree
                _init_antecedents(tree)

                '''
                if self.language == LANG_EN:
                    tree.setPassiveAntecedents()
                    tree.setWHAntecedents()
                '''

                return tree

        return None

    def _next_token(self):
        if not self.tokens:
            line = self.fin.readline()  # get tokens from the next line

            if not line:  # end of the file
                return None
            if not line.strip():  # blank line
                return self._next_token()

            self.tokens = [t for t in self._DELIM.split(line) if t.strip()]

        return self.tokens.pop(0)


# ==================================== Utilities ====================================

def _traverse(node: ConstituencyNode, condition, lst: List[ConstituencyNode]=None) -> ConstituencyNode or List[ConstituencyNode]:
    """
    :param condition: def(ConstituencyNode) -> bool.
    :return: the list containing results from traversing if lst is None; otherwise, the first resulting node.
    """
    if condition(node):
        if lst is None:
            return node
        lst.append(node)

    for child in node.children_list:
        f = _traverse(child, condition, lst)
        if lst is None and f:
            return f

    return lst


def _node_iterator(node: ConstituencyNode, nodemap, **kwargs):
    """
    :param nodemap: def(ConstitudencyNode) -> ConstitudencyNode.
    :param kwargs: see the descriptions in ConstituencyNode.match().
    :return: the first node matching kwargs if exists; otherwise, None.
    """
    node = nodemap(node)

    while node:
        if node.match(**kwargs):
            return node
        node = nodemap(node)

    return None


def _set_siblings(left: ConstituencyNode, right: ConstituencyNode):
    if left:
        left.right_sibling = right
    if right:
        right.left_sibling = left


def _init_antecedents(tree: ConstituencyTree):
    """
    initializes antecedents of all empty categories.
    """
    def aux(node: ConstituencyNode, ante_dict):
        if not node.is_empty_category and node.has_co_index:
            ante_dict[node.co_index] = node
        for child in node.children_list:
            aux(child, ante_dict)

        return ante_dict

    d = aux(tree.root, dict())
    for n in (node for node in tree.terminal_list if node.is_empty_category and node.has_co_index):
        n.antecedent = d.get(n.co_index, None)


# normalizes all co-indices and gap-indices
def normalize_indices(self):
    index_dict = self._co_index_dict()
    if not index_dict:
        return
    gap_dict = dict()
    co_index = 1

    for (key, l) in sorted(index_dict.items()):
        l = index_dict[key]
        ante_found = False

        for i, node in enumerate(reversed(l)):
            ec = node.first_empty_category()
            if ec:
                if i == 0 or ante_found or self.RE_NORM.match(ec.word_form):
                    node.co_index = -1
                else:
                    node.co_index = co_index
                    co_index += 1

                if ante_found or i < len(l) - 1:
                    ec.word_form += '-' + str(co_index)
            elif ante_found:
                print('Error: too many antecedents of co-index ' + key)
                node.co_index = -1
            else:
                gap_dict[key] = co_index
                node.co_index = co_index
                ante_found = True

        co_index += 1

    self._remap_gap_indices(gap_dict, co_index)


# returns a dictionary of (co-index, list of corresponding nodes) : Dictionary
def _co_index_dict(self):
    def aux(curr, d):
        if curr.has_co_index and not curr.is_terminal:
            d.setdefault(curr.co_index, list()).append(curr)

        for child in curr.children_list:
            aux(child, d)

        return d

    return aux(self.root, dict())


def _remap_gap_indices(self, gap_dict, last_index):
    def aux(d, index, curr):
        if curr.has_gap_index:
            curr.gap_index = d.setdefault(curr.gap_index, index[0])
            index[0] += 1

        for child in curr.children:
            aux(d, index, child)

    aux(gap_dict, [last_index], self.root)
