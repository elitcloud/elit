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
import re
__author__ = 'Jinho D. Choi'

TOP = 'TOP'      # root
NONE = '-NONE-'  # empty category


# ==================================== ConstituencyNode ====================================
class ConstituencyNode:
    _TAG_DELIM = re.compile('([-=])')

    def __init__(self, tags=None, word_form=None):
        self.word_form = word_form    # String
        self.constituency_tag = None  # String
        self.function_tagset = set()  # Set of Strings
        self.children_list = list()   # List of ConstituencyNode
        self.parent = None            # ConstituencyNode
        self.left_sibling = None      # ConstituencyNode
        self.right_sibling = None     # ConstituencyNode
        self.antecedent = None        # ConstituencyNode
        self.co_index = -1            # Integer
        self.gap_index = -1           # Integer
        self.terminal_id = -1         # Integer
        self.token_id = -1            # Integer

        # init constituency tag, function tags, co-index, gap-index
        self.tags = tags

    def __str__(self):
        return self.to_parse_tree(single_line=True)

    # ==================================== ConstituencyNode: Basic ====================================

    @property
    # returns all tags (e.g., 'NP-LOC-PRD-1=2') : String
    def tags(self):
        ls = [str(self.constituency_tag)]

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
    # tags (e.g., 'NP-LOC-PRD-1=2') : String
    def tags(self, tags):
        if tags is None or tags.startswith('-'):
            self.constituency_tag = tags
            return

        ls = self._TAG_DELIM.split(tags)
        self.constituency_tag = ls[0]

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
    # returns True if this node has a co-index to an empty category.
    def has_co_index(self):
        return self.co_index >= 0

    @property
    # returns True if this node has a co-index to a gapping relation.
    def has_gap_index(self):
        return self.gap_index >= 0

    @property
    # returns True if this node is a terminal : Boolean
    def is_terminal(self):
        return not self.children_list

    @property
    # returns the list of terminals under the subtree of this node : List of ConstituencyNode
    def terminal_list(self):
        return self._traverse(self, lambda node: node.is_terminal, [])

    # returns the index'th child if exists; otherwise, None : ConstituencyNode
    def child(self, index):
        return self.children_list[index] if 0 <= index < len(self.children_list) else None

    # node : ConstituencyNode
    # index : int
    # if index < 0 or index >= len(self.children), the child node is added to the end
    def add_child(self, node, index=-1):
        if not (0 <= index < len(self.children_list)):
            index = len(self.children_list)

        node.parent = self
        self.children_list.insert(index, node)
        self._set_siblings(self.child(index - 1), node)
        self._set_siblings(node, self.child(index + 1))

    # ==================================== ConstituencyNode: Match ====================================

    # returns True if this node matches all constraints in **kwargs; otherwise False
    # kwargs['crex'] - regular expression of constituency tags : Pattern
    # kwargs['ctag'] - constituency tag : String
    # kwargs['cset'] - set of constituency tags : Set of String
    # kwargs['ftag'] - function tag : String
    # kwargs['fset'] - set of function tags : Set of String
    def match(self, **kwargs):
        return ('crex' not in kwargs or kwargs['crex'].match(self.constituency_tag)) and \
               ('ctag' not in kwargs or kwargs['ctag'] == self.constituency_tag) and \
               ('cset' not in kwargs or self.constituency_tag in kwargs['cset']) and \
               ('ftag' not in kwargs or kwargs['ftag'] in self.function_tagset) and \
               ('fset' not in kwargs or kwargs['fset'].intersection(self.function_tagset))

    # returns the first child matching **kwargs if exists; otherwise, None : ConstituencyNode
    # **kwags - see the descriptions in self.match()
    def first_child(self, **kwargs):
        return next((child for child in self.children_list if child.match(**kwargs)), None)

    # returns the last child matching **kwargs if exists; otherwise, None : ConstituencyNode
    # **kwags - see the descriptions in self.match()
    def last_child(self, **kwargs):
        return next((child for child in reversed(self.children_list) if child.match(**kwargs)), None)

    # returns the left nearest sibling matching **kwargs if exists; otherwise, None : ConstituencyNode
    # **kwags - see the descriptions in self.match()
    def left_nearest_sibling(self, **kwargs):
        return self._node_iterator(lambda node: node.left_sibling, **kwargs)

    # returns the right nearest sibling matching **kwargs if exists; otherwise, None : ConstituencyNode
    # **kwags - see the descriptions in self.match()
    def right_nearest_sibling(self, **kwargs):
        return self._node_iterator(lambda node: node.right_sibling, **kwargs)

    # returns the nearest ancestor matching **kwargs : ConstituencyNode
    # **kwags - see the descriptions in self.match()
    def nearest_ancestor(self, **kwargs):
        return self._node_iterator(lambda node: node.parent, **kwargs)

    # ==================================== ConstituencyNode: Empty Category ====================================

    # returns True if this node is an empty category : Boolean
    # recursive - check if this node contains only empty category recursively : Boolean
    def is_empty_category(self, recursive=False):
        return self._is_empty_category(self) if recursive else self.constituency_tag == NONE

    def _is_empty_category(self, node):
        if node.is_terminal:
            return node.is_empty_category()

        if len(node.children_list) > 1:
            return False

        return self._is_empty_category(node.child(0))

    # regex(e.g., '\*ICH\*.*') : Pattern
    # returns the empty category matching the regular expression in the subtree of this node : ConstituencyNode
    def first_empty_category_in_subtree(self, regex):
        return self._traverse(self, lambda node: node.is_empty_category(False) and regex.match(node.word_form))

    # ==================================== ConstituencyNode: String ====================================

    # returns this node in the Penn Treebank word_format : String
    def to_parse_tree(self, **kwargs):
        single_line = 'single_line' in kwargs and kwargs['single_line']
        numbered = 'numbered' in kwargs and kwargs['numbered']

        lst = list()
        self._to_parse_tree(self, lst, '', single_line, numbered)
        return ' '.join(lst) if single_line else '\n'.join(lst)

    def _to_parse_tree(self, curr, lst, tags, single_line, numbered):
        if curr.is_terminal:
            ls = []
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

            self._to_parse_tree(curr.child(0), lst, tags, single_line, numbered)

            for child in curr.children_list[1:]:
                self._to_parse_tree(child, lst, indent, single_line, numbered)

            lst[-1] += ')'

# ==================================== ConstituencyNode: Helpers ====================================

    @staticmethod
    # left, right : ConstituencyNode
    def _set_siblings(left, right):
        if left:
            left.right_sibling = right
        if right:
            right.left_sibling = left

    # returns the collection containing results from traversing
    # node : ConstituencyNode
    # condition : def(ConstituencyNode) -> Boolean
    # lst : List of results
    def _traverse(self, node, condition, lst=None):
        if condition(node):
            if lst is None:
                return node

            lst.append(node)

        for child in node.children_list:
            f = self._traverse(child, condition, lst)
            if lst is None and f:
                return f

        return lst

    # returns the first node matching **kwargs if exists; otherwise, None : ConstituencyNode
    # nodemap - takes a node and returns a specific node : def(ConstitudencyNode) -> ConstitudencyNode
    # **kwags - see the descriptions in self.match()
    def _node_iterator(self, nodemap, **kwargs):
        node = nodemap(self)

        while node:
            if node.match(**kwargs):
                return node
            node = nodemap(node)

        return None

# ==================================== ConstituencyNode ====================================

    # PRE: English
    # returns the complementizer belongs to this node : ConstituencyNode
    def getComplementizer(self):
        if not self.constituency_tag.startswith('WH'):
            return None

        whNode = self
        while True:
            tmp = whNode.getFirstChild(pRex='WH.*')
            if tmp:
                whNode = tmp
            else:
                break

        terminals = whNode.terminal_list()

        for node in terminals:
            if node.isComplementizer():
                return node

        for node in terminals:
            if RE_COMP_FORM.match(node.word_form.lower()):
                return node

        return None

    # PRE: English
    # returns the lowest coindexed wh-node (including self) : ConstituencyNode
    def getCoIndexedWHNode(self):
        curr = self.parent

        while curr:
            if not curr.constituencyTag.startswith('WH'): break
            if curr.co_index != -1: return curr

            curr = curr.parent

        return None

    # PRE: English
    # returns the subject of this node : ConstituencyNode
    def getSubject(self):
        pred = self
        while pred.parent.constituencyTag == PTAG_VP:
            pred = pred.parent

        return pred.getPrevSibling(constituencyTag=PTAG_NP, fTag=FTAG_SBJ)

    # PRE: English
    # PRE: this node is a verb predicate (e.g., 'say')
    # returns the nearest PRN node if this node is a True if this node is in PRN and has an external argument : Boolean
    def getPredPRN(self):
        s = self.getNearestAncestor(pRex='^S.*')

        if s and s.parent.constituencyTag == PTAG_PRN:
            next = self.getNextSibling(pRex='^(S|SBAR)$')
            if next:
                ec = next.__getEmptySentence()
                if ec: return (s.parent, ec)

        return None

    # called by 'getPredPRN()'.
    def __getEmptySentence(self):
        if self.constituency_tag == PTAG_SBAR:
            if len(self.children_list) == 2 and \
                            self.children_list[0].constituencyTag == PTAG_NONE and self.children_list[0].word_form == '0' and \
                    self.children_list[1].isEmptyCategory(recursive=True):
                return self.children_list[1].getIncludedEmptyCategory('^\*(\?|T)\*')
        elif self.constituency_tag == PTAG_S:
            if self.isEmptyCategory(recursive=True):
                return self.children_list[0].getIncludedEmptyCategory('^\*(\?|T)\*')

        return None

    # PRE: English
    # returns the empty category of passive construction : ConstituencyNode
    def getPassiveEmptyCategory(self):
        if self.parent and self.parent.constituencyTag == PTAG_VP and \
                        self.siblingId > 0 and self.parent.children[self.siblingId - 1].constituencyTag.startswith(
            'VB') and \
                        self.constituency_tag == PTAG_NP and not self.function_tagset and \
                self.isEmptyCategory(recursive=True):
            return self.getIncludedEmptyCategory('^(\*|\*-\\d)$')

    def getLowestSingleChild(self):
        child = self.getLowestSingleChildRec(self)

        if child != self:
            return child
        else:
            return None

    def getLowestSingleChildRec(self, node):
        if len(node.children) == 1:
            child = node.children[0]
            if child.children:
                return self.getLowestSingleChildRec(child)
            else:
                return child.parent
        else:
            return node



            # PRE: English
        # returns True if this node is a complementizer : Boolean

    def isComplementizer(self):
        if not self.word_form: return False
        return RE_COMP_POS.match(self.constituency_tag) or (
        self.constituency_tag == PTAG_NONE and self.word_form == '0')


# ==================================== ConstituencyTree ====================================
class ConstituencyTree:
    RE_NORM = re.compile('\\*(ICH|RNR|PPA)\\*')
    DELIM_PLUS = re.compile('\+')

    # root : ConstituencyNode
    def __init__(self, root):
        self.nd_root = root
        self.ls_terminal = list()
        self.dc_token = dict()

    ########################### TBTree:getters ###########################

    # beginId, endId - terminal IDs (both inclusive) : String
    # returns the node whose span is 'beginId - endId' : ConstituencyNode
    def getNodeBySpan(self, beginId, endId):
        bNode = self.ls_terminal[beginId]

        while bNode:
            sId = bNode.getSubTerminalIdSet()
            m = max(sId)
            if m == endId:
                return bNode
            elif m > endId:
                break

            bNode = bNode.parent

        return None

    # EXPERIMENTAL
    # word_forms : String
    # returns the PropBank location covering 'word_forms' : String
    def getPBLoc(self, terminalId, word_forms):
        node = self.ls_terminal[terminalId]
        size = len(word_forms)

        while True:
            s = node.toForms()
            if s == word_forms: return node.getPBLoc()
            if len(s) > size: break
            if node.parent:
                node = node.parent
            else:
                break

        return None

    # terminalId, height : Integer
    # returns the node in 'terminalId:height' : ConstituencyNode
    def getNode(self, terminalId, height=0):
        node = self.ls_terminal[terminalId]
        for i in range(height):
            node = node.parent

        return node

    # tokenId : Integer
    # returns the 'tokenId'th token : ConstituencyNode
    def getToken(self, tokenId):
        return self.dc_token[tokenId]

    # coIndex : Integer
    # returns the antecedent of 'coIndex' : ConstituencyNode
    def getAntecedent(self, coIndex):
        return self.__getAntecedent(coIndex, self.nd_root)

    # called by 'getAntecendet'
    def __getAntecedent(self, coIndex, curr):
        if curr.co_index == coIndex:
            return curr

        for child in curr.children:
            ante = self.__getAntecedent(coIndex, child)
            if ante: return ante

        return None

    # returns a dictionary of co-index and list of corresponding nodes : Dictionary
    def getCoIndexDict(self):
        d = dict()
        self.__getCoIndexDict(self.nd_root, d)

        return d

    # called by 'getCoIndexDict'
    def __getCoIndexDict(self, curr, d):
        if not curr.children: return
        coIndex = curr.co_index

        if coIndex != -1:
            if coIndex in d:
                l = d[coIndex]
            else:
                l = list()
                d[coIndex] = l

            l.append(curr)

        for child in curr.children:
            self.__getCoIndexDict(child, d)

    # terminalId : Integer
    # delim : String
    # returns all previous token word_forms (including self) without space : String
    def getPrevTokenForms(self, terminalId, delim=''):
        node = self.getNode(terminalId)
        l = list()

        for i in range(node.tokenId + 1):
            l.append(self.dc_token[i].word_form)

        return delim.join(l)

    # prevTokenForms - returned by getPrevTokenForms(delim) : String
    # delim : String
    # return the token of 'prevForms' : Token
    def getTokenByPrevForms(self, prevTokenForms, delim=''):
        l = list()

        for i in range(len(self.dc_token)):
            token = self.dc_token[i]
            l.append(token.word_form)
            s = delim.join(l)

            if s == prevTokenForms:
                return token
            elif len(s) > len(prevTokenForms):
                break

        return None

    def getPrevTerminalForms(self, terminalId, delim=''):
        l = list()

        for i in range(terminalId + 1):
            node = self.ls_terminal[i]

            if node.isEmptyCategory():
                l.append('*NULL*')
            else:
                l.append(node.word_form)

        return delim.join(l)

    def getTerminalByPrevForms(self, prevTerminalForms, delim=''):
        l = list()

        for i in range(len(self.ls_terminal)):
            node = self.ls_terminal[i]
            if node.isEmptyCategory():
                l.append('*NULL*')
            else:
                l.append(node.word_form)
            s = delim.join(l)

            if s == prevTerminalForms:
                return self.ls_terminal[i]
            elif len(s) > len(prevTerminalForms):
                break

        return None

    def getTerminals(self):
        return self.ls_terminal

    ########################### TBTree:setters ###########################

    # node : ConstituencyNode
    def addTerminal(self, node):
        self.ls_terminal.append(node)

    # node : ConstituencyNode
    def addToken(self, tokenId, node):
        self.dc_token[tokenId] = node

    # initializes antecedents of all empty categories.
    def setAntecedents(self):
        for node in self.ls_terminal:
            if not node.isEmptyCategory(): continue
            coIndex = node.word_form[node.word_form.rfind('-') + 1:]
            if coIndex.isdigit():
                node.antecedent = self.getAntecedent(int(coIndex))

    # assigns PropBank locations to all nodes.
    def setPBLocs(self):
        for node in self.ls_terminal:
            terminalId = node.terminalId
            height = 0
            node.setPBLoc(terminalId, height)

            while node.parent and not node.parent.pbLoc:
                node = node.parent
                height += 1
                node.setPBLoc(terminalId, height)

    # normalizes all co-indices and gap-indices
    def normalizeIndices(self):
        dIndex = self.getCoIndexDict()
        if not dIndex: return

        dGap = dict()
        keys = dIndex.keys()
        keys.sort()
        coIndex = 1

        for key in keys:
            l = dIndex[key]
            l.reverse()
            isAnteFound = False

            for i, node in enumerate(l):
                if node.isEmptyCategory(True):
                    ec = self.ls_terminal[node.pbLoc[0]]

                    if i == 0 or isAnteFound or self.RE_NORM.match(ec.word_form):
                        node.co_index = -1
                    else:
                        node.co_index = coIndex
                        coIndex += 1

                    if isAnteFound or i < len(l) - 1:
                        ec.word_form += '-' + str(coIndex)
                elif isAnteFound:
                    print('Error: too many antecedents of co-index ' + key)
                    node.co_index = -1
                else:
                    dGap[key] = coIndex
                    node.co_index = coIndex
                    isAnteFound = True

            coIndex += 1

        self.__remapGapIndices(dGap, coIndex)

    # called by 'normalizeIndices'
    def __remapGapIndices(self, dGap, lastIndex):
        self.__remapGapIndicesAux(dGap, [lastIndex], self.nd_root)

    # called by '__remapGapIndices()'.
    def __remapGapIndicesAux(self, dGap, lastIndex, curr):
        gapIndex = curr.gap_index

        if gapIndex in dGap:
            curr.gap_index = dGap[gapIndex]
        elif curr.gap_index != -1:
            dGap[gapIndex] = lastIndex[0]
            curr.gap_index = lastIndex[0]
            lastIndex[0] += 1

        for child in curr.children:
            self.__remapGapIndicesAux(dGap, lastIndex, child)

    # PRE: English
    # initializes antecedents of all complementizers.
    def setWHAntecedents(self):
        self.__setWHAntecedents(self.nd_root)

    # called by 'setWHAntecedents()'.
    def __setWHAntecedents(self, curr):
        if RE_COMP_ANTE.match(curr.constituencyTag):
            comp = curr.getComplementizer()
            sbar = self.__getHighestSBAR(curr)

            if comp and sbar:
                p = sbar.parent
                if not p: return

                if p.constituencyTag == PTAG_NP:
                    ante = sbar.getPrevSibling(constituencyTag=PTAG_NP)
                    if ante: comp.antecedent = ante
                elif p.constituencyTag == PTAG_WHNP:
                    ante = sbar.getPrevSibling(constituencyTag=PTAG_WHNP)
                    if ante: comp.antecedent = ante
                elif p.constituencyTag == PTAG_VP:
                    ante = p.first_child(fTag=FTAG_PRD)
                    if ante and (ante.constituencyTag == PTAG_NP or (
                            curr.constituencyTag == PTAG_WHPP and ante.constituencyTag == PTAG_PP)):
                        comp.antecedent = ante
            # elif FTAG_PRD in p.functionTags:
            #       ante = p.getSubject()
            #       if ante: comp.antecedent = ante

            return

        for child in curr.children:
            self.__setWHAntecedents(child)

    # called by '__setWHAntecedents()'.
    def __getHighestSBAR(self, whNode):
        sbar = whNode
        while sbar.parent.constituencyTag == PTAG_SBAR:
            sbar = sbar.parent

        if sbar.constituencyTag == PTAG_SBAR:
            if sbar.co_index != -1:
                for i in range(whNode.pbLoc[0] - 1, -1, -1):
                    node = self.ls_terminal[i]
                    if node.isEmptyCategory():
                        t = node.word_form.split('-')
                        if len(t) > 1 and t[1].isdigit() and sbar.co_index == int(t[1]):
                            return self.__getHighestSBAR(node)

            return sbar

        return None

    # PRE: English
    # initializes antecedent of all '*' for passive construction.
    def setPassiveAntecedents(self):
        self.__setPassiveAntecedents(self.nd_root)

    # called by 'setPassiveAntecedents().'
    def __setPassiveAntecedents(self, curr):
        ec = curr.getPassiveEmptyCategory()

        if ec and ec.word_form == '*':
            vp = curr.parent
            while vp.parent.constituencyTag == PTAG_VP: vp = vp.parent

            if vp.parent.constituencyTag == PTAG_NP:
                ante = vp.getPrevSibling(constituencyTag=PTAG_NP)
                if ante: ec.antecedent = ante
        else:
            for child in curr.children:
                self.__setPassiveAntecedents(child)

    def transword_formSL(self):
        size = len(self.ls_terminal)
        i = 0

        while i < size:
            curr = self.ls_terminal[i]

            if curr.constituencyTag == 'sL':
                curr.constituencyTag = 'sl'
                p = curr.parent
                l = p.children
                j = l.index(curr)

                if j > 0:
                    s = l[j:]
                    (r, t) = self.includeSR(s)

                    if not r:
                        i = self.ls_terminal.index(s[-1].terminal_list()[-1])

                        if i + 1 < size:
                            next = self.ls_terminal[i + 1]
                            del l[j:]

                            p = next.parent
                            l = p.children
                            j = l.index(next)
                            for node in reversed(s):
                                node.parent = p
                                l.insert(j, node)

            i += 1

    def mergePRN(self):
        for node in self.ls_terminal:
            if node.word_form == '-LRB-':
                p = node.parent
                l = p.children
                j = l.index(node)
                (r, t) = self.includeSR(l[j:])

                gp = p.parent
                gl = gp.children
                gj = gl.index(p)

                if gj > 0 and j == 0 and r and r[-1].word_form == '-RRB-':
                    pr = gl[gj - 1]

                    if p.constituencyTag == 'NP' and pr.constituencyTag == 'NP':
                        for node in l:
                            node.parent = pr
                            pr.add_child(node)

                        gl.remove(p)

    def addPRN(self):
        for node in self.ls_terminal:
            if node.word_form == '-LRB-':
                p = node.parent
                l = p.children_list
                j = l.index(node)

                if j > 0:
                    (r, t) = self.includeSR(l[j:])

                    if r and r[-1].word_form == '-RRB-':
                        prn = ConstituencyNode(p, 'NP-PRN')
                        del l[j:j + len(r)]
                        l.insert(j, prn)

                        for node in r:
                            node.parent = prn
                            prn.add_child(node)

    def markPRN(self):
        self.markPRNAux(self.nd_root)

    def markPRNAux(self, curr):
        i = 0

        if curr.constituencyTag != 'TOP' and 'PRN' not in curr.functionTags:
            terms = curr.terminal_list()

            if terms[0].word_form == '-LRB-':
                if terms[-1].word_form == '-RRB-':
                    curr.functionTags.add('PRN')
                    i = 100000
                elif len(terms) > 2 and self.isEJX(terms[-1]) and terms[-2].word_form == '-RRB-':
                    curr.functionTags.add('PRN')
                    p = curr.parent
                    l = p.children
                    i = 100000
                    j = l.index(curr)

                    ejx = terms[-1]
                    ejx.parent.children.remove(ejx)
                    ejx.parent = p
                    p.children.insert(j + 1, ejx)

        while i < len(curr.children):
            self.markPRNAux(curr.children[i])
            i += 1

    def includeSR(self, s):
        for i, node in enumerate(s):
            if node.constituencyTag == 'sr':
                return (s[:i + 1], s[i + 1:])

        return (None, None)

    def isEJX(self, node):
        for constituencyTag in self.DELIM_PLUS.split(node.constituencyTag):
            c = constituencyTag[0]
            if c != 'e' and c != 'j' and c != 'x':
                return False

        return True

    ########################### TBTree:helpers ###########################

    # returns the number of terminals : Integer
    def countTerminals(self):
        return len(self.ls_terminal)

    # returns the number of tokens : Integer
    def countTokens(self):
        return len(self.dc_token)

    # returns all terminal word_forms : String
    def toForms(self, includeEC=True):
        ls = list()

        for node in self.ls_terminal:
            if includeEC or not node.isEmptyCategory():
                ls.append(node.word_form)

        return ' '.join(ls)

    # returns this tree in the Penn Treebank word_format : String
    def toParseTree(self, numbered=False):
        return self.nd_root.toParseTree(numbered)


##########################################################################################
## BEGIN: TBReader
##########################################################################################

# USAGE 1:
# reader = TBReader()
# reader.open('treeFile')
# tree = reader.getTree()
# for tree in reader: do something
#
# USAGE 2:
# reader = TBReader('byteFile')
# tree = getTree(int(treeId))
class TBReader:
    # tree delimiters: '(', ')', white spaces
    re_delim = re.compile('([()\s])')
    # re_comment = re.compile('<.*>')

    # byteFile - gerneated by 'generate-byte-index.py' : String
    def __init__(self, byteFile=None, **kwargs):
        if byteFile:
            self.d_byte = self.__getByteDict(byteFile)
        else:
            self.d_byte = None

        if 'lang' in kwargs:
            self.setLanguage(kwargs['lang'])
        else:
            self.setLanguage(LANG_EN)
        if 'ante' in kwargs:
            self.b_ante = kwargs['ante']
        else:
            self.b_ante = False

    # byteFile : String
    # returns a dictionary containing byte indices : Dictionary
    def __getByteDict(self, byteFile):
        fin = open(byteFile)
        dByte = dict()

        for line in fin:
            l = line.split()
            treeFile = l[0]
            lByte = list()

            dByte[treeFile] = lByte
            for byte in l[1:]: lByte.append(int(byte))

        return dByte

    # returns iteration.
    def __iter__(self):
        return self

    # returns the next tree : TBTree
    def __next__(self):
        tree = self.getTree()
        if tree:
            return tree
        else:
            raise StopIteration

    ########################## TBReader:getters ##########################

    # treeId : Integer
    # if 'treeId' is None, returns the next tree : TBTree
    # else returns the 'treeId'th tree : TBTree
    def getTree(self, treeId=None):
        del self.ls_tokens[:]

        if treeId:
            self.f_tree.seek(self.l_byte[treeId])
            token = self.__nextToken()  # tok = '('
        else:
            while True:
                token = self.__nextToken()
                if not token:
                    return None  # end of the file
                elif token == '(':
                    break  # loop until '(' is found

        root = ConstituencyNode(None, PTAG_TOP)  # dummy head
        tree = ConstituencyTree(root)
        curr = root
        nBrackets = 1
        terminalId = 0
        tokenId = 0

        while True:
            token = self.__nextToken()
            if nBrackets == 1 and token == PTAG_TOP: continue

            if token == '(':  # token_0 = '(', token_1 = 'tags'
                nBrackets += 1
                tags = self.__nextToken()
                node = ConstituencyNode(curr, tags)
                curr.add_child(node)
                curr = node
            elif token == ')':  # token_0 = ')'
                nBrackets -= 1
                curr = curr.parent
            else:  # token_0 = 'word_form'
                curr.word_form = token
                curr.terminalId = terminalId
                tree.addTerminal(curr)
                terminalId += 1
                if curr.constituencyTag != PTAG_NONE:
                    curr.tokenId = tokenId
                    tree.addToken(tokenId, curr)
                    tokenId += 1

            if nBrackets == 0:  # the end of the current tree
                tree.setPBLocs()

                if self.b_ante:
                    tree.setAntecedents()

                    if self.s_language == LANG_EN:
                        tree.setPassiveAntecedents()
                        tree.setWHAntecedents()

                return tree

        return None

    # called by 'getTree()'.
    def __nextToken(self):
        if not self.ls_tokens:
            line = self.f_tree.readline()  # get tokens from the next line

            if not line:  # end of the file
                self.close()
                return None
            if not line.strip():  # blank line
                return self.__nextToken()

            for tok in self.re_delim.split(line):  # skip white-spaces
                if tok.strip(): self.ls_tokens.append(tok)

        return self.ls_tokens.pop(0)

    ########################## TBReader:setters ##########################

    def setLanguage(self, language):
        self.s_language = language

    ########################## TBReader:helpers ##########################

    # treeFile : String
    # opens 'treeFile'.
    def open(self, treeFile):
        self.f_tree = open(treeFile)
        self.ls_tokens = list()

        if self.d_byte:
            self.l_byte = self.d_byte[treeFile]

    # closes the current Treebank file.
    def close(self):
        self.f_tree.close()

    # PRE: 'byteFile' must have been initialized.
    # treeFile : String
    # returns the number of trees in the current Treebank file : Integer
    def countTrees(self, treeFile=None):
        if treeFile:
            return len(self.d_byte[treeFile])
        else:
            return len(self.l_byte)


##########################################################################################
## END: TBReader
##########################################################################################









# lTag - list of constituencyTags : List of String
def constituencyTagsToRegex(lTag):
    return '^(' + '|'.join(lTag) + ')$'

