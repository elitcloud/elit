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
from elit.structure.constituency import *
import unittest
import io


class ConstituencyTest(unittest.TestCase):
    def test(self):
        self.testConstituencyNode()
        self.testConstituencyTree()
        self.testConstituencyReader()

    def testConstituencyNode(self):
        # tags
        tags = 'NP-LOC-PRD-1=2'
        np = ConstituencyNode(tags)
        self.assertEqual(np.syntactic_tag, 'NP')
        self.assertEqual(np.function_tagset, {'LOC', 'PRD'})
        self.assertEqual(np.co_index, 1)
        self.assertEqual(np.gap_index, 2)
        self.assertEqual(np.tags, tags)

        # match
        self.assertTrue(np.match(ctag='NP'))
        self.assertTrue(np.match(crex=re.compile('(NP|NN)')))
        self.assertTrue(np.match(cset={'NP', 'NN'}))
        self.assertTrue(np.match(ftag='LOC'))
        self.assertTrue(np.match(fset={'TMP', 'PRD'}))
        self.assertTrue(np.match(ctag='NP', ftag='LOC'))

        self.assertFalse(np.match(ctag='DT'))
        self.assertFalse(np.match(crex=re.compile('(DT|NN)')))
        self.assertFalse(np.match(cset={'DT', 'NN'}))
        self.assertFalse(np.match(ftag='TMP'))
        self.assertFalse(np.match(fset={'TMP', 'SBJ'}))
        self.assertFalse(np.match(ctag='NP', ftag='TMP'))

        # parent, children, siblings, ancestors
        s = ConstituencyNode('S')
        dt = ConstituencyNode('DT', 'a')
        jj0 = ConstituencyNode('JJ', 'big')
        jj1 = ConstituencyNode('JJ', 'red')
        nn0 = ConstituencyNode('NN', 'car')
        nn1 = ConstituencyNode('NN', 'truck')

        s.add_child(np)
        np.add_child(dt)
        np.add_child(nn0)
        np.add_child(jj0, 1)
        np.add_child(nn1)
        np.add_child(jj1, 2)

        nodes = [dt, jj0, jj1, nn0, nn1]
        for i in range(len(np.children_list)):
            self.assertEqual(np.child(i).parent, np)
            self.assertEqual(np.child(i), nodes[i])
            self.assertEqual(np.child(i).left_sibling, np.child(i-1))
            self.assertEqual(np.child(i).right_sibling, np.child(i+1))

        self.assertEqual(np.first_child(ctag='NN'), nn0)
        self.assertEqual(np.last_child(ctag='NN'), nn1)
        self.assertEqual(nn0.left_nearest_sibling(ctag='JJ'), jj1)
        self.assertEqual(nn1.left_nearest_sibling(ctag='JJ'), jj1)
        self.assertEqual(jj0.right_nearest_sibling(ctag='NN'), nn0)
        self.assertEqual(jj1.right_nearest_sibling(ctag='NN'), nn0)

        self.assertEqual(dt.nearest_ancestor(ctag='NP'), np)
        self.assertEqual(dt.nearest_ancestor(ctag='S'), s)

        # terminals
        self.assertTrue(dt.is_terminal)
        self.assertFalse(np.is_terminal)
        self.assertEqual(s.terminal_list, nodes)
        self.assertEqual(s.first_terminal, dt)

        # empty categories
        ec = ConstituencyNode(PTB.NONE, '*T*-1')
        np.add_child(ec)
        np2 = ConstituencyNode('NP')
        np2.add_child(ec)

        self.assertEqual(ec.co_index, 1)
        self.assertTrue(ec.is_empty_category)
        self.assertTrue(ec.is_empty_category_deep)
        self.assertTrue(np2.is_empty_category_deep)
        self.assertFalse(np2.is_empty_category)
        self.assertFalse(np.is_empty_category_deep)

        self.assertEqual(np2.first_empty_category(), ec)
        self.assertEqual(np2.first_empty_category(re.compile('\*T\*')), ec)
        self.assertEqual(np2.first_empty_category(re.compile('\*E\*')), None)

    def testConstituencyTree(self):
        reader = ConstituencyReader(inputstream=open('resources/sample.parse'))
        tree = reader.next_tree

        for i, node in enumerate(tree.terminal_list):
            self.assertEqual(node.terminal_id, i)
            self.assertEqual(node, tree.terminal(i))

        for i, node in enumerate(tree.token_list):
            self.assertEqual(node.token_id, i)
            self.assertEqual(node, tree.token(i))

    def testConstituencyReader(self):
        reader = ConstituencyReader(inputstream=open('resources/sample.parse'))
        parses = [tree for tree in reader]
        parse1 = '\n'.join(map(str, parses))
        reader.open(io.StringIO(parse1))
        parse2 = '\n'.join([str(tree) for tree in reader])
        self.assertEqual(parse2, parse1)

        # antecedents of empty categories
        tree = parses[2]
        self.assertEqual(tree.terminal(6).antecedent, tree.node(5, 1))
        self.assertEqual(tree.terminal(10).antecedent, tree.node(6, 1))
        self.assertEqual(tree.terminal(11).antecedent, tree.node(10, 1))


if __name__ == '__main__':
    unittest.main()
