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


__author__ = 'T Lee'

#
# class StructureTest(unittest.TestCase):
#     def test_nlpnode_ancestor(self):
#         filename = 'resources/sample/sample.tsv'
#         reader = TSVReader(1, 2, 3, 4, 5, 6, 7, 8)
#         reader.open(filename)
#         graph = reader.next
#         nodes = []
#         for node in graph.nodes:
#             nodes.append(node)
#
#         self.assertEqual(nodes[0].parent, None)
#         self.assertEqual(nodes[0].grandparent, None)
#         self.assertEqual(nodes[3].parent, nodes[0])
#         self.assertEqual(nodes[8].parent, nodes[10])
#         self.assertEqual(nodes[8].grandparent, nodes[3])
#         self.assertTrue(nodes[3].parent_of(nodes[0]))
#         self.assertTrue(nodes[8].parent_of(nodes[10]))
#
#     def test_dependencyLabel(self):
#         filename = 'resources/sample/sample.tsv'
#         reader = TSVReader(1, 2, 3, 4, 5, 6, 7, 8)
#         reader.open(filename)
#         graph = reader.next
#         nodes = []
#         for node in graph.nodes:
#             nodes.append(node)
#
#         self.assertEqual(nodes[5].get_dependency_label(nodes[3]), 'advcl')
#         self.assertEqual(nodes[3].get_dependency_label(nodes[5]), None)
#         self.assertEqual(nodes[14].get_dependency_label(nodes[10]), 'ppmod')
#
#     def test_getChild(self):
#         filename = 'resources/sample/sample.tsv'
#         reader = TSVReader(1, 2, 3, 4, 5, 6, 7, 8)
#         reader.open(filename)
#         graph = reader.next
#         nodes = []
#         for node in graph.nodes:
#             nodes.append(node)
#
#         self.assertEqual(nodes[14].get_leftmost_child(), nodes[11])
#         self.assertEqual(nodes[19].get_leftmost_child(), nodes[16])
#         self.assertEqual(nodes[3].get_leftmost_child(), nodes[1])
#         self.assertEqual(nodes[3].get_rightmost_child(),nodes[20])
#         #self.assertEqual(nodes[19].get_rightmost_child(), nodes[18])
#         #self.assertEqual(nodes[0].get_leftmost_child(), nodes[3])
#
#     def test_nlpnode_set(self):
#         filename = 'resources/sample/sample.tsv'
#         reader = TSVReader(1, 2, 3, 4, 5, 6, 7, 8)
#         reader.open(filename)
#         graph = reader.next
#         container = []
#         for node in graph.nodes:
#             container.append(node)
#
#         tempnode1 = NLPNode("TEMP_1")
#         tempnode2 = NLPNode("TEMP_2")
#         container[0].set_parent(tempnode1, 'dep_1')
#         self.assertEqual(container[0].parent, tempnode1)
#         #container[0].set_parent(tempnode2, 'dep_2')
#         #self.assertNotEquals(container[0].parent, tempnode1)
#
#
# if __name__ == '__main__':
#     unittest.main()
