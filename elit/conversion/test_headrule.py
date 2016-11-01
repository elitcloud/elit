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
import unittest
import os
from elit.conversion.headrule import *
from elit.structure.constituency import *

__author__ = 'Jinho D. Choi'


class HeadruleTest(unittest.TestCase):
    def test(self):
        self.testHeadGroup()
        self.testHeadRule()
        self.testHeadDict()

    def testHeadGroup(self):
        tagset = HeadGroup('NN.*|-PRD|-TMP|VB')

        prd = ConstituencyNode('NP-PRD')
        tmp = ConstituencyNode('S-TMP')
        nns = ConstituencyNode('NNS')
        nnp = ConstituencyNode('NNP')
        vb = ConstituencyNode('VB')
        np = ConstituencyNode('NP')
        vbz = ConstituencyNode('VBZ')

        self.assertTrue(tagset.match(prd))
        self.assertTrue(tagset.match(tmp))
        self.assertTrue(tagset.match(nns))
        self.assertTrue(tagset.match(nnp))
        self.assertTrue(tagset.match(vb))
        self.assertFalse(tagset.match(np))
        self.assertFalse(tagset.match(vbz))

    def testHeadRule(self):
        rule = HeadRule(Direction('r'), 'NN.*|-TMP;VB.*|-PRD')

        nns = ConstituencyNode('NNS')
        tmp = ConstituencyNode('S-TMP')
        vbz = ConstituencyNode('VBZ')
        prd = ConstituencyNode('NP-PRD')

        self.assertTrue(rule[0].match(nns))
        self.assertTrue(rule[0].match(tmp))
        self.assertTrue(rule[1].match(vbz))
        self.assertTrue(rule[1].match(prd))

        self.assertFalse(rule[1].match(nns))
        self.assertFalse(rule[1].match(tmp))
        self.assertFalse(rule[0].match(vbz))
        self.assertFalse(rule[0].match(prd))

    def testHeadDict(self):
        print(os.path.abspath('.'))
        d = read_head_dict(open('resources/en-headrules.txt'))
        self.assertTrue(d['NX'][2].match(ConstituencyNode('NP')))
        self.assertFalse(d['NX'][2].match(ConstituencyNode('VP')))


if __name__ == '__main__':
    unittest.main()
