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
import unittest
from model import LabelMap

__author__ = 'Jinho D. Choi'

class LabelMapTest(unittest.TestCase):
	#checking str add to LabelMap
	def test_str_str(self):
		the_map = LabelMap()
		self.assertEqual(the_map.__str__(), '[]')
		the_map.add("hello")
		self.assertEqual(the_map.__str__(), "[\'hello\']")

	#checking integer add to LabelMap
	def test_str_int(self):
		the_map = LabelMap()
		the_map.add(1)
		self.assertEqual(the_map.__str__(), '[\'1\']')

	def test_len(self):
		the_map = LabelMap()
		self.assertEqual(the_map.__len__(), 0)
		for item in range(10):
			the_map.add(item)
		self.assertEqual(the_map.__len__(), 10)

	def test_index(self):
		the_map = LabelMap()
		for item in range(10):
			the_map.add(item)
		for item in range(10):
			self.assertEqual(the_map.index(item), int(item))

	def test_add(self):
		the_map = LabelMap()
		self.assertEqual(the_map.add('a'), 0)
		self.assertEqual(the_map.add('b'), 1)
		self.assertEqual(the_map.add('a'), 0)
		self.assertEqual(the_map.add('b'), 1)
		self.assertEqual(the_map.add(1), 2)


if __name__ == '__main__':
	unittest.main()
