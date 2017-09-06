from elit.reader import TSVReader
import unittest
import os

class ReaderTest(unittest.TestCase):
    def setUp(self):
        self.tsv_file = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                     './../elit/resources/sample/sample.tsv')
        self.reader = TSVReader(1, 2, 3, 4, 5, 6, 7, 8)

    def test_tsv_reader_nodes(self):
        self.reader.open(self.tsv_file)
        self.assertEqual(len(self.reader.next_all), 2)
        self.reader.close()

    def test_tsv_reader_node(self):
        self.reader.open(self.tsv_file)
        nodes = [7, 11]
        for i, node in enumerate(self.reader.next_all):
            with self.subTest(i=i):
                self.assertEqual(len(node), nodes[i])
        self.reader.close()

    def test_tsv_reader_graph(self):
        self.reader.open(self.tsv_file)
        node = self.reader.next.nodes[1]
        self.assertEqual(node.node_id, 1)
        self.assertEqual(node.word, 'John')
        self.assertEqual(node.lemma, 'john')
        self.assertEqual(node.pos, 'NNP')
        self.assertEqual(node.nament, 'U-PERSON')
        self.assertEqual(node.feats, {})
        self.reader.close()


if __name__ == '__main__':
    unittest.main()
