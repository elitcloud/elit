from elit.reader import TSVReader
import unittest
import os


def check_node(node_num, graph):
    assert len(graph) == node_num


class ReaderTest(unittest.TestCase):
    def setUp(self):
        self.tsv_file = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                     './../resources/sample/sample.tsv')
        self.reader = TSVReader(1, 2, 3, 4, 5, 6, 7, 8)

    def test_tsv_reader_nodes(self):
        self.reader.open(self.tsv_file)
        assert len(self.reader.next_all) == 2
        self.reader.close()

    def test_tsv_reader_node(self):
        self.reader.open(self.tsv_file)
        nodes = [7, 11]
        for i, node in enumerate(self.reader.next_all):
            yield check_node(nodes[i + 1], node)
        self.reader.close()

    def test_tsv_reader_graph(self):
        self.reader.open(self.tsv_file)
        node = self.reader.next.nodes[1]
        assert node.node_id == 1
        assert node.word == 'John'
        assert node.lemma == 'john'
        assert node.pos == 'NNP'
        assert node.nament == 'U-PERSON'
        assert node.feats == {}
        self.reader.close()


if __name__ == '__main__':
    unittest.main()
