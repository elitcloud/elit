# from elit.dev.reader import TSVReader
# import os
# import pytest
#
# @pytest.fixture()
# def tsv_file():
#     return os.path.join(os.path.abspath(os.path.dirname(__file__)),
#                         'test_reader/sample.tsv')
#
# @pytest.fixture()
# def reader():
#     return TSVReader(1, 2, 3, 4, 5, 6, 7, 8)
#
# def test_tsv_reader_nodes(tsv_file, reader):
#     reader.open(tsv_file)
#     assert len(reader.next_all) == 2
#     reader.close()
#
# def test_tsv_reader_node(tsv_file, reader):
#     reader.open(tsv_file)
#     nodes = [7, 11]
#     for i, node in enumerate(reader.next_all):
#         assert len(node) == nodes[i]
#     reader.close()
#
# def test_tsv_reader_graph(tsv_file, reader):
#     reader.open(tsv_file)
#     node = reader.next.nodes[1]
#     assert node.token_id == 1
#     assert node.form == 'John'
#     assert node.lemma == 'john'
#     assert node.pos == 'NNP'
#     assert node.nament == 'U-PERSON'
#     assert node.feats == {}
#     reader.close()