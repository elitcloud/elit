# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-05-30 18:24
from elit.component.dep.dependency_parser import DependencyParser

parser = DependencyParser()
save_dir = 'data/model/dep/jumbo'
parser.load(save_dir)
sentence = [('Is', 'VBZ'), ('this', 'DT'), ('the', 'DT'), ('future', 'NN'), ('of', 'IN'), ('chamber', 'NN'),
            ('music', 'NN'), ('?', '.')]
print(parser.parse(sentence))
