# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-05-30 18:24
from elit.component.dep.dependency_parser import DependencyParser
from elit.resources.pre_trained_models import DEP_JUMBO

parser = DependencyParser()
parser.load(DEP_JUMBO)
sentence = [('Is', 'VBZ'), ('this', 'DT'), ('the', 'DT'), ('future', 'NN'), ('of', 'IN'), ('chamber', 'NN'),
            ('music', 'NN'), ('?', '.')]
print(parser.parse(sentence))
