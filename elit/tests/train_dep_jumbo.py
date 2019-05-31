# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-05-30 21:54
from elit.component.dep.parser.dep_parser import DepParser

parser = DepParser()
save_dir = 'data/model/dep/jumbo2'
parser.train(train_file='data/dat/en-ddr.trn.conllx',
             dev_file='data/dat/en-ddr.dev.conllx',
             test_file='data/dat/en-ddr.tst.conllx', save_dir=save_dir,
             pretrained_embeddings=('fasttext', 'crawl-300d-2M-subword'), word_dims=300)
parser.load(save_dir)
# parser.evaluate(test_file='tests/data/biaffine/ptb/test-debug.conllx', save_dir='tests/data/biaffine/model',
#                 num_buckets_test=4)
parser.evaluate(test_file='data/dat/en-ddr.tst.conllx', save_dir=save_dir)
sentence = [('Is', 'VBZ'), ('this', 'DT'), ('the', 'DT'), ('future', 'NN'), ('of', 'IN'), ('chamber', 'NN'),
            ('music', 'NN'), ('?', '.')]
print(parser.parse(sentence))