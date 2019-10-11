# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-07-12 18:31
from elit.component.dep.parser.dep_parser import DepParser

parser = DepParser()
save_dir = 'data/model/dep/dep_biaffine_en_mixed_mst'
parser.train(train_file='data/dat/en-ddr.trn.auto.conll',
             dev_file='data/dat/en-ddr.dev.conllx',
             test_file='data/dat/en-ddr.tst.conllx', save_dir=save_dir,
             pretrained_embeddings=('fasttext', 'crawl-300d-2M-subword'), word_dims=300)
parser.load(save_dir)
parser.evaluate(test_file='data/dat/en-ddr.tst.auto.conllx', save_dir=save_dir)
sentence = [('Is', 'VBZ'), ('this', 'DT'), ('the', 'DT'), ('future', 'NN'), ('of', 'IN'), ('chamber', 'NN'),
            ('music', 'NN'), ('?', '.')]
print(parser.parse(sentence))
