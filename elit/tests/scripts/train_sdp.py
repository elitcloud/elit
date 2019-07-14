# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-07-12 18:42
from elit.component.sdp.biaffine_sdp import BiaffineSDPParser

parser = BiaffineSDPParser()
save_dir = 'data/model/sdp/sdp_biaffine_en_mixed'
parser.train(train_file='data/dat/en-ddr.trn.auto.conll',
             dev_file='data/dat/en-ddr.dev',
             save_dir=save_dir,
             pretrained_embeddings_file=('fasttext', 'crawl-300d-2M-subword'), word_dims=300)
parser.load(save_dir)
parser.evaluate(test_file='data/dat/en-ddr.tst', save_dir=save_dir)
