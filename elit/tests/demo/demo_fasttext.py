# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-10-11 14:41
from elit.component.embedding.fasttext import FastText

fs = FastText('https://elit-models.s3-us-west-2.amazonaws.com/cc.en.300.bin.zip')
tokens = ['A', 'Lorillard', 'spokewoman', 'said', ',', '``', 'This', 'is', 'an', 'old', 'story', '.']
print(fs.emb_list(tokens))
