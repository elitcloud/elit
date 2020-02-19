# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-10-03 16:27
from elit.component.embedding.fasttext import FastText
from elit.component.token_tagger import CNNTokenTagger
from elit.util.mx import mxnet_prefer_gpu

# https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
tagger = CNNTokenTagger(ctx=mxnet_prefer_gpu(), key='pos', embs=[FastText()])
