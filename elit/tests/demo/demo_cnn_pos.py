# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-10-03 16:27
from types import SimpleNamespace

from elit.component.embedding.fasttext import FastText
from elit.component.tagger.corpus import conll_to_documents, label_map_from_conll
from elit.component.token_tagger.cnn import CNNTokenTagger
from elit.util.mx import mxnet_prefer_gpu

label_map = label_map_from_conll('data/ptb/pos/train.tsv')
print(label_map)
tagger = CNNTokenTagger(ctx=mxnet_prefer_gpu(), key='pos',
                        embs=[FastText('https://elit-models.s3-us-west-2.amazonaws.com/fasttext.debug.bin.zip')],
                        input_config=SimpleNamespace(row='what?', col='why?', dropout=0.5),
                        output_config=SimpleNamespace(num_class=12, flatten=True),
                        label_map=label_map
                        )
tagger.train(conll_to_documents('data/conll-03/debug/eng.trn'), conll_to_documents('data/conll-03/debug/eng.dev'),
             'data/model/cnntagger')
