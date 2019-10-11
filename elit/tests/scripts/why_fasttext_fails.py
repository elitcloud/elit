# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-10-11 15:09
import re

import gluonnlp as nlp


def simple_tokenize(source_str, token_delim=' ', seq_delim='\n'):
    return filter(None, re.split(token_delim + '|' + seq_delim, source_str))


text = " hello world \n hello nice world \n hi world \n"

counter = nlp.data.count_tokens(simple_tokenize(text))
vocab = nlp.Vocab(counter)
fs = nlp.embedding.create('fasttext', source='crawl-300d-2M-subword')
vocab.set_embedding(fs)
print(vocab.embedding['hello'])
print(vocab.embedding['beautiful'])
