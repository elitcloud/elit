# -*- coding:utf-8 -*-
# Author：hankcs
# Date: 2018-09-13 18:34
from elit.nlp.tagger.corpus import Dictionary


def test_dictionary():
    text = 'hello world!'
    d = Dictionary()
    for c in text:
        d.add_item(c)
    assert len(d) == 10