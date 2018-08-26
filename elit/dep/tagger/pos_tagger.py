# -*- coding:utf-8 -*-
# Filename: pos_tagger.py
# Author：hankcs
# Date: 2018-02-23 18:03
from elit.dep.tagger.tagger import Tagger


class POSTagger(Tagger):
    def tag(self, words: list, ret_tulple=True) -> list:
        tags = self._tagger.tag(words)
        if ret_tulple:
            return list(zip(words, tags))
        return tags


if __name__ == '__main__':
    postagger = POSTagger('data/ptb/pos/config.ini')
    postagger.train()
    postagger.load()
    postagger.evaluate()
    print(postagger.tag('I looove languages'.split()))
