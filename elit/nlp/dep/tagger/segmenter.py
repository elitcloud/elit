# -*- coding:utf-8 -*-
# Filename: segmenter.py
# Author：hankcs
# Date: 2018-02-23 11:35
from elit.dep.common.utils import bmes_to_words, WSEvaluator
from elit.dep.tagger.tagger import Tagger


class Segmenter(Tagger):
    def __init__(self, config_file_path) -> None:
        super().__init__(config_file_path)
        self._evaluator = WSEvaluator

    def segment(self, sentence):
        chars = [c for c in sentence]
        tags = self._tagger.tag(chars)
        return bmes_to_words(chars, tags)


def main():
    segmenter = Segmenter('iparser/static/ctb/seg/config.ini')
    segmenter.train()
    segmenter.load()
    print(segmenter.segment('商品和服务'))


if __name__ == '__main__':
    main()
