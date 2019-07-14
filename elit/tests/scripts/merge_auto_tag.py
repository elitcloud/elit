# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-07-12 16:46

with open('data/dat/test.tsv') as auto, open('data/dat/en-ddr.trn') as gold, open('data/dat/en-ddr.trn.auto.conll',
                                                                                  'w') as out:
    for auto_line, gold_line in zip(auto, gold):
        auto_line: str = auto_line.strip()
        gold_line: str = gold_line.strip()
        if not auto_line:
            assert not gold_line
            out.write('\n')
            continue
        word, gold_tag, pred_tag = auto_line.split()
        cells = gold_line.split()
        assert cells[3] == gold_tag
        cells[3] = pred_tag
        out.write('\t'.join(cells) + '\n')
