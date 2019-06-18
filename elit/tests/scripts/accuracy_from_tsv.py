# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-06-17 15:52

def accuracy_from(tsv):
    total, correct = 0, 0
    with open(tsv) as src:
        for line in src:
            cells = line.strip().split()
            if cells:
                total += 1
                if cells[1] == cells[2]:
                    correct += 1
    return correct / total * 100


if __name__ == '__main__':
    print('%.2f' % (accuracy_from('data/model/pos/jumbo-gluonfasttext-lm/test.tsv')))
