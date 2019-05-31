# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-05-30 18:24
import time

from elit.component.dep.parser.dep_parser import DepParser

parser = DepParser()
save_dir = 'data/model/dep/jumbo'
parser.load(save_dir)
time.sleep(60)