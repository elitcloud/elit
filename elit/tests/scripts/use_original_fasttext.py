# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-10-11 15:23
import fasttext

from elit.util.io import fetch_resource

url = 'https://elit-models.s3-us-west-2.amazonaws.com/cc.en.300.bin.zip'
filepath = fetch_resource(url)
fs = fasttext.load_model(filepath)
print(fs[['fakeword', 'fakeword2']])