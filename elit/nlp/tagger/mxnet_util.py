# -*- coding:utf-8 -*-
# Author：hankcs
# Date: 2018-09-16 16:47
import os

import mxnet as mx


def mxnet_prefer_gpu():
    '''
    If gpu available return gpu, else cpu
    :return:
    '''
    if 'cuda' not in os.environ['PATH']:
        return mx.cpu()
    return mx.gpu(0)


def mxnet_gpus():
    '''
    If gpu available return all gpus, else cpu
    :return:
    '''
    if 'cuda' not in os.environ['PATH']:
        return [mx.cpu()]
    gpus = mx.test_utils.list_gpus()
    return [mx.gpu(i) for i in gpus]