# ========================================================================
# Copyright 2018 ELIT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========================================================================
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
    gpu = int(os.environ.get('MXNET_GPU', default=0))
    if gpu == -1:
        return mx.cpu()
    return mx.gpu(gpu)


def mxnet_gpus():
    '''
    If gpu available return all gpus, else cpu
    :return:
    '''
    if 'cuda' not in os.environ['PATH']:
        return [mx.cpu()]
    gpus = mx.test_utils.list_gpus()
    return [mx.gpu(i) for i in gpus]