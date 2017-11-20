# ========================================================================
# Copyright 2017 Emory University
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
import mxnet as mx
import time
import sys

__author__ = 'Jinho D. Choi'

vocab_size = 40000
batch_size = 50
emb_dim = 200
document_size = 256

input_x = mx.sym.Variable('x')
input_y = mx.sym.Variable('softmax_label')
emb_layer = mx.sym.Embedding(data=input_x, input_dim=vocab_size, output_dim=emb_dim, name='vocab_emb')
conv_input = mx.sym.Reshape(data=emb_layer, target_shape=(batch_size, 1, document_size, emb_dim))

filter_sizes = [2, 3, 4, 5]
num_filter = 100
pooled_outputs = []

for i, filter_size in enumerate(filter_sizes):
    convi = mx.sym.Convolution(data=conv_input, kernel=(filter_size, emb_dim), num_filter=num_filter)
    relui = mx.sym.Activation(data=convi, act_type='relu')
    pooli = mx.sym.Pooling(data=relui, pool_type='max', kernel=(document_size - filter_size + 1, 1), stride=(1, 1))
    pooled_outputs.append(pooli)

total_filters = num_filter * len(filter_sizes)
concat = mx.sym.Concat(*pooled_outputs, dim=1)
h_pool = mx.sym.Reshape(data=concat, target_shape=(batch_size, total_filters))

dropout = 0.5
h_drop = mx.sym.Dropout(data=h_pool, p=dropout) if dropout > 0 else h_pool

num_labels = 2
cls_weight = mx.sym.Variable('cls_weight')
cls_bias = mx.sym.Variable('cls_bias')
fc = mx.sym.FullyConnected(data=h_drop, weight=cls_weight, bias=cls_bias, num_hidden=num_labels)
sm = mx.sym.SoftmaxOutput(data=fc, label=input_y, name='softmax')
cnn = sm


epochs = 10
learning_rate = 0.01
opt = mx.optimizer.create('rmsprop')
opt.lr = learning_rate
updater = mx.optimizer.get_updater(opt)
logs = sys.stderr

for iter in range(epochs):
    tic = time.time()
    correct = 0
    total = 0
