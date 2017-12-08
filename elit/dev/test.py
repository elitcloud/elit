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
import pickle
import random

__author__ = 'Jinho D. Choi'

import mxnet as mx
from mxnet import nd, gluon, autograd

random.seed(1)
x = nd.random_normal(shape=(1000, 2))
y = x * 2

train_data = gluon.data.DataLoader(gluon.data.ArrayDataset(x, y), batch_size=4)
ctx = mx.cpu()
net = gluon.nn.Dense(1, in_units=2)
net.collect_params().initialize(mx.init.Normal(sigma=1.), ctx=ctx)
trainer = gluon.Trainer(net.collect_params(), 'adagrad', {'learning_rate': 0.01})
L = gluon.loss.L2Loss()


for e in range(5):
    cumulative_loss = 0
    for data, label in train_data:
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        with autograd.record():
            output = net(data)
            loss = L(output, label)
        loss.backward()
        trainer.step(data.shape[0])
        cumulative_loss += nd.mean(loss).asscalar()
    print("Epoch %s, loss: %s" % (e, cumulative_loss / len(x)))

# pickle.dump(trainer, open('task.txt', 'wb'))
# trainer = pickle.load(open('task.txt', 'rb'))

for e in range(5):
    cumulative_loss = 0
    for data, label in train_data:
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        with autograd.record():
            output = net(data)
            loss = L(output, label)
        loss.backward()
        trainer.step(data.shape[0])
        cumulative_loss += nd.mean(loss).asscalar()
    print("Epoch %s, loss: %s" % (e, cumulative_loss / len(x)))


