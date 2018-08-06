# ========================================================================
# Copyright 2018 Emory University
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
from types import SimpleNamespace

import mxnet as mx
import numpy as np
from mxnet import gluon, nd, autograd

from elit.model import FFNNModel

__author__ = 'Jinho D. Choi'


def demo():
    ngrams = range(2, 5)
    train_size = 10
    ctx = mx.cpu(0)
    batch_size = 3

    input_config = SimpleNamespace(row=5, col=20, dropout=0.0)
    output_config = SimpleNamespace(dim=3)
    conv2d_config = None  # tuple(SimpleNamespace(ngram=i, filters=4, activation='relu', pool='max', dropout=0.0) for i in ngrams)
    hidden_config = None  # tuple(SimpleNamespace(dim=10, activation='relu', dropout=0.0) for i in range(2))

    xs = nd.array([np.random.rand(input_config.row, input_config.col) for i in range(train_size)])
    ys = nd.array(np.random.randint(output_config.dim, size=train_size))

    initializer = mx.init.Xavier(rnd_type='gaussian', magnitude=2.24)
    model = FFNNModel(input_config, output_config, conv2d_config, hidden_config)
    model.collect_params().initialize(mx.init.Xavier(rnd_type='gaussian', magnitude=2.24), ctx=ctx)

    # train
    batches = gluon.data.DataLoader(gluon.data.ArrayDataset(xs, ys), batch_size=batch_size)
    trainer = gluon.Trainer(model.collect_params(), 'sgd', {'learning_rate': 0.01, 'wd': 0.0})
    loss_func = mx.gluon.loss.SoftmaxCrossEntropyLoss()
    correct = 0

    for x, y in batches:
        x = x.as_in_context(ctx)
        y = y.as_in_context(ctx)

        with autograd.record():
            output = model(x)
            loss = loss_func(output, y)
            loss.backward()

        trainer.step(x.shape[0])
        correct += int(sum(mx.ndarray.argmax(output, axis=0) == y).asscalar())

    print(correct, len(ys))

    # evaluate
    # batches = gluon.data.DataLoader(gluon.data.ArrayDataset(xs, ys), batch_size=batch_size)
    # outputs = []
    #
    # for x, _ in batches:
    #     x = x.as_in_context(ctx)
    #     outputs.append(model(x))
    #
    # with open('/Users/jdchoi/Downloads/tmp.pkl', 'wb') as fout:
    #     pickle.dump(model, fout)
    #
    # with open('/Users/jdchoi/Downloads/tmp.pkl', 'rb') as fin:
    #     model = pickle.load(fin)
    #
    # batches = gluon.data.DataLoader(gluon.data.ArrayDataset(xs, ys), batch_size=batch_size)
    # outputs2 = []
    # print(str(model))
    #
    # for x, _ in batches:
    #     x = x.as_in_context(ctx)
    #     outputs2.append(model(x))
    #
    # c = 0
    # for i in range(len(outputs)):
    #     c += (sum(outputs[i] - outputs2[i]))
    #     print(outputs[i] - outputs2[i])
    # print(c)


if __name__ == '__main__':
    demo()
