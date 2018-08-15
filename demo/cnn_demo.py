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
from mxnet.ndarray import NDArray

from elit.model import FFNNModel

__author__ = 'Jinho D. Choi'


class CNN(gluon.Block):
    """
    :class:`FFNNModel` implements a Feed-Forward Neural Network (FFNN) consisting of
    an input layer, n-gram convolution layers (optional), hidden layers (optional), and an output layer.
    """

    def __init__(self, input_config: SimpleNamespace, output_config: SimpleNamespace, conv2d_config, **kwargs):
        super().__init__(**kwargs)

        self.conv2d = [mx.gluon.nn.Conv2D(channels=c.filters, kernel_size=(c.ngram, input_config.col), strides=(1, input_config.col), activation=c.activation)
                       for c in conv2d_config]

        with self.name_scope():
            for i, c in enumerate(self.conv2d, 0):
                setattr(self, 'conv_' + str(i), c)

            self.output = mx.gluon.nn.Dense(output_config.dim)

    def forward(self, x: NDArray) -> NDArray:
        """
        :param x: the 3D input matrix whose dimensions represent (batch size, feature size, embedding size).
        :return: the output.
        """
        print(x.shape)
        def conv(c: SimpleNamespace):
            return c.dropout(c.pool(c.conv(x))) if c.pool else c.dropout(c.conv(x).reshape((0, -1)))

        # (batches, input.row, input.col) -> (batches, 1, input.row, input.col)
        x = x.reshape((0, 1, x.shape[1], x.shape[2]))

        # conv: [(batches, filters, maxlen - ngram + 1, 1) for ngram in ngrams]
        # pool: [(batches, filters, 1, 1) for ngram in ngrams]
        # reshape: [(batches, filters * x * y) for ngram in ngrams]
        t = [c(x) for c in self.conv2d]
        print('=====')
        for a in t: print(a.shape)
        print('=====')
        x = nd.concat(*t, dim=1)

        # output layer
        y = self.output(x)
        return y


def demo():
    ngrams = range(1, 4)
    train_size = 10
    batch_size = 3
    ctx = mx.cpu()

    input_config = SimpleNamespace(row=5, col=20)
    output_config = SimpleNamespace(dim=3)
    conv2d_config = [SimpleNamespace(ngram=i, filters=4, activation='relu', dropout=0.0) for i in ngrams]

    xs = nd.array([np.random.rand(input_config.row, input_config.col) for _ in range(train_size)])
    ys = nd.array(np.random.randint(output_config.dim, size=train_size))

    model = CNN(input_config, output_config, conv2d_config)
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
