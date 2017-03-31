from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from concurrent.futures import wait

import fasttext
import logging

'''
path = '/home/ubuntu/gdrive/public/word-embeddings/corpus.friends+nyt+wiki+amazon.f2v.skip.d200.bin'
print(path)
v = f2v.load_model(path)

print(type(v))
print(v[''])
print(v['@#r$%'])

import mxnet as mx
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

xs = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], dtype=np.float32)
ys = np.array([0,1,2,3], dtype=np.float32)

net = mx.sym.Variable('data')
net = mx.sym.FullyConnected(net, name='fc1', num_hidden=4)
net = mx.sym.SoftmaxOutput(net, name='softmax')
mod = mx.mod.Module(symbol=net)

dat = mx.io.NDArrayIter(data=xs, label=ys, batch_size=4)
mod.fit(train_data=dat, num_epoch=10)

xxs = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], dtype=np.float32)
dat = mx.io.NDArrayIter(xxs, None, batch_size=8)
mod.bind(dat.provide_data, None, for_training=False, force_rebind=True)
ys = mod.predict(dat).asnumpy()
print(ys[0].shape)
'''


def add(i):
    return i

pool = ThreadPoolExecutor(4)
futures = [pool.submit(add, i) for i in range(4)]

# for a in as_completed(futures): print(a.result())

print(wait(futures)[1])
for future in futures:
    print(type(future))
    print(future.result())