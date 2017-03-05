import mxnet as mx
import numpy as np

from elit.test.data_iter import SyntheticData

# mlp
num_classes = 3
num_features = 128

net = mx.sym.Variable('data')
net = mx.sym.FullyConnected(net, name='fc1', num_hidden=64)
net = mx.sym.Activation(net, name='relu1', act_type="relu")
net = mx.sym.FullyConnected(net, name='fc2', num_hidden=num_classes)
net = mx.sym.SoftmaxOutput(net, name='softmax')
data = SyntheticData(num_classes, num_features)
mod = mx.mod.Module(symbol=net,
                    context=mx.cpu(),
                    data_names=['data'],
                    label_names=['softmax_label'])

import logging
logging.basicConfig(level=logging.INFO)

batch_size=4
mod.fit(data.get_iter(batch_size),
        optimizer='sgd',
        optimizer_params={'learning_rate':0.1},
        eval_metric='acc',
        num_epoch=1)

print(mod.get_outputs()[0].asnumpy())


#mod.predict(mx.io.NDArrayIter(np.random.rand(num_features)))
#y = mod.predict(data.get_iter(batch_size))



'''
for preds, i_batch, batch in mod.iter_predict(data.get_iter(batch_size)):
    print(preds[0].asnumpy())
    break
    pred_label = preds[0].asnumpy().argmax(axis=1)
    label = batch.label[0].asnumpy().astype('int32')



print('shape of predict: %s' % (y.shape,))
for preds, i_batch, batch in mod.iter_predict(data.get_iter(batch_size)):
    pred_label = preds[0].asnumpy().argmax(axis=1)
    label = batch.label[0].asnumpy().astype('int32')
    print('batch %d, accuracy %f' % (i_batch, float(sum(pred_label==label))/len(label)))
'''