import mxnet as mx

from elit.test.data_iter import SyntheticData

# mlp
net = mx.sym.Variable('data')
net = mx.sym.FullyConnected(net, name='fc1', num_hidden=64)
net = mx.sym.Activation(net, name='relu1', act_type="relu")
net = mx.sym.FullyConnected(net, name='fc2', num_hidden=10)
net = mx.sym.SoftmaxOutput(net, name='softmax')
data = SyntheticData(10, 128)
mod = mx.mod.Module(symbol=net,
                    context=mx.cpu(),
                    data_names=['data'],
                    label_names=['softmax_label'])
type(mod)
import logging
logging.basicConfig(level=logging.INFO)

batch_size=32
mod.fit(data.get_iter(batch_size),
        eval_data=data.get_iter(batch_size),
        optimizer='sgd',
        optimizer_params={'learning_rate':0.1},
        eval_metric='acc',
        num_epoch=1)
print(type(mod))
y = mod.predict(data.get_iter(batch_size))
print('shape of predict: %s' % (y.shape,))
for preds, i_batch, batch in mod.iter_predict(data.get_iter(batch_size)):
    pred_label = preds[0].asnumpy().argmax(axis=1)
    label = batch.label[0].asnumpy().astype('int32')
    print('batch %d, accuracy %f' % (i_batch, float(sum(pred_label==label))/len(label)))

