import mxnet as mx
import numpy as np
import logging


class SimpleIter:
    def __init__(self, num_classes, num_features, batch_size, num_batches=1):
        self.num_classes = num_classes
        self.num_features = num_features
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.data_shape = (batch_size, num_features)
        self.label_shape = (batch_size,)

        self.mu = np.random.rand(num_classes, num_features)
        self.sigma = np.ones((num_classes, num_features)) * 0.1
        self.curr_batch = 0

    def __iter__(self):
        return self

    def reset(self):
        self.curr_batch = 0

    def __next__(self):
        return self.next()

    @property
    def provide_data(self):
        return [('data', self.data_shape)]

    @property
    def provide_label(self):
        return [('softmax_label', self.label_shape)]

    def next(self):
        if self.curr_batch < self.num_batches:
            self.curr_batch += 1
            label = np.random.randint(0, num_classes, self.label_shape)
            data = np.zeros(self.data_shape)
            for i in range(num_classes):
                data[label==i,:] = np.random.normal(self.mu[i,:], self.sigma[i,:], (sum(label==i), self.data_shape[1]))
            return mx.io.DataBatch(data=[mx.nd.array(data)], label=[mx.nd.array(label)], pad=0)
        else:
            raise StopIteration

def fit(module: mx.module.Module, train_data, dev_data,
        kvstore='local',
        optimizer='sgd', optimizer_params=(('learning_rate', 0.01),),
        initializer=mx.initializer.Uniform(0.01),
        arg_params=None, aux_params=None, allow_missing=False,
        force_rebind=False, force_init=False, num_epoch=1):
    module.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label, for_training=True,
              force_rebind=force_rebind)
    module.init_params(initializer=initializer, arg_params=arg_params, aux_params=aux_params,
                     allow_missing=allow_missing, force_init=force_init)
    module.init_optimizer(kvstore=kvstore, optimizer=optimizer, optimizer_params=optimizer_params)

    ################################################################################
    # training loop
    ################################################################################
    for epoch in range(num_epoch):
        for data_batch in train_data:
            print(module.data_shapes[0][1])
            module.forward_backward(data_batch)
            module.update()

        y = module.predict(dev_data)
        print(y)

        # sync aux params across devices
        arg_params, aux_params = module.get_params()
        module.set_params(arg_params, aux_params)

        # end of 1 epoch, reset the data-iter for another epoch
        train_data.reset()

num_classes = 3
num_features = 250
trn_size = 1000
dev_size = 500
tst_size = 300
batch_size = 200

net = mx.sym.Variable('data')
net = mx.sym.FullyConnected(net, name='fc1', num_hidden=64)
net = mx.sym.Activation(net, name='relu1', act_type="relu")
net = mx.sym.FullyConnected(net, name='fc2', num_hidden=num_classes)
net = mx.sym.SoftmaxOutput(net, name='softmax')
mod = mx.mod.Module(symbol=net,
                    context=mx.cpu(),
                    data_names=['data'],
                    label_names=['softmax_label'])

trn=mx.io.NDArrayIter(np.zeros((trn_size, num_features)),label=np.zeros((trn_size,)),batch_size=batch_size)
dev=mx.io.NDArrayIter(np.zeros((dev_size, num_features)),batch_size=batch_size)
fit(mod, trn, dev, num_epoch=1)


'''
mod = mx.mod.Module(symbol=sym, context=mx.cpu())
mod.set_params(arg_params, aux_params)

trn=mx.io.NDArrayIter(np.ones((trn_size, num_features)),label=np.zeros((trn_size,)),batch_size=batch_size)
dev=mx.io.NDArrayIter(np.zeros((dev_size, num_features)),label=np.zeros((dev_size,)),batch_size=batch_size)
fit(mod, trn, dev, num_epoch=1, force_rebind=True)

mod.fit(trn,
        optimizer='sgd',
        optimizer_params={'learning_rate':0.1},
        eval_metric='acc',
        num_epoch=1)

tst=mx.io.NDArrayIter(np.zeros((tst_size, num_features)),label=np.zeros((tst_size,)),batch_size=batch_size)
ys = mod.predict(tst)
print(ys)

mod.fit(trn,
        eval_data=dev,
        optimizer='sgd',
        optimizer_params={'learning_rate':0.1},
        eval_metric='acc',
        begin_epoch=1,
        num_epoch=1)

tst=mx.io.NDArrayIter(np.zeros((tst_size, num_features)),label=np.zeros((tst_size,)),batch_size=batch_size)
ys = mod.predict(tst)
print(ys)
'''