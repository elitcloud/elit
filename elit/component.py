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
import abc
import datetime
import inspect
import logging
import time
from typing import Sequence, Any

from mxnet import gluon, autograd, nd
from mxnet.gluon import Trainer
from tqdm import trange, tqdm

from elit.util.structure import Document, to_gold
from elit.util.vsm import LabelMap

__author__ = 'Jinho D. Choi, Gary Lai'


class Component(abc.ABC):
    """
    :class:`Component` is an abstract class; any component developed in ELIT must inherit this class.

    Abstract methods to be implemented:
      - :meth:`Component.init`
      - :meth:`Component.load`
      - :meth:`Component.save`
      - :meth:`Component.train`
      - :meth:`Component.decode`
      - :meth:`Component.evaluate`
    """

    @abc.abstractmethod
    def init(self, **kwargs):
        """
        :param kwargs: custom parameters.

        Initializes this component.
        """
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))

    @abc.abstractmethod
    def load(self, model_path: str, **kwargs):
        """
        :param model_path: the filepath where a model can be loaded.

        Loads a model for this component from the filepath.
        """
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))

    @abc.abstractmethod
    def save(self, model_path: str, **kwargs):
        """
        :param model_path: the filepath where the current model can be saved.

        Saves the current model of this component to the filepath.
        """
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))

    @abc.abstractmethod
    def train(self, trn_data: Any, dev_data: Any, model_path: str, **kwargs) -> float:
        """
        :param trn_data: training data.
        :param dev_data: development (validation) data.
        :param model_path: the filepath where the trained model(s) are to be saved.
        :return: the best score form the development data.

        Trains a model for this component and saves the model to the filepath.
        """
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))

    @abc.abstractmethod
    def decode(self, data: Any, **kwargs):
        """
        :param data: input data.

        Processes the input data, make predictions, and saves the predicted labels back to the
        input data.
        """
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))

    @abc.abstractmethod
    def evaluate(self, data: Any, **kwargs):
        """
        :param data: input data.

        Evaluates the current model of this component with the input data.
        """
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))


class NLPComponent(Component):
    """
    :class:`NLPComponent` is an abstract class; any NLP component developed in ELIT must inherit
    this class.
    It is similar to :class:`Component` except that the type of training and development data is
    specified to :class:`elit.structure.Document`.

    Abstract methods to be implemented:
      - :meth:`Component.init`
      - :meth:`Component.load`
      - :meth:`Component.save`
      - :meth:`NLPComponent.train`
      - :meth:`NLPComponent.decode`
      - :meth:`NLPComponent.evaluate`
    """

    @abc.abstractmethod
    def train(self, trn_docs: Sequence[Document], dev_docs: Sequence[Document], model_path: str, **kwargs) -> float:
        """
        :param trn_docs: the sequence of documents for training.
        :param dev_docs: the sequence of documents for development (validation).
        :param model_path: the filepath where trained model(s) are to be saved.
        :return: the best score form the development data.

        Trains a model for this component and saves the model to the filepath.
        """
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))

    @abc.abstractmethod
    def decode(self, docs: Sequence[Document], **kwargs):
        """
        :param docs: the sequence of input documents.

        Processes the input documents, make predictions, and saves the predicted labels back to the
        input documents.
        """
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))

    @abc.abstractmethod
    def evaluate(self, docs: Sequence[Document], **kwargs):
        """
        :param docs: the sequence of input documents.

        Evaluates the current model of this component with the input documents.
        """
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))


class MXNetComponent(NLPComponent):

    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.model = None

    @abc.abstractmethod
    def init(self, **kwargs):
        pass

    def load(self, model_path: str, **kwargs):
        pass

    def save(self, model_path: str, **kwargs):
        pass

    def train(self, trn_docs: Sequence[Document], dev_docs: Sequence[Document], model_path: str,
              label_map: LabelMap = None,
              epoch=100,
              trn_batch=64,
              dev_batch=2048,
              loss_fn=gluon.loss.SoftmaxCrossEntropyLoss(),
              optimizer='adagrad',
              optimizer_params=None,
              **kwargs):
        if optimizer_params is None:
            optimizer_params = {'learning_rate': 0.01}

        log = ('Configuration',
               '- context(s): {}'.format(self.ctx),
               '- trn_batch size: {}'.format(trn_batch),
               '- dev_batch size: {}'.format(dev_batch),
               '- max epoch : {}'.format(epoch),
               '- loss func : {}'.format(loss_fn),
               '- optimizer : {} <- {}'.format(optimizer, optimizer_params))
        logging.info('\n'.join(log))
        logging.info("Load trn data")
        trn_data = self.data_loader(docs=trn_docs, batch_size=trn_batch, shuffle=True)
        logging.info("Load dev data")
        dev_data = self.data_loader(docs=dev_docs, batch_size=dev_batch, shuffle=False)
        trainer = Trainer(self.model.collect_params(),
                          optimizer=optimizer,
                          optimizer_params=optimizer_params)

        logging.info('Training')
        best_e, best_eval = -1, -1
        self.model.hybridize()

        epochs = trange(1, epoch + 1)
        for e in epochs:
            trn_st = time.time()
            correct, total, = 0, 0
            for i, (data, label) in enumerate(tqdm(trn_data, leave=False)):
                data = data.as_in_context(self.ctx)
                label = label.as_in_context(self.ctx)
                with autograd.record():
                    output = self.model(data)
                    loss = loss_fn(output, label)
                    loss.backward()
                trainer.step(data.shape[0])
                correct += len([1 for o, y in zip(nd.argmax(output, axis=1), label) if int(o.asscalar()) == int(y.asscalar())])
                total += len(label)
            trn_acc = 100.0 * correct / total
            trn_et = time.time()

            dev_st = time.time()
            dev_acc = self.accuracy(data_iterator=dev_data, docs=dev_docs)
            dev_et = time.time()

            if best_eval < dev_acc:
                best_e, best_eval = e, dev_acc
                self.save(model_path=model_path)

            desc = ("epoch: {}".format(e),
                    "trn time: {}".format(datetime.timedelta(seconds=(trn_et - trn_st))),
                    "dev time: {}".format(datetime.timedelta(seconds=(dev_et - dev_st))),
                    "train_acc: {}".format(trn_acc),
                    "dev acc: {}".format(dev_acc),
                    "best epoch: {}".format(best_e),
                    "best eval: {}".format(best_eval))
            epochs.set_description(desc=' '.join(desc))

    def decode(self, docs: Sequence[Document], batch_size: int = 2048, **kwargs):
        data_iterator = self.data_loader(docs=docs, batch_size=batch_size, shuffle=False, label=False)
        preds = []
        idx = 0
        for data, _ in data_iterator:
            data = data.as_in_context(self.ctx)
            output = self.model(data)
            pred = nd.argmax(output, axis=1)
            [preds.append(self.label_map.get(int(p.asscalar()))) for p in pred]

        for doc in docs:
            for sen in doc.sentences:
                try:
                    del sen[to_gold(self.key)]
                except KeyError:
                    pass
                sen[self.key] = preds[idx:idx + len(sen)]
                idx += len(sen)

        return docs

    def evaluate(self, docs: Sequence[Document], batch_size: int = 2048, **kwargs):
        data_iterator = self.data_loader(docs=docs, batch_size=batch_size, shuffle=False, label=False)
        st = time.time()
        acc = self.accuracy(data_iterator=data_iterator, docs=docs)
        et = time.time()
        logging.info('acc: {} time: {} (sec)'.format(acc, et - st))
        return acc, et - st

    @abc.abstractmethod
    def accuracy(self, **kwargs):
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))

    @abc.abstractmethod
    def data_loader(self, **kwargs):
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))

