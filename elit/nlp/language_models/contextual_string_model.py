# -*- coding:utf-8 -*-
# Author：ported from PyTorch implementation of flair: https://github.com/zalandoresearch/flair to MXNet
# Date: 2018-09-13 18:55
import datetime
import math
import os
import pickle
import time
from typing import List, Dict

import mxnet as mx
import mxnet.ndarray as nd
from mxnet import gluon, autograd
from mxnet.gluon import nn, rnn
from mxnet.gluon.loss import SoftmaxCrossEntropyLoss

from elit.nlp.dep.common.utils import make_sure_path_exists
from elit.nlp.tagger.corpus import Dictionary, TextCorpus
from elit.nlp.tagger.lm_config import LanguageModelConfig
from elit.nlp.tagger.mxnet_util import mxnet_prefer_gpu
from elit.nlp.tagger.reduce_lr_on_plateau import ReduceLROnPlateau


class ContextualStringModel(nn.Block):
    """
    Container module with an encoder, a recurrent module, and a decoder.
    Ported from PyTorch implementation https://github.com/zalandoresearch/flair
    """

    def __init__(self,
                 dictionary: Dictionary,
                 is_forward_lm: bool,
                 hidden_size: int,
                 nlayers: int,
                 embedding_size: int = 100,
                 nout=None,
                 dropout=0.5,
                 init_params: Dict = None):

        super(ContextualStringModel, self).__init__()

        self.dictionary = dictionary
        self.is_forward_lm = is_forward_lm

        self.dropout = dropout
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.nlayers = nlayers

        with self.name_scope():
            self.drop = nn.Dropout(dropout)
            self.encoder = nn.Embedding(len(dictionary), embedding_size, weight_initializer=mx.initializer.Constant(
                init_params['encoder.weight']) if init_params else mx.initializer.Uniform(0.1))

            if nlayers == 1:
                if init_params:
                    self.rnn = rnn.LSTM(hidden_size, nlayers, dropout=dropout, input_size=embedding_size,
                                        i2h_weight_initializer=mx.initializer.Constant(init_params['rnn.weight_ih_l0']),
                                        h2h_weight_initializer=mx.initializer.Constant(init_params['rnn.weight_hh_l0']),
                                        i2h_bias_initializer=mx.initializer.Constant(init_params['rnn.bias_ih_l0']),
                                        h2h_bias_initializer=mx.initializer.Constant(init_params['rnn.bias_hh_l0'])
                                        )
                else:
                    self.rnn = rnn.LSTM(hidden_size, nlayers, input_size=embedding_size)
            else:
                self.rnn = rnn.LSTM(hidden_size, nlayers, dropout=dropout, input_size=embedding_size)

            self.hidden = None

            self.nout = nout
            if nout is not None:
                self.proj = nn.Dense(nout, weight_initializer='Xavier', in_units=hidden_size)
                self.decoder = nn.Dense(len(dictionary), weight_initializer=mx.initializer.Uniform(0.1),
                                        bias_initializer='zero', in_units=nout)
            else:
                self.proj = None
                self.decoder = nn.Dense(len(dictionary), weight_initializer=mx.initializer.Constant(
                    init_params['decoder.weight']) if init_params else mx.initializer.Uniform(0.1),
                                        bias_initializer=mx.initializer.Constant(
                                            init_params['decoder.bias']) if init_params else 'zero',
                                        in_units=hidden_size)

    def set_hidden(self, hidden):
        self.hidden = hidden

    def forward(self, input, hidden, cell, ordered_sequence_lengths=None):
        encoded = self.encoder(input)
        emb = self.drop(encoded)

        output, hc = self.rnn(emb, [hidden, cell])
        hidden, cell = hc[0], hc[1]

        if self.proj is not None:
            output = self.proj(output)

        output = self.drop(output)

        decoded = self.decoder(output.reshape(-1, output.shape[2]))

        return decoded.reshape(output.shape[0], output.shape[1], decoded.shape[1]), output, hidden, cell

    def get_representation(self, strings: List[str], detach_from_lm=True):

        sequences_as_char_indices = []
        for string in strings:
            char_indices = [self.dictionary.get_idx_for_item(char) for char in string]
            sequences_as_char_indices.append(char_indices)

        batch = nd.array(sequences_as_char_indices).transpose((1, 0))  # (T, N)

        hidden = self.init_hidden(len(strings))
        prediction, rnn_output, hidden, cell = self.forward(batch, hidden, hidden.copy())

        if detach_from_lm:
            rnn_output = self.repackage_hidden(rnn_output)

        return rnn_output

    def repackage_hidden(self, h: nd.NDArray):
        """Wraps hidden states in new Variables, to detach them from their history."""
        return h.detach()
        # if type(h) == torch.Tensor:
        #     return Variable(h.data)
        # else:
        #     return tuple(self.repackage_hidden(v) for v in h)

    def freeze(self):
        """
        Freeze this model to make it static, thus non trainable
        """
        self.collect_params().setattr('grad_req', 'null')

    @classmethod
    def load_language_model(cls, model_file, context: mx.Context = None):
        config = LanguageModelConfig.load(os.path.join(model_file, 'config.pkl'))
        with context:
            model = ContextualStringModel(config.dictionary,
                                          config.is_forward_lm,
                                          config.hidden_size,
                                          config.nlayers,
                                          config.embedding_size,
                                          config.nout,
                                          config.dropout)
        model.load_parameters(os.path.join(model_file, 'model.bin'), ctx=context)
        return model

    @staticmethod
    def load_dumped_model(pkl_file):
        with open(pkl_file, 'rb') as f:
            params = pickle.load(f)
            dictionary = Dictionary(add_unk=False)

            for k, v in params['dictionary'][0].items():
                k = k.decode('UTF-8')
                if k in dictionary.item2idx:
                    print('conflict in char mapping')
                    k = k + '-'
                dictionary.item2idx[k] = v
            for k in params['dictionary'][1]:
                dictionary.idx2item.append(k.decode('UTF-8'))
            config = LanguageModelConfig(dictionary, params['is_forward_lm'], params['hidden_size'],
                                         params['nlayers'], params['embedding_size'], params['nout'], params['dropout'])
            model = ContextualStringModel(config.dictionary,
                                          config.is_forward_lm,
                                          config.hidden_size,
                                          config.nlayers,
                                          config.embedding_size,
                                          config.nout,
                                          config.dropout,
                                          params)
            return model

    def save(self, file):
        config = LanguageModelConfig(
            dictionary=self.dictionary,
            is_forward_lm=self.is_forward_lm,
            hidden_size=self.hidden_size,
            nlayers=self.nlayers,
            embedding_size=self.embedding_size,
            nout=self.nout,
            dropout=self.dropout
        )
        make_sure_path_exists(file)
        config.save(os.path.join(file, 'config.pkl'))
        self.save_parameters(os.path.join(file, 'model.bin'))

    def init_hidden(self, mini_batch_size) -> nd.NDArray:
        return nd.zeros((self.nlayers, mini_batch_size, self.hidden_size))


class ContextualStringModelTrainer:
    """
    Ported from PyTorch implementation https://github.com/zalandoresearch/flair
    """

    def __init__(self, model: ContextualStringModel, corpus: TextCorpus, test_mode: bool = False):
        self.model = model
        self.corpus = corpus
        self.test_mode = test_mode

        self.loss_function = SoftmaxCrossEntropyLoss()
        self.log_interval = 100

    def train(self,
              base_path: str,
              sequence_length: int,
              learning_rate: float = 20,
              mini_batch_size: int = 100,
              anneal_factor: float = 0.25,
              patience: int = 10,
              clip=0.25,
              max_epochs: int = 10000):

        number_of_splits = len(self.corpus.train_files)
        val_data = self._batchify(self.corpus.valid, mini_batch_size)

        os.makedirs(base_path, exist_ok=True)
        loss_txt = os.path.join(base_path, 'loss.txt')
        savefile = os.path.join(base_path, 'best-lm.pt')

        try:
            with mx.Context(mxnet_prefer_gpu()):
                self.model.initialize()
                best_val_loss = 100000000
                scheduler = ReduceLROnPlateau(lr=learning_rate, verbose=True, factor=anneal_factor,
                                              patience=patience)
                optimizer = mx.optimizer.SGD(learning_rate=learning_rate, lr_scheduler=scheduler)
                trainer = gluon.Trainer(self.model.collect_params(),
                                        optimizer=optimizer)

                for epoch in range(1, max_epochs + 1):

                    print('Split %d' % epoch + '\t - ({:%H:%M:%S})'.format(datetime.datetime.now()))

                    # for group in optimizer.param_groups:
                    #     learning_rate = group['lr']

                    train_slice = self.corpus.get_next_train_slice()

                    train_data = self._batchify(train_slice, mini_batch_size)
                    print('\t({:%H:%M:%S})'.format(datetime.datetime.now()))

                    # go into train mode
                    # self.model.train()

                    # reset variables
                    epoch_start_time = time.time()
                    total_loss = 0
                    start_time = time.time()

                    hidden = self.model.init_hidden(mini_batch_size)
                    cell = hidden.copy()

                    # not really sure what this does
                    ntokens = len(self.corpus.dictionary)

                    # do batches
                    for batch, i in enumerate(range(0, len(train_data) - 1, sequence_length)):

                        data, targets = self._get_batch(train_data, i, sequence_length)

                        # Starting each batch, we detach the hidden state from how it was previously produced.
                        # If we didn't, the model would try backpropagating all the way to start of the dataset.
                        hidden = self._repackage_hidden(hidden)
                        cell = self._repackage_hidden(cell)

                        # self.model.zero_grad()
                        # optimizer.zero_grad()

                        # do the forward pass in the model
                        with autograd.record():
                            output, rnn_output, hidden, cell = self.model.forward(data, hidden, cell)
                            # try to predict the targets
                            loss = self.loss_function(output.reshape(-1, ntokens), targets).mean()
                            loss.backward()

                        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)

                        trainer.step(mini_batch_size)

                        total_loss += loss.asscalar()

                        if batch % self.log_interval == 0 and batch > 0:
                            cur_loss = total_loss.item() / self.log_interval
                            elapsed = time.time() - start_time
                            print('| split {:3d} /{:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                                  'loss {:5.2f} | ppl {:8.2f}'.format(
                                epoch, number_of_splits, batch, len(train_data) // sequence_length,
                                                                elapsed * 1000 / self.log_interval, cur_loss,
                                self._safe_exp(cur_loss)))
                            total_loss = 0
                            start_time = time.time()

                    print('epoch {} done! \t({:%H:%M:%S})'.format(epoch, datetime.datetime.now()))
                    scheduler.step(cur_loss)

                    ###############################################################################
                    # TEST
                    ###############################################################################
                    # skip evaluation
                    # val_loss = self.evaluate(val_data, mini_batch_size, sequence_length)
                    # scheduler.step(val_loss)
                    #
                    # # Save the model if the validation loss is the best we've seen so far.
                    # if val_loss < best_val_loss:
                    #     self.model.save(savefile)
                    #     best_val_loss = val_loss
                    #     print('best loss so far {:5.2f}'.format(best_val_loss))
                    val_loss = cur_loss
                    if (self.corpus.current_train_file_index + 1) % 100 == 0 or self.corpus.is_last_slice:
                        self.model.save(savefile)

                    ###############################################################################
                    # print info
                    ###############################################################################
                    print('-' * 89)

                    local_split_number = epoch % number_of_splits
                    if local_split_number == 0: local_split_number = number_of_splits

                    summary = '| end of split {:3d} /{:3d} | epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | ' \
                              'valid ppl {:8.2f} | learning rate {:3.2f}'.format(local_split_number,
                                                                                 number_of_splits,
                                                                                 epoch,
                                                                                 (time.time() - epoch_start_time),
                                                                                 val_loss,
                                                                                 self._safe_exp(val_loss),
                                                                                 learning_rate)

                    with open(loss_txt, "a") as myfile:
                        myfile.write('%s\n' % summary)

                    print(summary)
                    print('-' * 89)

        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

    @staticmethod
    def _safe_exp(val_loss):
        return float('nan') if val_loss > 100 else math.exp(val_loss)

    def evaluate(self, data_source, eval_batch_size, sequence_length):
        # Turn on evaluation mode which disables dropout.
        # self.model.eval()
        total_loss = 0
        ntokens = len(self.corpus.dictionary)

        hidden = self.model.init_hidden(eval_batch_size)
        cell = hidden.copy()

        for i in range(0, len(data_source) - 1, sequence_length):
            data, targets = self._get_batch(data_source, i, sequence_length)
            prediction, rnn_output, hidden, cell = self.model.forward(data, hidden, cell)
            output_flat = prediction.reshape(-1, ntokens)
            total_loss += len(data) * self.loss_function(output_flat, targets).mean()
            hidden = self._repackage_hidden(hidden)
            cell = cell.detach()
        return total_loss.asscalar() / len(data_source)

    @staticmethod
    def _batchify(data: nd.NDArray, batch_size):
        """
        Make a batch tensor out of a vector
        :param data: vector
        :param batch_size: NN
        :return: (IN,NN) tensor
        """
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = len(data) // batch_size
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data[0: nbatch * batch_size]
        # Evenly divide the data across the bsz batches.
        data = data.reshape(batch_size, -1).transpose()
        # if torch.cuda.is_available():
        #     data = data.cuda()
        return data

    @staticmethod
    def _get_batch(source: nd.NDArray, i, sequence_length):
        seq_len = min(sequence_length, len(source) - 1 - i)
        data = source[i:i + seq_len]
        target = source[i + 1:i + 1 + seq_len].reshape(-1)
        return data, target

    @staticmethod
    def _repackage_hidden(h: nd.NDArray):
        """Wraps hidden states in new Variables, to detach them from their history."""
        return h.detach()


def _convert_dumped_model():
    for path in ['data/model/lm-news-forward', 'data/model/lm-news-backward']:
        model = ContextualStringModel.load_dumped_model(path + '/params.pkl')
        model.initialize()
        model.save(path)


def _train():
    corpus = TextCorpus('data/raw')
    language_model = ContextualStringModel(corpus.dictionary,
                                           is_forward_lm=False,
                                           hidden_size=1024,
                                           nlayers=1,
                                           dropout=0.25)
    trainer = ContextualStringModelTrainer(language_model, corpus)
    trainer.train('data/model/lm-jumbo-backward1024',
                  sequence_length=250,
                  mini_batch_size=100,
                  max_epochs=99999)
    # LanguageModel.load_language_model('data/model/lm')


def _load():
    lm = ContextualStringModel.load_language_model('data/model/lm-news-forward')


if __name__ == '__main__':
    _train()
    # _convert_dumped_model()
    # _load()
