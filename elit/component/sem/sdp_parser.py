# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-06-08 18:28
from typing import Sequence

from elit.component.dep.common.utils import fetch_resource
from elit.component.nlp import NLPComponent
from elit.component.sem.data import SDPDataLoader, conll_to_sdp_document
from elit.component.sem.biaffine_sdp import BiaffineSDPParser
from elit.component.tagger.mxnet_util import mxnet_prefer_gpu
from elit.resources.pre_trained_models import SDP_JUMBO
from elit.structure import Document, SEM


class SDPParser(NLPComponent):

    def __init__(self, context=mxnet_prefer_gpu()) -> None:
        super().__init__()
        self.context = context
        self._parser = None  # type: BiaffineSDPParser

    def train(self, trn_docs: Sequence[Document], dev_docs: Sequence[Document], model_path: str,
              pretrained_embeddings_file=None, min_occur_count=2,
              lstm_layers=3, word_dims=100, tag_dims=100, dropout_emb=0.33, lstm_hiddens=400,
              dropout_lstm_input=0.33, dropout_lstm_hidden=0.33, mlp_arc_size=500, mlp_rel_size=100,
              dropout_mlp=0.33, learning_rate=1e-3, decay=.75, decay_steps=5000, beta_1=.9, beta_2=.9, epsilon=1e-12,
              num_buckets_train=40,
              num_buckets_valid=10, train_iters=50000, train_batch_size=5000, dev_batch_size=5000, validate_every=100,
              save_after=5000, root='root', bert_path=None, debug=False):
        """Train a deep biaffine dependency parser

        Parameters
        ----------
        train_file : str
            path to training set
        dev_file : str
            path to dev set
        save_dir : str
            a directory for saving model and related meta-data
        pretrained_embeddings_file : tuple
            pre-trained embeddings
        min_occur_count : int
            threshold of rare words, which will be replaced with UNKs,
        lstm_layers : int
            layers of lstm
        word_dims : int
            dimension of word embedding
        tag_dims : int
            dimension of tag embedding
        dropout_emb : float
            word dropout
        lstm_hiddens : int
            size of lstm hidden states
        dropout_lstm_input : int
            dropout on x in variational RNN
        dropout_lstm_hidden : int
            dropout on h in variational RNN
        mlp_arc_size : int
            output size of MLP for arc feature extraction
        mlp_rel_size : int
            output size of MLP for rel feature extraction
        dropout_mlp : float
            dropout on the output of LSTM
        learning_rate : float
            learning rate
        decay : float
            see ExponentialScheduler
        decay_steps : int
            see ExponentialScheduler
        beta_1 : float
            see ExponentialScheduler
        beta_2 : float
            see ExponentialScheduler
        epsilon : float
            see ExponentialScheduler
        num_buckets_train : int
            number of buckets for training data set
        num_buckets_valid : int
            number of buckets for dev data set
        train_iters : int
            training iterations
        train_batch_size : int
            training batch size
        dev_batch_size : int
            test batch size
        validate_every : int
            validate on dev set every such number of batches
        save_after : int
            skip saving model in early epochs
        root : str
            token for ROOT
        debug : bool
            debug mode

        Returns
        -------
        BiaffineSDPParser
            parser itself
        """
        parser = self._parser = BiaffineSDPParser()
        parser.train(trn_docs, dev_docs, model_path, pretrained_embeddings_file, min_occur_count, lstm_layers,
                     word_dims, tag_dims, dropout_emb, lstm_hiddens, dropout_lstm_input, dropout_lstm_hidden,
                     mlp_arc_size, mlp_rel_size, dropout_mlp, learning_rate, decay, decay_steps, beta_1, beta_2,
                     epsilon, num_buckets_train, num_buckets_valid, train_iters, train_batch_size, dev_batch_size,
                     validate_every, save_after, root, bert_path, debug)
        return self

    def decode(self, docs: Sequence[Document], num_buckets_test=10, test_batch_size=5000, **kwargs):
        vocab = self._parser._vocab
        for d in docs:
            for s in d:
                s[SEM] = [[(0, vocab.id2rel(0))]] * len(s)  # placeholder
        data_loader = SDPDataLoader(docs, num_buckets_test, self._parser._vocab)
        record = data_loader.idx_sequence
        parser = self._parser._parser
        results = [None] * len(record)
        idx = 0
        # seconds = time.time()
        with self.context:
            for words, bert, tags, arcs, rels in data_loader.get_batches(batch_size=test_batch_size,
                                                                         shuffle=False):
                outputs = parser.forward(words, bert, tags)

                for output in outputs:
                    sent_idx = record[idx]
                    results[sent_idx] = output
                    idx += 1

        idx = 0
        for d in docs:
            for s in d:
                s[SEM] = []
                arcs, rels = results[idx]
                length = arcs.shape[0]
                for i in range(1, length):
                    head_rel = []
                    for j in range(0, length):
                        if arcs[j, i]:
                            head_rel.append((j, data_loader.vocab.id2rel(int(rels[j, i].asscalar()))))
                    s[SEM].append(head_rel)
                idx += 1
        return docs

    def evaluate(self, docs: Sequence[Document], **kwargs):
        return self._parser.evaluate(test_file=docs, context=self.context)

    def load(self, model_path: str=SDP_JUMBO, **kwargs):
        parser = self._parser = BiaffineSDPParser()
        model_path = fetch_resource(model_path)
        parser.load(model_path, self.context)
        return self

    def save(self, model_path: str, **kwargs):
        raise NotImplementedError(
            'save is not implemented, use copy & paste to make a new copy of your model, use train to create a new '
            'model. Why save is needed?')


if __name__ == '__main__':
    save_dir = 'data/model/sdp/jumbo'
    parser = SDPParser()
    parser.load(save_dir)
    docs = [conll_to_sdp_document('data/dat/en-ddr.debug.conll')]
    print(parser.evaluate(docs))
