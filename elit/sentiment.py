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
from types import SimpleNamespace

import mxnet as mx
import numpy as np
from mxnet import gluon

from elit.component import NLPComponent
from elit.state import NLPState

__author__ = 'Jinho D. Choi'


class SentenceClassificationState(NLPState):
    def __init__(self, document, vsm, label_map, maxlen, key, key_out=None):
        """
        SentenceClassificationState labels each sentence in the input document with a certain class
        (e.g., positive or negative for sentiment analysis).
        :param document: an input document.
        :type document: elit.structure.Document
        :param vsm: a vector space model for word embeddings.
        :type vsm: elit.vsm.VectorSpaceModel
        :param label_map: the mapping between class labels and their unique IDs.
        :type label_map: elit.vsm.LabelMap
        :param maxlen: the maximum length of each sentence.
        :type maxlen: int
        :param key: the key to each sentence in the input document where the inferred labels (self.labels) are saved.
        :type key: str
        :param key_out: the key to each sentence in the input document where the predicted outputs (self.outputs) are saved.
        :type key_out: str
        """
        super().__init__(document)
        self.embs = [vsm.document_matrix(sen.tokens, maxlen) for sen in document]
        self.label_map = label_map
        self.key = key
        self.key_out = key_out if key_out else key + '-out'

    def process(self, outputs):
        """
        Saves the predicted outputs to self.outputs.
        :param outputs: a matrix where each row contains the prediction scores for the corresponding sentence.
        :param outputs: numpy.array
        """
        self.outputs = outputs

    def finalize(self):
        """
        Saves the predicted outputs (self.outputs) and the inferred labels (self.labels) to the input document once decoding is done.
        """
        for i, labels in enumerate(self.labels):
            d = self.document.get_sentence(i)
            d[self.key] = labels
            d[self.key_out] = self.outputs[i]

    def eval(self, metric):
        """
        :param metric: the accuracy metric.
        :type metric: elit.util.Accuracy
        """
        pass


    def reset(self):
        """
        Nothing to reset.
        """
        pass

    def has_next(self):
        """
        No use for this class.
        :return: False.
        """
        return False





    @property
    def labels(self):
        return [self.label_map.get(np.argmax(output)) for output in self.outputs]

    @property
    def x(self):
        """
        :return: the n * d matrix where n = # of feature_windows and d = sum(vsm_list) + position emb + label emb
        """
        t = len(self.document.get_sentence(self.sen_id))
        l = ([x_extract(self.tok_id, w, t, emb[self.sen_id], zero) for w in self.feature_windows] for emb, zero in self.embs)
        n = np.column_stack(l)
        return n

    @property
    def y(self):
        pass


class DocumentClassificationCNNModel(gluon.Block):
    def __init__(self, input_config, output_config, conv2d_config, **kwargs):
        super().__init__(**kwargs)




        def pool(c):
            if c.pool is None: return None
            p = mx.gluon.nn.MaxPool2D if c.pool == 'max' else mx.gluon.nn.AvgPool2D
            return mx.gluon.nn.MaxPool2D(pool_size=(input_config.row - c.ngram + 1, 1))




        self.conv2d = [SimpleNamespace(
            conv=mx.gluon.nn.Conv2D(channels=c.filters, kernel_size=(c.ngram, input_config.dim), strides=(1, input_config.dim), activation=c.activation),
            dropout=mx.gluon.nn.Dropout(c.dropout)) for c in conv2d_config] if conv2d_config else None






class SentimentAnalyzer(NLPComponent):
    def __init__(self, ctx, vsm):
        """
        :param ctx:
        :type ctx: mx.
        :param vsm:
        """
        super().__init__(ctx)
        self.vsm = vsm

        # to be initialized
        self.label_map = None
