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

import numpy as np
from keras import backend as K
from keras.layers import Conv1D, Average, Multiply
from keras.layers import Dense, AveragePooling1D, Input, Lambda
from keras.models import Model
import tensorflow as tf

__author__ = 'Bonggun Shin, Jinho D. Choi'


class SentimentAnalyzer(object):
    @abc.abstractmethod
    def decode(self, documents):
        """
        :param documents: the list of input documents, where each document is a list of tokens.
        :type documents: list<list<str>>
        :return:
            the tuple of (predictions[, custom values]*), where the predictions are stored in a nx3 matrix (n = len(documents).
            the prediction of each document is represented by [negative score, neutral score, positive score].
        :rtype: (list of [float, float, float], *args)
        """
        return


class CNNSentimentAnalyzer(SentimentAnalyzer):
    def __init__(self, emb_model, model_path, maxlen=60):
        self.emb_model = emb_model
        self.model_path = model_path
        self.maxlen = maxlen
        self.p_model, self.a_model = self.load_model()
        self.graph = tf.get_default_graph()

    @abc.abstractmethod
    def prediction_model(self, model_input):
        return

    @abc.abstractmethod
    def attention_model(self, model_input):
        return

    def load_model(self):
        input_shape = (self.maxlen, self.emb_model.dim)
        model_input = Input(shape=input_shape)
        p_model = self.prediction_model(model_input)
        a_model = self.attention_model(model_input)

        for i in range(len(a_model.layers)):
            a_model.layers[i].set_weights(p_model.layers[i].get_weights())

        return p_model, a_model

    def decode(self, documents, batch_size=2000, att=False):
        x = self.emb_model.docs_to_emb(documents, self.maxlen)

        with self.graph.as_default():
            y = self.p_model.predict(x, batch_size=batch_size, verbose=0)
            all_norm_att = []
            # all_raw_att = []
            if not att: return y, all_norm_att

            document_len_list = [len(document) for document in documents]
            attention_matrix = self.a_model.predict(x, batch_size=batch_size, verbose=0)

            for sample_index in range(len(document_len_list)):
                sample_norm_att_list = []
                sample_raw_att_list = []
                for gram_index in range(5):
                    sample_raw_att = attention_matrix[gram_index][sample_index][0]
                    sample_norm_att = sample_raw_att / max(sample_raw_att)
                    valid_token_start_index = -document_len_list[sample_index] + gram_index
                    if valid_token_start_index>=0:
                        valid_token_start_index = len(sample_norm_att)

                    sample_norm_att_list.append(sample_norm_att[valid_token_start_index:])
                    sample_raw_att_list.append(sample_raw_att[valid_token_start_index:])

                new_norm_att = np.zeros([5, len(sample_norm_att_list[0])])
                new_norm_att[0, :] = sample_norm_att_list[0]

                new_raw_att = np.zeros([5, len(sample_raw_att_list[0])])
                new_raw_att[0, :] = sample_raw_att_list[0]

                raw_att_avg = np.zeros([1, len(sample_raw_att_list[0])])
                for i in range(1, 5):
                    for j in range(len(sample_norm_att_list[i])):
                        ngram = i + 1
                        for n in range(ngram):
                            if len(sample_norm_att_list[i]) != 0:
                                new_norm_att[i, j + n] += sample_norm_att_list[i][j] / (ngram)
                                new_raw_att[i, j + n] += sample_raw_att_list[i][j] / (ngram)

                    if max(new_norm_att[i]) != 0:
                        new_norm_att[i] = new_norm_att[i] / max(new_norm_att[i])

                raw_att_avg[0] = new_raw_att.sum(0)
                norm_att_avg = raw_att_avg / max(raw_att_avg[0])
                all_norm_att.append(np.concatenate((norm_att_avg, new_norm_att), axis=0))
                # all_raw_att.append(np.concatenate((raw_att_avg, new_raw_att), axis=0))

        return y, all_norm_att


class TwitterSentimentAnalyzer(CNNSentimentAnalyzer):
    def __init__(self, emb_model, model_path):
        super(TwitterSentimentAnalyzer, self).__init__(emb_model=emb_model, model_path=model_path, maxlen=60)

    def prediction_model(self, model_input):
        print('Init: '+self.model_path)
        filter_sizes = (1, 2, 3, 4, 5)
        num_filters = 80
        hidden_dims = 20
        conv_blocks = []
        for sz in filter_sizes:
            conv = Conv1D(num_filters,
                          sz,
                          padding="valid",
                          activation="relu",
                          strides=1)(model_input)
            print(conv)
            conv = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1)))(conv)
            conv = AveragePooling1D(pool_size=num_filters)(conv)

            attention_size = self.maxlen - sz + 1

            multiplied_vector_list = []
            for i in range(attention_size):
                selected_attention = Lambda(lambda x: x[:, 0, i] / float(sz))(conv)

                for j in range(sz):
                    selected_token = Lambda(lambda x: x[:, i + j, :])(model_input)
                    multiplied_vector = Lambda(lambda x: Multiply()(x))([selected_token, selected_attention])

                    multiplied_vector_list.append(multiplied_vector)

            attentioned_conv = Average()(multiplied_vector_list)

            print(attentioned_conv)
            conv_blocks.append(attentioned_conv)

        z = Average()(conv_blocks)
        z = Dense(hidden_dims, activation="relu")(z)
        model_output = Dense(3, activation="softmax")(z)

        model = Model(model_input, model_output)
        model.load_weights(self.model_path)
        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

        return model

    def attention_model(self, model_input):
        filter_sizes = (1, 2, 3, 4, 5)
        num_filters = 80
        att_list = []
        for sz in filter_sizes:
            conv = Conv1D(num_filters,
                          sz,
                          padding="valid",
                          activation="relu",
                          strides=1)(model_input)
            conv = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1)))(conv)
            att = AveragePooling1D(pool_size=num_filters)(conv)
            att_list.append(att)

        model = Model(model_input, att_list)
        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

        return model


class MovieSentimentAnalyzer(CNNSentimentAnalyzer):
    def __init__(self, emb_model, model_path):
        super(MovieSentimentAnalyzer, self).__init__(emb_model=emb_model, model_path=model_path, maxlen=100)

    def prediction_model(self, model_input):
        print('Init: ' + self.model_path)
        filter_sizes = (1, 2, 3, 4, 5)
        num_filters = 64
        hidden_dims = 50
        conv_blocks = []
        for sz in filter_sizes:
            conv = Conv1D(num_filters,
                          sz,
                          padding="valid",
                          activation="relu",
                          strides=1)(model_input)
            print(conv)
            conv = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1)))(conv)
            conv = AveragePooling1D(pool_size=num_filters)(conv)

            attention_size = self.maxlen - sz + 1

            multiplied_vector_list = []
            for i in range(attention_size):
                selected_attention = Lambda(lambda x: x[:, 0, i] / float(sz))(conv)

                for j in range(sz):
                    selected_token = Lambda(lambda x: x[:, i + j, :])(model_input)
                    multiplied_vector = Lambda(lambda x: Multiply()(x))([selected_token, selected_attention])

                    multiplied_vector_list.append(multiplied_vector)

            attentioned_conv = Average()(multiplied_vector_list)

            print(attentioned_conv)
            conv_blocks.append(attentioned_conv)

        z = Average()(conv_blocks)
        z = Dense(hidden_dims, activation="relu")(z)
        model_output = Dense(3, activation="softmax")(z)

        model = Model(model_input, model_output)
        model.load_weights(self.model_path)
        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

        return model

    def attention_model(self, model_input):
        filter_sizes = (1, 2, 3, 4, 5)
        num_filters = 64
        att_list = []
        for sz in filter_sizes:
            conv = Conv1D(num_filters,
                          sz,
                          padding="valid",
                          activation="relu",
                          strides=1)(model_input)
            conv = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1)))(conv)
            att = AveragePooling1D(pool_size=num_filters)(conv)
            att_list.append(att)

        model = Model(model_input, att_list)
        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

        return model
