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
from keras.layers import Conv1D, Average, Multiply
from keras.layers import Dense, AveragePooling1D, Input, Lambda
from keras.models import Model
from keras.preprocessing import sequence
from keras import backend as K

import os
import gensim
import numpy as np

__author__ = 'Bonggun Shin'

from elit.tokenizer import english_tokenizer

# english_tokenizer.tokenize("string")

class SentimentAnalysis():
    def __init__(self, w2v_dim=400, maxlen=60, w2v_path='../../resources/sentiment/w2v/w2v-400-semevaltrndev.gnsm'):
        self.w2v_dim = w2v_dim
        self.maxlen = maxlen
        self.embedding, self.vocab = self.load_embedding(w2v_path)
        self.load_model()

    def load_embedding(self, w2v_path):
        print('Loading w2v...')
        emb_model = gensim.models.KeyedVectors.load(w2v_path, mmap='r')

        print('creating w2v mat...')
        word_index = emb_model.vocab
        embedding_matrix = np.zeros((len(word_index) + 1, 400), dtype=np.float32)
        for word, i in word_index.items():
            embedding_vector = emb_model[word]
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i.index] = embedding_vector

        return embedding_matrix, emb_model.vocab

    def load_model(self, model_path = '../../resources/sentiment/model/s17-400-v2'):
        filter_sizes = (1, 2, 3, 4, 5)
        num_filters = 80
        hidden_dims = 20

        def prediction_model(model_input, model_path):
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
            model.load_weights(model_path)
            model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

            return model

        def attention_model(model_input):
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

        input_shape = (self.maxlen, self.w2v_dim)
        model_input = Input(shape=input_shape)
        self.p_model = prediction_model(model_input, model_path)
        self.a_model = attention_model(model_input)

        for i in range(len(self.a_model.layers)):
            self.a_model.layers[i].set_weights(self.p_model.layers[i].get_weights())

    def preprocess_x(self, sentences):
        x = []
        for s in sentences:
            one_doc = []
            for token in s:
                try:
                    one_doc.append(self.vocab[token[0]].index)
                except:
                    one_doc.append(len(self.vocab))

            x.append(one_doc)

        x = np.array(x)
        sentence_len_list = [len(sentence) for sentence in x]

        x = sequence.pad_sequences(x, maxlen=self.maxlen)
        x = self.embedding[x]

        return x, sentence_len_list

    def decode(self, sentences):
        x, sentence_len_list = self.preprocess_x(sentences)
        y = self.p_model.predict(x, batch_size=2000, verbose=0)
        attention_matrix = self.a_model.predict(x, batch_size=2000, verbose=0)

        all_att = []
        for sample_index in range(len(sentence_len_list)):
            one_sample_att = []
            for gram_index in range(5):
                norm_one_sample = attention_matrix[gram_index][sample_index][0] / max(
                    attention_matrix[gram_index][sample_index][0])
                one_sample_att.append(norm_one_sample[-sentence_len_list[sample_index] + gram_index:])

            all_att.append(one_sample_att)


        return y, all_att
