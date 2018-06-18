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
import abc
from elitsdk import Component
from elit.util.string import *
from elit.nlp.structure import SENTENCE_ID, TOKEN, OFFSET

__author__ = "Gary Lai"


class Segmenter(Component):
    @abc.abstractmethod
    def decode(self, input_data, offsets, **kwargs):
        pass

    def load(self, model_path, *args, **kwargs):
        pass

    def train(self, trn_data, dev_data, *args, **kwargs):
        pass

    def save(self, model_path, *args, **kwargs):
        pass


class EnglishSegmenter(Segmenter):
    def decode(self, input_data, offsets, **kwargs):
        def sentence(sid, begin, end):
            return {
                SENTENCE_ID: sid,
                TOKEN: input_data[begin:end],
                OFFSET: offsets[begin:end]
            }

        sentences = []
        begin = 0
        sid = 0
        right_quote = True

        for i, token in enumerate(input_data):
            t = token[0]
            if t == '"':
                right_quote = not right_quote

            if begin == i:
                if sentences and (is_right_bracket(t) or t == u'\u201D' or t == '"' and right_quote):
                    d = sentences[-1]
                    d[TOKEN].append(token)
                    d[OFFSET].append(offsets[i])
                    begin = i + 1
            elif all(is_final_mark(c) for c in token):
                sentences.append(sentence(sid, begin, i + 1))
                sid += 1
                begin = i + 1

        if begin < len(input_data):
            sentences.append(sentence(sid, begin, len(input_data)))
            sid += 1

        return sentences
