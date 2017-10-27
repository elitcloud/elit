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

from elit.util.string_util import is_right_bracket, is_final_mark
from elit.util.structure import KEY_TOKENS, KEY_OFFSETS

__author__ = 'Jinho D. Choi'


class Segmenter(object):
    @abc.abstractmethod
    def decode(self, tokens, offsets):
        """
        :param tokens: the input tokens.
        :type tokens: list of str
        :param offsets: the offsets of the corresponding tokens in the original text.
        :type offsets: list of (int, int)
        :return: the list of sentences, where each sentence is a dictionary containing tokens and offsets as keys.
        """
        return


class EnglishSegmenter(Segmenter):
    def decode(self, tokens, offsets):
        def sentence(begin, end):
            return {KEY_TOKENS: tokens[begin:end], KEY_OFFSETS: offsets[begin:end]}

        sentences = []
        begin = 0
        right_quote = True

        for i, token in enumerate(tokens):
            t = token[0]
            if t == '"': right_quote = not right_quote

            if begin == i:
                if sentences and (is_right_bracket(t) or t == u'\u201D' or t == '"' and right_quote):
                    d = sentences[-1]
                    d[KEY_TOKENS].append(token)
                    d[KEY_OFFSETS].append(offsets[i])
                    begin = i + 1
            elif all(is_final_mark(c) for c in token):
                sentences.append(sentence(begin, i + 1))
                begin = i + 1

        if begin < len(tokens):
            sentences.append(sentence(begin, len(tokens)))

        return sentences