# ========================================================================
# Copyright 2018 ELIT
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
from typing import Sequence

from elit.nlp.embedding import Embedding
from elit.nlp.language_models.contextual_string_model import ContextualStringModel
from elit.structure import Document

__author__ = "Gary Lai"


class ContextualStringEmbedding(Embedding):
    """
    :class:`ContextualStringEmbedding` is the context-based model proposed by `Akbik et al., 2018 <http://aclweb.org/anthology/C18-1139>`_.
    """

    def __init__(self, model_path: str, detach: bool = True):
        """
        :param model_path: the path to the model file.
        :param detach: if `False`, the gradient will propagate into the language model,
                       which dramatically slows down training and often leads to worse results.
        """
        self.lm = ContextualStringModel.load_language_model(model_path)
        super().__init__(self.lm.embedding_size)

        self.detach = detach
        if detach: self.lm.freeze()
        self.is_forward_lm = self.lm.is_forward_lm

    def embed(self, docs: Sequence[Document], key: str, bucket: bool = True):
        """
        Adds a list of embeddings to each sentence corresponding to its tokens.

        :param docs: a sequence of input documents.
        :param key: the key to each sentence where the list of embeddings is to be added.
        :param bucket: if ``True``, sentences are bucketed for faster decoding.
        """
        # get text sentences
        # TODO: use bucket to improve performance
        text_sentences = [' '.join(sentence.tokens) for doc in docs for sentence in doc]
        longest_character_sequence_in_batch = len(max(text_sentences, key=len))

        # pad strings with whitespaces to longest sentence
        sentences_padded = []
        append_padded_sentence = sentences_padded.append

        end_marker = ' '
        extra_offset = 1
        for sentence_text in text_sentences:
            pad_by = longest_character_sequence_in_batch - len(sentence_text)
            if self.is_forward_lm:
                padded = '\n{}{}{}'.format(sentence_text, end_marker, pad_by * ' ')
                append_padded_sentence(padded)
            else:
                padded = '\n{}{}{}'.format(sentence_text[::-1], end_marker, pad_by * ' ')
                append_padded_sentence(padded)

        # get hidden states from language model
        all_hidden_states_in_lm = self.lm.get_representation(sentences_padded, self.detach)

        # take first or last hidden states from language model as word representation
        i = 0
        for doc in docs:
            for sentence in doc:
                sentence_text = sentence.to_tokenized_string()

                offset_forward = extra_offset
                offset_backward = len(sentence_text) + extra_offset
                embeddings = []
                sentence[key] = embeddings

                # TODO: can this be improved by taking the list of embeddings per sentence instead of creating embeddings per token
                for token in sentence:
                    offset_forward += len(token)

                    if self.is_forward_lm:
                        offset = offset_forward
                    else:
                        offset = offset_backward

                    embedding = all_hidden_states_in_lm[offset, i, :]

                    # if self.tokenized_lm or token.whitespace_after:
                    offset_forward += 1
                    offset_backward -= 1

                    offset_backward -= len(token)
                    embeddings.append(embedding)

                i += 1