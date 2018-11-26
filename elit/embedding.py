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
import inspect
import logging
from typing import List, Sequence, Union

import abc
import fastText
import numpy as np
import mxnet as mx
from gensim.models import KeyedVectors

from elit.nlp.language_models.contextual_string_model import ContextualStringModel
from elit.nlp.tagger.mxnet_util import mxnet_prefer_gpu
from elit.structure import Document

__author__ = 'Jinho D. Choi'


class Embedding(abc.ABC):
    """
    :class:`Embedding` is an abstract class to implement embedding models.

    :ivar dim: the dimension of each embedding.
    :ivar pad: the zero vector whose dimension is the same as the embedding (can be used for zero-padding).

    Abstract methods to be implemented:
      - :meth:`Embedding.embed`
    """
    def __init__(self, dim: int):
        """
        :param dim: the dimension of each embedding.
        """
        self.dim = dim
        self.pad = np.zeros(dim).astype('float32')

    @abc.abstractmethod
    def embed(self, docs: Sequence[Document], key: str, **kwargs):
        """
        Adds embeddings to the input documents.

        :param docs: a sequence of input documents.
        :param key: the key to a sentence or a document where embeddings are to be added.
        """
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))


class TokenEmbedding(Embedding):
    """
    :class:`TokenEmbedding` is an abstract class to implement token-based embedding models.

    Subclasses:
      - :class:`Word2Vec`
      - :class:`FastText`

    Abstract methods to be implemented:
      - :meth:`TokenEmbedding.emb`
    """

    def __init__(self, dim: int):
        """
        :param dim: the dimension of each embedding.
        """
        super().__init__(dim)

    # override
    def embed(self, docs: Sequence[Document], key: str, **kwargs):
        """
        Adds a list of embeddings to each sentence corresponding to its tokens.

        :param docs: a sequence of input documents.
        :param key: the key to each sentence where the list of embeddings is to be added.
        """
        for doc in docs:
            for sen in doc:
                sen[key] = self.emb_list(sen.tokens)

    @abc.abstractmethod
    def emb(self, token: str) -> np.ndarray:
        """
        :param token: the input token.
        :return: the embedding of the input token.
        """
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))

    def emb_list(self, tokens: Sequence[str]) -> List[np.ndarray]:
        """
        :param tokens: the sequence of input tokens.
        :return: the list of embeddings for the corresponding tokens.
        """
        return [self.emb(value) for value in tokens]

    def emb_matrix(self, tokens: Sequence[str], maxlen: int) -> List[np.ndarray]:
        """
        :param tokens: the sequence of input tokens.
        :param maxlen: the maximum length of the output list;
                       if ``> len(values)``, the bottom part of the matrix is padded with zero embeddings;
                       if ``< len(values)``, embeddings of the exceeding values are discarded from the resulting matrix.
        :return: the matrix where each row is the embedding of the corresponding value.
        """
        return [self.emb(tokens[i]) if i < len(tokens) else self.pad for i in range(maxlen)]


class FastText(TokenEmbedding):
    """
    :class:`FastText` is a token-based model trained by `FastText <https://github.com/facebookresearch/fastText>`_.
    """

    def __init__(self, filepath: str):
        """
        :param filepath: the path to the binary file containing a word embedding model trained by FastText (``*.bin``).
        """
        logging.info('FastText')
        logging.info('- model: {}'.format(filepath))
        self.model = fastText.load_model(filepath)
        dim = self.model.get_dimension()
        super().__init__(dim)
        logging.info('- vocab = %d, dim = %d' % (len(self.model.get_words()), dim))

    # override
    def emb(self, value: str) -> np.ndarray:
        return self.model.get_word_vector(value)


class Word2Vec(TokenEmbedding):
    """
    :class:`Word2Vec` is a token-based model trained by `Word2Vec <https://github.com/tmikolov/word2vec>`_.
    """

    def __init__(self, filepath: str):
        """
        :param filepath: the path to the binary file containing a word embedding model trained by Word2Vec (``*.bin``).
        """
        logging.info('Word2Vec')
        logging.info('- model: {}'.format(filepath))
        self.model = KeyedVectors.load(filepath) if filepath.lower().endswith('.gnsm') else KeyedVectors.load_word2vec_format(filepath, binary=True)
        dim = self.model.syn0.shape[1]
        super().__init__(dim)
        logging.info('- vocab = %d, dim = %d' % (len(self.model.vocab), dim))

    # override
    def emb(self, value: str) -> np.ndarray:
        vocab = self.model.vocab.get(value, None)
        return self.model.syn0[vocab.index] if vocab else self.pad


class ContextualStringEmbedding(Embedding):
    """
    :class:`ContextualStringEmbedding` is the context-based model proposed by `Akbik et al., 2018 <http://aclweb.org/anthology/C18-1139>`_.
    """

    def __init__(self, model_path: str, detach: bool = True, context: mx.Context = None):
        """
        :param model_path: the path to the model file.
        :param detach: if `False`, the gradient will propagate into the language model,
                       which dramatically slows down training and often leads to worse results.
        """
        self.context = context if context else mxnet_prefer_gpu()
        self.lm = ContextualStringModel.load_language_model(model_path, context)
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


def init_emb(config: list) -> Union[Word2Vec, FastText]:
    model, path = config
    if model.lower() == 'word2vec':
        emb = Word2Vec
    elif model.lower() == 'fasttext':
        emb = FastText
    else:
        raise TypeError('model {} is not supported'.format(model))
    return emb(path)
