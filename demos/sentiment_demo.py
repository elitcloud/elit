
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
from elit.nlp.task.sentiment import TwitterSentimentAnalyzer, MovieSentimentAnalyzer
from elit.nlp.task.tokenize import EnglishTokenizer
from elit.nlp.structure import TOKEN
from elit.nlp.lexicon import Word2VecTmp

__author__ = 'Bonggun Shin, Jinho D. Choi'


def run(analyzer, emb_file, model_file, text):
    emb_model = Word2VecTmp(emb_file)
    sentiment = analyzer(emb_model, model_file)
    tokenizer = EnglishTokenizer('../../../resources/tokenize')

    tokens, offsets = tokenizer.decode(text)[:2]
    sentences = [d[TOKEN] for d in tokenizer.segment(tokens, offsets)]
    y, att = sentiment.decode(sentences, att=True)[:2]

    print(y)
    print(att)


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    emb_file = '../../../resources/embedding/w2v-400-twitter.gnsm'
    model_file = '../../../resources/sentiment/sentiment-semeval17-400-v2'
    text = 'I feel a little bit tired today, but I am really happy! Although the rain stopped, I hate this thick cloud in the sky.'
    run(TwitterSentimentAnalyzer, emb_file, model_file, text)

    emb_file = '../../../resources/embedding/w2v-400-amazon-review.gnsm'
    model_file = '../../../resources/sentiment/sentiment-sst-400-v2'
    text = 'This is a film well worth seeing, talking and singing heads and all. The sort of movie that gives tastelessness a bad rap. Marisa Tomei is good, but Just A Kiss is just a mess.'
    run(MovieSentimentAnalyzer, emb_file, model_file, text)
