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
from elit.lexicon import Word2Vec
from elit.sentiment.sentiment_analyzer import SemEvalSentimentAnalyzer, SSTSentimentAnalyzer
from elit.tokenizer import english_tokenizer

__author__ = 'Bonggun Shin'

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
EMB_FILE = '/Users/jdchoi/Downloads/tmp/w2v/w2v-400-twitter.gnsm'
MODEL_PATH = '/Users/jdchoi/Downloads/tmp/model/sentiment-semeval17-400-v2'

emb_model = Word2Vec(EMB_FILE)
sa = SemEvalSentimentAnalyzer(emb_model, MODEL_PATH)

sentences = ["I feel a little bit tired today, but I am really happy!",
             "Although the rain stopped, I hate this thick cloud in the sky."]

tokenized_sentences = [[token[0] for token in english_tokenizer.tokenize(s, False)] for s in sentences]

y, att, raw_att = sa.decode(tokenized_sentences, attn=True)
print(y)
print(att)
print(raw_att)

EMB_FILE = '/Users/jdchoi/Downloads/tmp/w2v/w2v-400-amazon-review.gnsm'
MODEL_PATH = '/Users/jdchoi/Downloads/tmp/model/sentiment-sst-400-v2'

emb_model = Word2Vec(EMB_FILE)
sa = SSTSentimentAnalyzer(emb_model, MODEL_PATH)

sentences = ["This is a film well worth seeing, talking and singing heads and all.",
             "The sort of movie that gives tastelessness a bad rap.",
             "Marisa Tomei is good, but Just A Kiss is just a mess."]

tokenized_sentences = [[token[0] for token in english_tokenizer.tokenize(s, False)] for s in sentences]

y, att, raw_att = sa.decode(tokenized_sentences, attn=True)
print(y)
print(att)
print(raw_att)
