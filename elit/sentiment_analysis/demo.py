from elit.tokenizer import english_tokenizer
from elit.sentiment_analysis.decode import SentimentAnalysis

sentences = ["I feel a little bit tired today, but I am really happy!",
             "Although the rain stopped, I hate this thick cloud in the sky."]

tokenized_sentences = []
for s in sentences:
    tokenized_sentences.append(english_tokenizer.tokenize(s, False))

sa = SentimentAnalysis()
y, att = sa.decode(tokenized_sentences)

print(y)
print(att[0][0])

