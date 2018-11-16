# Tokenizers

A tokenizer takes raw text and splits it into morphologically meaningful tokens.
It also returns the begin (inclusive) and end (exclusive) character offsets of each token from the original text.
ELIT's tokenizers provide an option of performing several types of sentence segmentation, which groups chunks of consecutive tokens into sentences:

* `0`: no segmentation.
* `1`: segment by newlines (`\n`).
* `2`: segment by [puctuation rules](../_modules/elit/tokenizer.html#Tokenizer.segment).
* `3`: segment by `1` and `2`.


## Whitespace Tokenizer

This tokenizer splits input text by only whitespaces, which is useful when the input text is already tokenized (either manually or by some other tool) such that no further tokenization is necessary.

* Model ID: `elit-tok-whitespace-un`.
* Supported segmentation: `0`, `1`, `2` (default), `3`.

### Web-API

The type of [segmentation](#segmentation) can be configured:

```python
models = [{'model': 'elit-tok-whitespace-un', 'args': {'segment': 1}}]  # segment by newlines
```

### API

```python
from elit.tools import SpaceTokenizer
text = 'John bought a car .\nMary sold a truck .'
tok = SpaceTokenizer()
print(tok.decode(text, segment=1))
```

```json
{'sen': [
  {
     'sen_id': 0,
     'tok': ['John', 'bought', 'a', 'car', '.'],
     'off': [(0, 4), (5, 11), (12, 13), (14, 17), (18, 19)]}, 
   {
     'sen_id': 1,
     'tok': ['Mary', 'sold', 'a', 'truck', '.'],
     'off': [(20, 24), (25, 29), (30, 31), (32, 37), (38, 39)]
   }
]}
```

## English Tokenizer

* Model ID: `elit-tok-lexrule-en`

```python
from elit.tools import EnglishTokenizer
text = "Mr. Johnson doesn't like cats! What's his favorite then? He likes puffy-dogs."
tok = EnglishTokenizer()
print(tok.decode(text, segment=2))
```

```json
{'sen': [
  {
    'sen_id': 0,
    'tok': ['Mr.', 'Johnson', 'does', "n't", 'like', 'cats', '!'], 
    'off': [(0, 3), (4, 11), (12, 16), (16, 19), (20, 24), (25, 29), (29, 30)], 
  }, 
  {
    'sen_id': 1,
    'tok': ['What', "'s", 'his', 'favorite', 'then', '?'], 
    'off': [(31, 35), (35, 37), (38, 41), (42, 50), (51, 55), (55, 56)] 
  }, 
  {
    'sen_id': 2,
    'tok': ['He', 'likes', 'puffy', '-', 'dogs', '.'], 
    'off': [(57, 59), (60, 65), (66, 71), (71, 72), (72, 76), (76, 77)] 
  }
]}
```

