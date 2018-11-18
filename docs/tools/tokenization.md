# Tokenization

A tokenizer takes raw text and splits it into string tokens.
It also returns the begin (inclusive) and the end (exclusive) character offsets from the original text for each token.
ELIT's tokenizers provide an option of performing several types of sentence segmentation, which groups chunks of consecutive tokens into sentences:

* `0`: no segmentation.
* `1`: segment by newlines (`\n`).
* `2`: segment by [symbol rules](../apidocs/tokenizers.html#elit.nlp.tokenizer.Tokenizer.segment).
* `3`: segment by `1` and `2`.


## Space Tokenizer

The Space Tokenizer splits input text by whitespaces, which is useful when the input text is already tokenized (either manually or by some other tool) such that no further tokenization is necessary.

* Associated models: `elit-tok-space-un`
* API reference: [SpaceTokenizer](../apidocs/tokenizers.html#elit.nlp.tokenizer.SpaceTokenizer)
* Decode parameters:
  * `segment`: `0`, `1` (_default_), `2`, or `3`

### Web-API

```json
{"model": "elit-tok-whitespace-un", "args": {"segment": 1}}
```

### Python API

```python
from elit.nlp.tokenizer import SpaceTokenizer
tok = SpaceTokenizer()
text = 'John bought a car\nMary sold a truck .'
print(tok.decode(text, segment=1))  # segment by newlines (default)
```

### Output

```json
{"sens": [
  {
    "sid": 0,
    "tok": ["John", "bought", "a", "car"], 
    "off": [[0, 4], [5, 11], [12, 13], [14, 17]] 
  }, 
  {
    "sid": 1,
    "tok": ["Mary", "sold", "a", "truck"], 
    "off": [[18, 22], [23, 27], [28, 29], [30, 35]] 
   }
]}
```


## English Tokenizer

The English Tokenizer splits input text into linguistic tokens using lexicalized rules.

* Associated models: `elit-tok-lexrule-en`
* API reference: [EnglishTokenizer](../apidocs/tokenizers.html#elit.nlp.tokenizer.EnglishTokenizer)
* Decode parameters:
  * `segment`: `0`, `1`, `2` (_default_), or `3`

The followings show key features of this tokenizer:

| Feature | Input Text | Tokens |
|---------|------------|--------|
| Email addresses | `Email (support@elit.cloud)`                    | [`Email`, `(`, `support@elit.cloud`, `)`] |
| Hyperlinks      | `URL: https://elit.cloud`                       | [`URL`, `:`, `https://elit.cloud`] |
| Emoticons       | `I love ELIT :-)!?.`                            | [`I`, `love`, `ELIT`, `:-)`, `!?.`] |
| Hashtags        | `ELIT is the #1 platform #elit2018.`            | [`ELIT`, `is`, `the`, `#`, `1`, `platform`, `#elit2018`, `.`] |
| HTML entities   | `A&larr;B`                                      | [`A`, `&larr;`, `B`] |
| Hyphens         | `(123) 456-7890, 123-456-7890, 2014-2018`       | [`(123)`, `456-7890`, `,`, `123-456-7890`, `,`, `2014`, `-`, `2018`] |
| List items      | `(A)First (A.1)Second [2a]Third [Forth]`        | [`(A)`, `First`, `(A.1)`, `Second`, `[2a]`, `Third`, `[`, `Forth`, `]`] |
| Units           | `$1,000 20mg 100cm 11:00a.m. 10:30PM`           | [`$`, `1,000`, `20`, `mg`, `100`, `cm`, `11:00`, `a.m.`, `10:30`, `PM`] |
| Acronyms        | `I'm gonna miss Dr. Choi 'cause he isn't here.` | [`I`, `'m`, `gon`, `na`, `miss`, `Dr.`, `Choi`, `'cause`, `he`, `is`, `n't`, `here`, `.`] |


### Web-API

```json
{"model": "elit-tok-lexrule-en", "args": {"segment": 2}}
```

### Python API

```python
from elit.nlp.tokenizer import EnglishTokenizer, SpaceTokenizer
tok = EnglishTokenizer()
text = "Mr. Johnson doesn't like cats! What's his favorite then? He likes puffy-dogs."
print(tok.decode(text, segment=2))  # segment by symbol rules (default)
```

### Output

```json
{"sens": [
  {
    "sid": 0,
    "tok": ["Mr.", "Johnson", "does", "n't", "like", "cats", "!"], 
    "off": [[0, 3], [4, 11], [12, 16], [16, 19], [20, 24], [25, 29], [29, 30]], 
  }, 
  {
    "sid": 1,
    "tok": ["What", "'s", "his", "favorite", "then", "?"], 
    "off": [[31, 35], [35, 37], [38, 41], [42, 50], [51, 55], [55, 56]] 
  }, 
  {
    "sid": 2,
    "tok": ["He", "likes", "puffy", "-", "dogs", "."], 
    "off": [[57, 59], [60, 65], [66, 71], [71, 72], [72, 76], [76, 77]] 
  }
]}
```

