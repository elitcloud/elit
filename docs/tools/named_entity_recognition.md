# Named Entity Recognition

## Flair Tagger

This is ELIT's replication of [Flair](https://github.com/zalandoresearch/flair/)'s tagger using contextual string embeddings.
It is ported from the PyTorch implementation of Flair version 0.2.

* Source: [https://github.com/zalandoresearch/flair](https://github.com/zalandoresearch/flair)
* Associated models: `elit_ner_flair_en_ontonotes`
* API reference: [NERFlairTagger](../documentation/apidocs.html#elit.component.tagger.ner_tagger.NERFlairTagger)
* Decode parameters: `none` 

### Web API

```json
{"model": "elit_ner_flair_en_ontonotes"}
```

### Python API

```python
from elit.structure import Document, Sentence, TOK
from elit.component import NERFlairTagger

tokens = ['Jinho', 'Choi', 'is', 'a', 'professor', 'at', 'Emory', 'University', 'in', 'Atlanta', ',', 'Georgia', '.']
doc = Document()
doc.add_sentence(Sentence({TOK: tokens}))

ner = NERFlairTagger()
ner.decode([doc])
print(doc.sentences[0])
```

### Output

```json
{
  "sid": 0,
  "tok": ["Jinho", "Choi", "is", "a", "professor", "at", "Emory", "University", "in", "Atlanta", ",", "Georgia", "."], 
  "ner": [[0, 2, "PERSON"], [6, 8, "ORG"], [9, 12, "LOC"]]
}
```

### Citation

```text
@InProceedings{akbik-blythe-vollgraf:COLING:2018,
  author    = {Akbik, Alan and Blythe, Duncan and Vollgraf, Roland},
  title     = {Contextual String Embeddings for Sequence Labeling},
  booktitle = {Proceedings of the 27th International Conference on Computational Linguistics},
  year      = {2018},
  series    = {COLING'18},
  url       = {http://aclweb.org/anthology/C18-1139}
}
```




<!--## CNN Tagger

The CNN Tagger uses _n_-gram convolutions to extract contextual features and 
predicts the part-of-speech tag of each token independently.

* Source: [https://github.com/elitcloud/elit-token-tagger](https://github.com/elitcloud/elit-token-tagger)
* Associated models: `elit-ner-cnn-en-mixed`
* Decode parameters: `none`

### Web-API

```json
{"model": "elit-ner-cnn-en-ontonotes"}
```

## RNN Tagger

This tagger uses bidirectional LSTM for sequence classification and predicts the part-of-speech tags of all tokens in each sentence.  

* Source: [https://github.com/elitcloud/elit-token-tagger](https://github.com/elitcloud/elit-token-tagger)
* Associated models: `elit-ner-rnn-en-mixed`
* Decode parameters: `none`

### Web-API

```json
{"model": "elit-ner-rnn-en-ontonotes"}
```-->
