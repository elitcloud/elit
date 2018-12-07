# Named Entity Recognition

## Output Format

The key `ner` is inserted to the [sentence](../documentation/output_format.html#sentence), where the value is a list of named entities.
The example below shows output for the input text, "_Mr. Johnson is from Atlanta_":

```json
{"sens": [
  {
    "tok": ["Mr.", "Johnson", "is", "from", "Atlanta"], 
    "ner": [[0, 2, "PERSON"], [4, 5, "GPE"]] 
  }
]}
```

Each named entity is represented by a tuple of the following three fields:

* `begin_token_id`: the ID of the first token in the entity (inclusive).
* `end_token_id`: the ID of the last token in the entity (exclusive).
* `entity_label`: the label of the entity.

All IDs start with 0 such that `[0, 2, "PERSON"]` in the above example indicates an entity that begins with the `0`th token ("_Mr._") and ends before the `2`nd token ("_is_").
Thus, two entities are found in this sentence, "_Mr. Johnson_" as `PERSON` and "_Altanta_" as `GPE` (Geo-Political Entity).


## CNN Tagger

The CNN Tagger uses _n_-gram convolutions to extract contextual features and 
predicts the part-of-speech tag of each token independently.

* Source: [https://github.com/elitcloud/token_tagger](https://github.com/elitcloud/token_tagger)
* Associated models: `elit_ner_cnn_en_ontonotes`
* Decode parameters: `none`

### Web-API

```json
{"model": "elit_ner_cnn_en_ontonotes"}
```

## RNN Tagger

This tagger uses bidirectional LSTM for sequence classification and predicts the part-of-speech tags of all tokens in each sentence.  

* Source: [https://github.com/elitcloud/token_tagger](https://github.com/elitcloud/token_tagger)
* Associated models: `elit_ner_rnn_en_ontonotes`
* Decode parameters: `none`

### Web-API

```json
{"model": "elit_ner_rnn_en_ontonotes"}
```

## Flair Tagger

This is ELIT's replication of [Flair](https://github.com/zalandoresearch/flair/)'s tagger using contextual string embeddings.
It is ported from the PyTorch implementation of Flair version 0.2.

* Source: [https://github.com/elitcloud/flair_tagger](https://github.com/elitcloud/flair_tagger)
* Associated models: `elit_ner_flair_en_ontonotes`
* Decode parameters: `none` 

### Web-API

```json
{"model": "elit_ner_flair_en_ontonotes"}
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
