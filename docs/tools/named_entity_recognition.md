# Named Entity Recognition

## Output Format

The key `coref` is inserted on the [document](../documentation/output_format.html#document) level, where the value is a list of entity clusters.
The example below shows output for the input text, "_Mr. Johnson bought a truck. He likes it very much!_":


## CNN Tagger

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
```

## Flair Tagger

This is ELIT's replication of [Flair](https://github.com/zalandoresearch/flair/)'s tagger using contextual string embeddings.
It is ported from the PyTorch implementation of Flair version 0.2.

* Source: [https://github.com/elitcloud/flair-tagger](https://github.com/elitcloud/flair-tagger)
* Associated models: `elit-ner-flair-en-mixed`
* Decode parameters: `none` 

### Web-API

```json
{"model": "elit-ner-flair-en-ontonotes"}
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
