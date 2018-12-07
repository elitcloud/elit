# Part-of-Speech Tagging

## Output Format

The key `pos` is inserted to the [sentence](../documentation/output_format.html#sentence), where the value is a list of part-of-speech tags.
The example below shows output for the input text, "_John bought a car_":

```json
{"sens": [
  {
    "tok": ["John", "bought", "a", "car"], 
    "pos": ["NNP", "VBD", "DT", "NN"] 
  }
]}
```

Each part-of-speech tag corresponds to the same index'th token; thus, the number of tags must match the number of tokens for each sentence. 


## CNN Tagger

The CNN Tagger uses _n_-gram convolutions to extract contextual features and 
predicts the part-of-speech tag of each token independently.

* Source: [https://github.com/elitcloud/token_tagger](https://github.com/elitcloud/token_tagger)
* Associated models: `elit_pos_cnn_en_mixed`
* Decode parameters: `none`

### Web-API

```json
{"model": "elit_pos_cnn_en_mixed"}
```

## RNN Tagger

This tagger uses bidirectional LSTM for sequence classification and predicts the part-of-speech tags of all tokens in each sentence.  

* Source: [https://github.com/elitcloud/token_tagger](https://github.com/elitcloud/token_tagger)
* Associated models: `elit_pos_rnn_en_mixed`
* Decode parameters: `none`

### Web-API

```json
{"model": "elit_pos_rnn_en_mixed"}
```

## Flair Tagger

This is ELIT's replication of [Flair](https://github.com/zalandoresearch/flair/)'s tagger using contextual string embeddings.
It is ported from the PyTorch implementation of Flair version 0.2.

* Source: [https://github.com/elitcloud/flair_tagger](https://github.com/elitcloud/flair_tagger)
* Associated models: `elit_pos_flair_en_mixed`
* Decode parameters: `none` 

### Web-API

```json
{"model": "elit_pos_flair_en_mixed"}
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
