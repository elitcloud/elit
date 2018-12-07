# Dependency Parsing

## Output Format

The key `dep` is inserted to the [sentence](../documentation/output_format.html#sentence), where the value is a list of dependency relations.
The example below shows output for the input text, "_John bought a car_":

```json
{"sens": [
  {
    "tok": ["John", "bought", "a", "car"], 
    "pos": [[1, "nsbj"], [-1, "root"], [3, "det"], [1, "obj"]] 
  }
]}
```

Each relation shows the dependency of the same index'th token to its head and is represented by a tuple of the following two fields:

* `head_id`: the token ID of the head.
* `deprel`: the dependency label.

Token IDs start with 0 such that `[1, "nsbj"]` in the above example indicates that the 0th token ("_John_") is `nsbj` (nominal subject) of the `1`st token ("_bought_" ).
A root is indicated by the ID `-1` such that `[-1, "root"]` implies that "_bought_" is the `root` that does not depend on any token.
For each sentence, the number of dependency relations must match the number of tokens so that each token is a dependent of exactly one other token, which makes the output structure to be a tree.


## Biaffine Parser

This is ELIT's replication of the dependency parser using deep biaffine attention introduced by Stanford University.

* Source: [https://github.com/elitcloud/biaffine-parser](https://github.com/elitcloud/biaffine-parser)
* Associated models: `elit_dep_biaffine_en_mixed`
* Decode parameters: `none`

### Web-API

```json
{"model": "elit_dep_biaffine_en_mixed"}
```

### Citation

```text
@InProceedings{dozat-manning:ICLR:2017,
  author    = {Dozat, Timothy and Manning, Christopher D.},
  title     = {Deep Biaffine Attention for Neural Dependency Parsing},
  booktitle = {Proceedings of the 5th International Conference on Learning Representations},
  year      = {2017},
  series    = {ICLR'17},
  url       = {https://arxiv.org/abs/1611.01734}
}
```
