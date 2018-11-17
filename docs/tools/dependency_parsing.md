# Dependency Parsing

## Biaffine Parser

This is ELIT's replication of the dependency parser using deep biaffine attention introduced by Stanford University.

* Associated models: `elit-dep-biaffine-en-mixed`.
* API reference: [BiaffineDEPParser](../apidocs/dependency_parsers.html#elit.nlp.dep.parser.parser.BiaffineDEPParser).
* Decode parameters: `none`.


### Web-API

```json
{"model": "elit-dep-biaffine-en-mixed"}
```

### Citation

```text
@InProceedings{lee-he-zettlemoyer:2018:N18-2,
  author    = {Dozat, Timothy and Manning, Christopher D.},
  title     = {Deep Biaffine Attention for Neural Dependency Parsing},
  booktitle = {Proceedings of the 5th International Conference on Learning Representations},
  year      = {2017},
  series    = {ICLR'17},
  url       = {https://arxiv.org/abs/1611.01734}
}
```
