# Dependency Parsing

## Biaffine Parser

This is ELIT's replication of the dependency parser using deep biaffine attention introduced by Stanford University.

* Source: [https://github.com/tdozat/Parser-v1](https://github.com/tdozat/Parser-v1)
* Associated models: `elit_dep_biaffine_en_mixed`
* API reference: [DEPBiaffineParser](../documentation/apidocs.html#elit.component.dep.dependency_parser. DEPBiaffineParser)
* Decode parameters: `none`

### Web-API

```json
{"model": "elit_dep_biaffine_en_mixed"}
```

### Python API

```python
from elit.structure import Document, Sentence, TOK, POS
from elit.component import DEPBiaffineParser

tokens = ['John', 'who', 'I', 'wanted', 'to', 'meet', 'was', 'smart']
postags = ['NNP', 'IN', 'WP', 'PRP', 'VBD', 'DT', 'NN', 'VBD', 'JJ'] 
doc = Document()
doc.add_sentence(Sentence({TOK: tokens, POS: postags}))

dep = DEPBiaffineParser()
dep.decode([doc])
print(doc.sentences[0])
```

### Output

```json
{
  "sid": 0,
  "tok": ["John", "who", "I", "wanted", "to", "meet", "was", "smart"],
  "pos": ["NNP", "WP", "PRP", "VBD", "TO", "VB", "VBD", "JJ"],
  "dep": [[7, "nsbj"], [5, "r-obj"], [3, "nsbj"], [0, "relcl"], [5, "aux"], [3, "comp"], [7, "cop"], [-1, "root"]]
}
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
