# Semantic Dependency Parsing

## Biaffine Parser

This is ELIT's replication of the semantic dependency parser using deep biaffine attention introduced by Stanford University.

* Source: [https://github.com/tdozat/Parser-v3](https://github.com/tdozat/Parser-v3)
* Associated models: `elit_sdp_biaffine_en_mixed`
* API reference: [SDPBiaffineParser](../documentation/apidocs.html#elit.component.sem.sdp_parser.SDPBiaffineParser)
* Decode parameters: `none`

### Web-API

```json
{"model": "elit_sdp_biaffine_en_mixed"}
```

### Python API

```python
from elit.structure import Document, Sentence, TOK, POS
from elit.component import SDPBiaffineParser

tokens = ['John', 'who', 'I', 'wanted', 'to', 'meet', 'was', 'smart']
postags = ['NNP', 'IN', 'WP', 'PRP', 'VBD', 'DT', 'NN', 'VBD', 'JJ'] 
doc = Document()
doc.add_sentence(Sentence({TOK: tokens, POS: postags}))

sdp = SDPBiaffineParser()
sdp.decode([doc])
print(doc.sentences[0])
```

### Output

```json
{
  "sid": 0,
  "tok": ["John", "who", "I", "wanted", "to", "meet", "was", "smart"],
  "pos": ["NNP", "WP", "PRP", "VBD", "TO", "VB", "VBD", "JJ"],
  "sdp": [[[7, "nsbj"], [5, 'obj']], [[5, "r-obj"]], [[3, "nsbj"], [5, "nsbj"]], [[0, "relcl"]], [[5, "aux"]], [[3, "comp"]], [[7, "cop"]], [[-1, "root"]]]
}
```

### Citation

```text
@InProceedings{dozat-manning:ACL:2018,
  author    = {Dozat, Timothy and Manning, Christopher D.},
  title     = {Simpler but More Accurate Semantic Dependency Parsing},
  booktitle = {Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics},
  year      = {2018},
  series    = {ACL'18},
  url       = {https://www.aclweb.org/anthology/P18-2077}
}
```
