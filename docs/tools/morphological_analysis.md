# Morphological Analysis

## English Analyzer

The English Analyzer takes an input token and its part-of-speech tag in the [Penn Treebank style](../documentation/english_datasets.html#mixed), 
and splits it into morphemes using [inflection](https://en.wikipedia.org/wiki/Inflection), [derivation](https://en.wikipedia.org/wiki/Morphological_derivation), and [prefix](https://en.wikipedia.org/wiki/Prefix) rules.

* Associated models: `elit-morph-idprule-en`
* API reference: [EnglishMorphAnalyzer](../documentation/apidocs.html#elit.component.morph_analyzer.EnglishMorphAnalyzer)
* [Supplementary documentation](supplementary/english_morph_analyzer.html)
* Decode parameters:
  * `derivation`: `True` (_default_) or `False`
  * `prefix`: `0` (no prefix analysis; _default_), `1` (shortest preferred), `2` (longest preferred)
  

### Web API

```json
{"model": "elit_morph_lexrule_en", "args": {"derivation": true, "prefix": 0}}
```

### Python API

```python
from elit.structure import Document, Sentence, TOK, POS, MORPH
from elit.component import EnglishMorphAnalyzer

tokens = ['dramatized', 'ownerships', 'environmentalists', 'certifiable', 'realistically']
postags = ['VBD', 'NNS', 'NNS', 'JJ', 'RB']
doc = Document()
doc.add_sentence(Sentence({TOK: tokens, POS: postags}))

morph = EnglishMorphAnalyzer()
morph.decode([doc], derivation=True, prefix=0)
print(doc.sentences[0][MORPH])
```

### Output

```json
[
  [["drama", "NN"], ["+tic", "J_IC"], ["+ize", "V_IZE"], ["+d", "I_PST"]], 
  [["own", "VB"], ["+er", "N_ER"], ["+ship", "N_SHIP"], ["+s", "I_PLR"]], 
  [["environ", "VB"], ["+ment", "N_MENT"], ["+al", "J_AL"], ["+ist", "N_IST"], ["+s", "I_PLR"]], 
  [["cert", "NN"], ["+ify", "V_FY"], ["+iable", "J_ABLE"]], 
  [["real", "NN"], ["+ize", "V_IZE"], ["+stic", "J_IC"], ["+ally", "R_LY"]]
]
```
