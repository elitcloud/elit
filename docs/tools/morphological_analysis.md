# Morphological Analysis

## Output Format

The key `morph` is inserted to the [sentence](../documentation/output_format.html#sentence), where the value is a list of morpheme sets.
The example below shows output for the input text, "_John bought a car_":

```json
{"sens": [
  {
    "tok": ["John", "studied", "environmental", "science"], 
    "pos": [[["john", "NN"]],
            [["study", "VB"], ["+ied", "I_PST"]],
            [["environ", "VB"], ["+ment", "N_MENT"], ["+al", "J_AL"]], 
            [["science", "NN"]]] 
  }
]}
```

Each morpheme set corresponds to the same index'th token; thus, the number of sets must match the number of tokens for each sentence.
The morpheme set is represented by a list of morphemes, where each morpheme is represented by a tuple of the following two fields:

* `morph_form`: the form of the morpheme.
* `morph_tag`: the tag of the morpheme.

Thus, `[["study", "VB"], ["+ied", "I_PST"]]` in the above example indicates the morpheme set of "_studied_", analyzed by the derivation rule for the past tense (`I_PST`).

## English Analyzer

The English Analyzer takes an input token and its part-of-speech tag in the [Penn Treebank style](../documentation/english_datasets.html#mixed), 
and splits it into morphemes using [inflection](https://en.wikipedia.org/wiki/Inflection), [derivation](https://en.wikipedia.org/wiki/Morphological_derivation), and [prefix](https://en.wikipedia.org/wiki/Prefix) rules.

* Source: [https://github.com/elitcloud/morphological_analyzer](https://github.com/elitcloud/morphological_analyzer)
* Associated models: `elit_morph_lexrule_en`
* [Supplementary documentation](supplementary/english_morph_analyzer.html)
* Decode parameters:
  * `derivation`: `True` (_default_) or `False`
  * `prefix`: `0` (no prefix analysis; _default_), `1` (shortest preferred), `2` (longest preferred)
  

### Web-API

```json
{"model": "elit_morph_lexrule_en", "args": {"derivation": true, "prefix": 0}}
```

### Python API

```python
from elit.structure import Document, Sentence, TOK, POS, MORPH
from elit.nlp.morph_analyzer import EnglishMorphAnalyzer

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
