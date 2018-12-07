# Output Format

## Document

A [document](../apidocs/structures.html#elit.structure.Document) is represented by a dictionary that consists of the following fields:

* `sens`: a list of [sentences](#sentence) in the document.
* `coref`: a list of [entity clusters](../tools/coreference_resolution.html#output-format) from coreference resolution.

```json
{
  "sens": [],
  "coref": []
}
```

## Sentence

A [sentence](../apidocs/structures.html#elit.structure.Sentence) is represented by a dictionary that consists of the following fields:

* `sid`: the sentence ID (starts with 0).
* `tok`: a list of [tokens](../tools/tokenization.html#output-format) in the sentence.
* `off`: a list of [character offsets](../tools/tokenization.html#output-format) of the tokens.
* `pos`: a list of [part-of-speech tags](../tools/part_of_speech_tagging.html#output-format) of the tokens.
* `ner`: a list of [named entities](../tools/named_entity_recognition.html#output-format) in the sentence.
* `dep`: a list of [dependency relations](../tools/dependency_parsing.html#output-format) of the tokens.
* `morph`: a list of [morpheme sets](../tools/morphological_analysis.html#output-format) of the tokens.

```json
{
  "sid": 0,
  "tok": [],
  "off": [],
  "pos": [],
  "ner": [],
  "dep": [],
  "morph": []
}
```