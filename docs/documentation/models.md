# Available Models

All models take the following naming convention:

```text
[team]-[task]-[method]-[language]-[training_data]*
```

* `team`: the team who developed this model.
* `task`: the task that this model is developed for.
* `method`: the method used to develop this model.
* `language`: the input language ([ISO 639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes); use `un` for universal models).
* `training_data`: the training data used to build this model (if applicable).

The 'dependencies' column in each table indicates tools that need to be run prior to this model.

## English Models

### Tokenization

| Model ID | Trained | Dependencies |
|----------|:-------:|:------------:|
| [elit-tok-space-un](../tools/tokenizers.html#space-tokenizer)     | - | - |
| [elit-tok-lexrule-en](../tools/tokenizers.html#english-tokenizer) | - | - |


### Part-of-Speech Tagging

| Model ID | Trained | Dependencies |
|----------|:-------:|--------------|
| elit-pos-en-cnn-mixed   | [Mixed](datasets.html#mixed) | [Tokenizer](#tokenization) with segmentation |
| elit-pos-en-lstm-mixed  | [Mixed](datasets.html#mixed) | [Tokenizer](#tokenization) with segmentation |
| elit-pos-en-flair-mixed | [Mixed](datasets.html#mixed) | [Tokenizer](#tokenization) with segmentation |


### Named Entity Recognition

| Model ID | Trained | Dependencies |
|----------|:-------:|--------------|
| elit-ner-en-cnn-ontonotes   | [OntoNotes](datasets.html#ontonotes) | [Tokenizer](#tokenization) with segmentation |
| elit-ner-en-lstm-ontonotes  | [OntoNotes](datasets.html#ontonotes) | [Tokenizer](#tokenization) with segmentation |
| elit-ner-en-flair-ontonotes | [OntoNotes](datasets.html#ontonotes) | [Tokenizer](#tokenization) with segmentation |


### Morphological Analysis

| Model ID | Trained | Dependencies |
|----------|:-------:|--------------|
| [elit-morph-lexrule-en](../tools/morphological_analysis.html#english-morphological-analyzer) | - | [POS tagger](#part-of-speech-tagging) trained on any dataset in [Mixed](datasets.html#mixed) |


### Dependency Parsing

| Model ID | Trained | Dependencies |
|----------|:-------:|--------------|
| elit-dep-en-biaffine-mixed | [Mixed](datasets.html#mixed) | [POS tagger](#part-of-speech-tagging) trained on [Mixed](datasets.html#mixed) |


### Coreference Resolution

| Model ID | Trained | Dependencies |
|----------|:-------:|--------------|
| elit-coref-en-e2e-ontonotes | [OntoNotes](datasets.html#ontonotes) | [Tokenizer](#tokenization) with segmentation |


### Sentiment Analysis

| Model ID | Trained | Dependencies |
|----------|:-------:|--------------|
| elit-senti-en-cnnatt-sst | [SST](datasets.html#stanford-sentiment-treebank) | [Tokenizer](#tokenization) with segmentation |
