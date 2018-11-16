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


### Morphological Analysis

| Model ID | Trained | Dependencies |
|----------|:-------:|--------------|
| [elit-morph-lexrule-en](../tools/morphological_analysis.html#english-morphological-analyzer) | - | [pos tagger](#part-of-speech-tagging) trained on any dataset in [mixed](#mixed) |


### Part-of-Speech Tagging

| Model ID | Trained | Dependencies |
|----------|:-------:|--------------|
| elit-pos-en-cnn-mixed   | [Mixed](#mixed) | [tokenizer](#tokenization) |
| elit-pos-en-lstm-mixed  | [Mixed](#mixed) | [tokenizer](#tokenization) |
| elit-pos-en-flair-mixed | [Mixed](#mixed) | [tokenizer](#tokenization) |


### Named Entity Recognition

| Model ID | Trained | Dependencies |
|----------|:-------:|--------------|
| elit-ner-en-cnn-ontonotes   | [OntoNotes](#OntoNotes) | [tokenizer](#tokenization) |
| elit-ner-en-lstm-ontonotes  | [OntoNotes](#OntoNotes) | [tokenizer](#tokenization) |
| elit-ner-en-flair-ontonotes | [OntoNotes](#OntoNotes) | [tokenizer](#tokenization) |


### Dependency Parsing

| Model ID | Trained | Dependencies |
|----------|:-------:|--------------|
| elit-dep-en-biaffine-mixed | [Mixed](#mixed) | [pos tagger](#part-of-speech-tagging) trained on [mixed](#mixed) |


### Coreference Resolution

| Model ID | Trained | Dependencies |
|----------|:-------:|--------------|
| elit-coref-en-e2e-ontonotes | [OntoNotes](#OntoNotes) | [tokenizer](#tokenization) |


### Sentiment Analysis

| Model ID | Trained | Dependencies |
|----------|:-------:|--------------|
| elit-senti-en-cnnatt-sst | [SST](#Stanford-Sentiment-Treebank) | [tokenizer](#tokenization) |


## English Datasets

### OntoNotes


### BOLT

### Mixed

OntoNotes + Medical + BOLT

### Stanford Sentiment Treebank