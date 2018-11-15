# Available Tools

The 'dependencies' column indicates tools that need to be run prior to the corresponding model.

## English Models

### Tokenization

| Model ID | Trained | Dependencies |
|----------|:-------:|:------------:|
| elit-tok-whitespace | - | - |
| elit-tok-en         | - | - |


### Morphological Analysis

| Model ID | Trained | Dependencies |
|----------|:-------:|--------------|
| elit-morph-en | - | Any [pos tagger](#part-of-speech-tagging) trained on dataset(s) in [Mixed](#mixed) | |


### Part-of-Speech Tagging

| Model ID | Trained | Dependencies |
|----------|:-------:|--------------|
| elit-pos-en-cnn-mixed   | [Mixed](#mixed) | Any [tokenizer](#tokenization) |
| elit-pos-en-lstm-mixed  | [Mixed](#mixed) | Any [tokenizer](#tokenization) |
| elit-pos-en-flair-mixed | [Mixed](#mixed) | Any [tokenizer](#tokenization) |


### Named Entity Recognition

| Model ID | Trained | Dependencies |
|----------|:-------:|--------------|
| elit-ner-en-cnn-ontonotes   | [OntoNotes](#OntoNotes) | Any [tokenizer](#tokenization) |
| elit-ner-en-lstm-ontonotes  | [OntoNotes](#OntoNotes) | Any [tokenizer](#tokenization) |
| elit-ner-en-flair-ontonotes | [OntoNotes](#OntoNotes) | Any [tokenizer](#tokenization) |


### Dependency Parsing

| Model ID | Trained | Dependencies |
|----------|:-------:|--------------|
| elit-dep-en-biaffine-mixed | [Mixed](#mixed) | Any [pos tagger](#part-of-speech-tagging) trained on [Mixed](#mixed) |


### Coreference Resolution

| Model ID | Trained | Dependencies |
|----------|:-------:|--------------|
| elit-coref-en-e2e-ontonotes | [OntoNotes](#OntoNotes) | Any [tokenizer](#tokenization) |


### Sentiment Analysis

| Model ID | Trained | Dependencies |
|----------|:-------:|--------------|
| elit-senti-en-cnnatt-sst | [SST](#Stanford-Sentiment-Treebank) | Any [tokenizer](#tokenization) |


## English Datasets

### OntoNotes


### BOLT

### Mixed

OntoNotes + Medical + BOLT

### Stanford Sentiment Treebank