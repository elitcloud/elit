# English Models

## Tokenization

| Model ID | Trained | Dependencies |
|----------|:-------:|:------------:|
| [elit-tok-space-un](../tools/tokenization.html#space-tokenizer)     | - | - |
| [elit-tok-lexrule-en](../tools/tokenization.html#english-tokenizer) | - | - |


## Morphological Analysis

| Model ID | Trained | Dependencies |
|----------|:-------:|--------------|
| [elit-morph-lexrule-en](../tools/morphological_analysis.html#english-analyzer) | - | [POS tagger](#part-of-speech-tagging) trained on any dataset in [Mixed](english_datasets.html#mixed) |


## Part-of-Speech Tagging

| Model ID | Trained | Dependencies |
|----------|:-------:|--------------|
| elit-pos-cnn-en-mixed   | [Mixed](english_datasets.html#mixed) | [Tokenizer](#tokenization) with segmentation |
| elit-pos-lstm-en-mixed  | [Mixed](english_datasets.html#mixed) | [Tokenizer](#tokenization) with segmentation |
| elit-pos-flair-en-mixed | [Mixed](english_datasets.html#mixed) | [Tokenizer](#tokenization) with segmentation |


## Named Entity Recognition

| Model ID | Trained | Dependencies |
|----------|:-------:|--------------|
| elit-ner-cnn-en-ontonotes   | [OntoNotes](english_datasets.html#ontonotes) | [Tokenizer](#tokenization) with segmentation |
| elit-ner-flair-en-ontonotes | [OntoNotes](english_datasets.html#ontonotes) | [Tokenizer](#tokenization) with segmentation |


## Dependency Parsing

| Model ID | Trained | Dependencies |
|----------|:-------:|--------------|
| elit-dep-biaffine-en-mixed | [Mixed](english_datasets.html#mixed) | [POS tagger](#part-of-speech-tagging) trained on [Mixed](english_datasets.html#mixed) |


## Coreference Resolution

| Model ID | Trained | Dependencies |
|----------|:-------:|--------------|
| uw-coref-e2e-en-ontonotes | [OntoNotes](english_datasets.html#ontonotes) | [Tokenizer](#tokenization) with segmentation |


## Sentiment Analysis

| Model ID | Trained | Dependencies |
|----------|:-------:|--------------|
| elit-senti-embatt-en-sst | [SST](english_datasets.html#stanford-sentiment-treebank) | [Tokenizer](#tokenization) with segmentation |
