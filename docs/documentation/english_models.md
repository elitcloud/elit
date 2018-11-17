# English Models

* PRE: indicates tools that need to be run prior to the corresponding model.
* tokseg: any [tokenizer](#tokenization) with sentence segmentation.
* inmixed: any dataset in the [mixed](english_datasets.html#mixed) corpus.

## Tokenization

| Model ID | Trained | PRE |
|----------|:-------:|:------------:|
| [elit-tok-space-un](../tools/tokenization.html#space-tokenizer)     | - | - |
| [elit-tok-lexrule-en](../tools/tokenization.html#english-tokenizer) | - | - |


## Morphological Analysis

| Model ID | Trained | PRE |
|----------|:-------:|:------------:|
| [elit-morph-idprule-en](../tools/morphological_analysis.html#english-analyzer) | - | [&#10035;-pos-&#10035;-en-inmixed](#part-of-speech-tagging) |


## Part-of-Speech Tagging

| Model ID | Trained | PRE |
|----------|:-------:|:------------:|
| elit-pos-cnn-en-mixed   | [Mixed](english_datasets.html#mixed) | [tokseg](#tokenization) |
| elit-pos-lstm-en-mixed  | [Mixed](english_datasets.html#mixed) | [tokseg](#tokenization) |
| elit-pos-flair-en-mixed | [Mixed](english_datasets.html#mixed) | [tokseg](#tokenization) |


## Named Entity Recognition

| Model ID | Trained | PRE |
|----------|:-------:|:------------:|
| elit-ner-cnn-en-ontonotes   | [OntoNotes](english_datasets.html#ontonotes) | [tokseg](#tokenization) |
| elit-ner-flair-en-ontonotes | [OntoNotes](english_datasets.html#ontonotes) | [tokseg](#tokenization) |


## Dependency Parsing

| Model ID | Trained | PRE |
|----------|:-------:|:------------:|
| elit-dep-biaffine-en-mixed | [Mixed](english_datasets.html#mixed) | [&#10035;-pos-&#10035;-en-inmixed](#part-of-speech-tagging) |


## Coreference Resolution

| Model ID | Trained | PRE |
|----------|:-------:|:------------:|
| uw-coref-e2e-en-ontonotes | [OntoNotes](english_datasets.html#ontonotes) | [tokseg](#tokenization) |


## Sentiment Analysis

| Model ID | Trained | PRE |
|----------|:-------:|:------------:|
| elit-senti-embatt-en-sst | [SST](english_datasets.html#stanford-sentiment-treebank) | [tokseg](#tokenization) |
