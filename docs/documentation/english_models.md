# English Models

* `PRE`: tools that need to be run prior to the model.
* `DATA`: the dataset used to build the model.
* `EVAL`: evaluation on the dataset that the model is trained on. 
* `BM`: the standard benchmark evaluation for the task.
* `tokseg`: any [tokenizer](#tokenization) with sentence segmentation.
* `inmixed`: any dataset in the [mixed](english_datasets.html#mixed) corpus.

## Tokenization

| Model ID |
|----------|
| [elit-tok-space-un](../tools/tokenization.html#space-tokenizer)     |
| [elit-tok-lexrule-en](../tools/tokenization.html#english-tokenizer) |


## Morphological Analysis

| Model ID | PRE |
|----------|:---:|
| [elit-morph-idprule-en](../tools/morphological_analysis.html#english-analyzer) | [&#10035;-pos-&#10035;-en-inmixed](#part-of-speech-tagging) |


## Part-of-Speech Tagging

| Model ID | PRE | DATA | EVAL | BM |
|----------|:---:|:----:|:----:|:--:|
| [elit-pos-cnn-en-mixed](../tools/part_of_speech_tagging.html#cnn-tagger)     | [tokseg](#tokenization) | [Mixed](english_datasets.html#mixed) | 97.xx | |
| [elit-pos-rnn-en-mixed](../tools/part_of_speech_tagging.html#rnn-tagger)     | [tokseg](#tokenization) | [Mixed](english_datasets.html#mixed) | 97.xx | | 
| [elit-pos-flair-en-mixed](../tools/part_of_speech_tagging.html#flair-tagger) | [tokseg](#tokenization) | [Mixed](english_datasets.html#mixed) | 97.80 | 97.72 |

* EVAL: accuracy.
* BM: accuracy on the Wall Street Journal portion of the [Penn Treebank](https://catalog.ldc.upenn.edu/ldc99t42) using the standard split (trn: 0-18; dev: 19-21; tst: 22-24).


## Named Entity Recognition

| Model ID | PRE | DATA | EVAL | BM |
|----------|:---:|:----:|:----:|:--:|
| [elit-ner-cnn-en-ontonotes](../tools/named_entity_recognition.html#cnn-tagger)     | [tokseg](#tokenization) | [OntoNotes](english_datasets.html#ontonotes) | 87.xx | |
| [elit-ner-rnn-en-ontonotes](../tools/named_entity_recognition.html#rnn-tagger)     | [tokseg](#tokenization) | [OntoNotes](english_datasets.html#ontonotes) | 86.xx | |
| [elit-ner-flair-en-ontonotes](../tools/named_entity_recognition.html#flair-tagger) | [tokseg](#tokenization) | [OntoNotes](english_datasets.html#ontonotes) | 88.75 | 92.74 | 

* EVAL: F1-score.
* BM: F1-score on the English dataset distributed by the [CoNLL'03 shared task](https://www.clips.uantwerpen.be/conll2003/ner/).


## Dependency Parsing

| Model ID | PRE | DATA | EVAL | BM |
|----------|:---:|:----:|:----:|:--:|
| [elit-dep-biaffine-en-mixed](../tools/dependency_parsing.html#biaffine-parser) | [&#10035;-pos-&#10035;-en-inmixed](#part-of-speech-tagging) | [Mixed](english_datasets.html#mixed) | 92.26/91.03 | 96.08/95.02 |  

* EVAL: UAS (unlabeled attachment score) / LAS (labeled attachment score).
* BM: UAS/LAS on the Wall Street Journal portion of the [Penn Treebank](https://catalog.ldc.upenn.edu/ldc99t42) using the standard split (trn: 2-21; dev: 22, 24; tst: 23) 
and the [Stanford typed dependencies](https://nlp.stanford.edu/software/stanford-dependencies.html).


## Coreference Resolution

| Model ID | PRE | DATA | EVAL |
|----------|:---:|:----:|:----:|
| [uw-coref-e2e-en-ontonotes](../tools/coreference_resolution.html#end-to-end-system) | [tokseg](#tokenization) | [CoNLL12](http://conll.cemantix.org/2012/) | 80.4/70.8/67.6 | 

* EVAL: F1-scores of MUC/B3/CEAF.
* CoNLL12: the English dataset distributed by the [CoNLL'12 shared task](http://conll.cemantix.org/2012/).


<!--
## Sentiment Analysis

| Model ID | PRE | DATA | EVAL | BM |
|----------|:---:|:----:|:----:|:--:|
| elit-senti-embatt-en-sst | [SST](english_datasets.html#stanford-sentiment-treebank) | [tokseg](#tokenization) |
-->