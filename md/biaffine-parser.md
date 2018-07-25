# Deep Biaffine Attention Neural Dependency Paser

## Introduction

This is a re-implementation of Stanford's "[Deep Biaffine Attention for Neural Dependency Parsing](https://arxiv.org/pdf/1611.01734.pdf)" at ICLR 2017. They achieved state-of-the-art score:

> Our parser gets state of the art or near state of the art performance on standard treebanks for six different languages, achieving 95.7% UAS and 94.1% LAS on the most popular English PTB dataset. This makes it the highest-performing graph-based parser on this benchmark— outperforming Kiperwasser & Goldberg (2016) by 1.8% and 2.2%, and comparable to the highest performing transition-based parser (Kuncoro et al., 2016), which achieves 95.8% UAS and 94.6% LAS.

![From my blog](http://wx3.sinaimg.cn/large/006Fmjmcly1fltvsqfjn0j31iu0lqdlg.jpg)

Actually Kuncoro et al. (2016) ensembled two models: they employed one separately trained generative model to rerank `100` phrase trees parsed by another discriminative model. Whereas the biaffine parser is relatively simple and elegant.

## Quick Start

### Data Format

#### CoNLL

Biaffine parser takes conll formatted files as input, eg. Universal Dependencies Treebank or PTB. Here is a example:

```
1	Is	_	VBZ	VBZ	_	4	cop	_	_
2	this	_	DT	DT	_	4	nsubj	_	_
3	the	_	DT	DT	_	4	det	_	_
4	future	_	NN	NN	_	0	root	_	_
5	of	_	IN	IN	_	4	prep	_	_
6	chamber	_	NN	NN	_	7	nn	_	_
7	music	_	NN	NN	_	5	pobj	_	_
8	?	_	.	.	_	4	punct	_	_
```

#### Word Embeddings

Most commonly available embeddings would work, eg. word2vec, [GloVe](https://nlp.stanford.edu/projects/glove/).

### Configuration File

You need to specify path to dataset and model hyper-parameters in a config file. One example can be found at `elit/dev/biaffineparser/configs/ptb.ini`.

If you don't care about hyper-parameters at all, you only need to specify the data section:

```
[Data]
pretrained_embeddings_file = data/glove/glove.6B.100d.txt
data_dir = data/ptb
train_file = %(data_dir)s/train.conllx
dev_file = %(data_dir)s/dev.conllx
test_file = %(data_dir)s/test.conllx
```

### Training and Testing

When you have settled data and configuration files down, you can train a model and evaluate it by running:

```
python3 -m elit.dev.biaffine_parser --config_file my/config.ini
```

The parser will save model and config file in `save_dir` specified of config file.

## API

After you trained a model, you can re-use it within some simple steps.

### 1.Read config file

As the config file contains information about the model, one needs to load it at first.

```
config = Config('save_dir/config.ini')
```

### 2.Load the vocabulary

The vocabulary is stored separately, should load it before the creation of parser:

```
vocab = pickle.load(open(config.save_vocab_path, 'rb'))
```

### 3.Create a BiaffineParser

Use the information inside the config file:

```
    parser = BiaffineParser(vocab, config.word_dims, config.tag_dims, config.mlp_keep_prob, config.lstm_layers,
                            config.lstm_hiddens, config.ff_keep_prob, config.recur_keep_prob,
                            config.mlp_arc_size, config.mlp_rel_size, config.dropout_mlp, config.learning_rate,
                            config.beta_1, config.beta_2, config.epsilon, config.save_model_path, config.debug)
```

### 4.Parse raw sentence

The input sentence should be a list of (word, pos-tag) pairs:

```
print(parser.parse([('Is', 'VBZ'), ('this', 'DT'), ('the', 'DT'), ('future', 'NN'), ('of', 'IN'), ('chamber', 'NN'),
                    ('music', 'NN'), ('?', '.')]))
```

The output is a a CoNLLSentence object, you can print it out:

```
1	Is	_	VBZ	VBZ	_	4	cop	_	_
2	this	_	DT	DT	_	4	nsubj	_	_
3	the	_	DT	DT	_	4	det	_	_
4	future	_	NN	NN	_	0	root	_	_
5	of	_	IN	IN	_	4	prep	_	_
6	chamber	_	NN	NN	_	7	nn	_	_
7	music	_	NN	NN	_	5	pobj	_	_
8	?	_	.	.	_	4	punct	_	_
```

You can also check more documentation at `elit.dev.biaffineparser.common.data.CoNLLSentence`. 


## Performance

All tricks are implemented now:

* [x] Word dropout
* [x] Dropout across time-steps in RNN cell
* [x] Orthonormal initialization of RNN weights
* [x] Learning rate decay

Now the performance is:

|  | UAS | LAS |
| --- | --- | --- |
| PTB-SD | 95.55 % | 94.48 % |

* Reported by the CoNLL 2006 official evaluation scripts `eval.pl`.
* The training script also reports unlabeled and labeled attachment accuracy in console, but it calculates punctuation differently from what is standard. You should instead use the perl script for test score. You can use `parser.parse_file(config.test_file, 'testout.conllx')` to produce test outputs.


