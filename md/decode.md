# Decoding

## Flag

The flag consists of 4 digits:

1. `1` if the input format is `line`, `0` if it is `raw`.
1. `1` if the the document is split by `line`, `0` if it is by `delim`.
1. `1` if tokenization is chosen; otherwise, `0`.
1. `1` if segmentation is chosen; otherwise, `0`.
1. `1` if sentiment analysis is chosen; otherwise, `0`.

For example, if the input format is `raw` and only tokenization and segmentation is chosen, the flag is `0110`.
If the input format is `line` and only sentiment analysis is chosen, the flag is `1001`.

## Input Format

ELIT supports two types of input formats:

* `raw`: no segmentation is assumed for the input text.

   ```
   This is an example of the 
   raw format. It assumes no
   segmentation for the input text.
   ```
   
   This option uses the delimiter `<@#DOC$%>` to indicate the end of each document.
   The delimiter must be the only non-white space string in the line of its presence.
   The delimiter is not required for the last document.
   
   ```
   This is the first
   document. Contents for
   the 1st document are here
   <@#DOC$%>
   This is the second
   documents. Contents for
   the 2nd document are here.
   ```

* `line`: each line is considered a paragraph, which is delimited by the newline character `'\n'`.

   ```
   The 1st sentence in the 1st paragraph. The 2nd sentence in the 1st paragraph.
   The 1st sentence in the 2nd paragraph.
   ```
   
   Each line may consist of multiple sentences.
   
   ```
   This is the first sentence. The second sentence is here.
   It ensures to separate the first two sentences from this sentence.
   ```
   
   This option uses a blank line (a line with only white spaces) to indicate the end each document.

## Document Split

There are two ways of splitting the input text into documents:

* `delim`: use the delimiter `<@#DOC$%>` to indicate the end of each document.

   ```
   This is the first document.
   Contents for the first document.
   <@#DOC$%>
   This is the second document.
   Contents for the second document.
   <@#DOC$%>
   This is the third document.
   Contents for the third document.
   ```
   
   The delimiter must be the only non-white space string in the line of its presence.

* `line`: each line is considered a document and delimited by the newline character `'\n'`.

   ```
   This is the first document. Contents for the first document.
   This is the second document. Contents for the second document.
   This is the third document. Contents for the third document.
   ```
   
   This option assumes no blank line


## Tokenization

Tokenization splits the input text into linguistic tokens.
For example,

```
I'm Jinho, a professor at Emory.
```

it splits the above text into the following tokens:

```python
["I", "'m", "Jinho", ",", "a", "professor", "at", "Emory", "."]
```

If tokenization is not chosen, the text is split by only white spaces,

```python
["I'm", "Jinho,", "a", "professor", "at", "Emory."]
```

which can be useful if the input is pre-tokenized (that is not the case for the above example).

## Segmentation

Segmentation gives boundaries for grouping tokens into sentences. Given the following tokens,

```
["This", "is", "the", "first", "sentence", ".", "The", "second", "sentence", "is", "here", "."]
```

it generates the following list:

```
[(0, 6), (0, 12)]
```

Each pair indicates a sentence boundary, and the first and the second numbers of each pair indicate the begin (inclusive) and the end (exclusive) indices of tokens in the corresponding sentence, respectively.
There are two sentences in this example, where the first sentence contains 0th ~ 5th tokens and the second sentence includes 6th ~ 11th tokens.

## Sentiment Analysis

Sentiment analysis gives a sentiment (positive, neutral, negative) of the input text.