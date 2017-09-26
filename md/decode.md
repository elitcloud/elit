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

* `line`: each line is considered a segment, delimited by the newline character (`\n`).
For the following example, the second and the third sentences are guaranteed to be separated, which is not ensured by the `raw` format.

   ```
   The first sentence in the first segment. The second sentence in the first segment
   The third sentence in the second segment.
   ```

By default, the entire input text is considered one document.
The input text can be split into multiple documents by annotating the document delimiter: `@#DOC$%`.
The document delimiter must be the only string in the line of its presence.
   
```
This is the first document.
Contents of the first document are here.
@#DOC$%
This is the second document.
The delimiter is not required for the last document.
```

The size of each document is limited to 10MB (including whitespaces) due to the memory efficiency.
Any document exceeding this size will be artificially truncated.

## Tokenization

Tokenization splits the input text into linguistic tokens.
For example,

```
I'm Dr. Choi, a professor at Emory University.
```

it splits the above text into the following tokens:

```python
["I", "'m", "Dr.", "Choi", ",", "a", "professor", "at", "Emory", "University", "."]
```

If tokenization is not chosen, the text is split by only white spaces,

```python
["I'm", "Dr.", "Choi,", "a", "professor", "at", "Emory", "University."]
```

which can be useful if the input is pre-tokenized (that is not the case for the above example).

## Segmentation

Segmentation separates tokens into sentences. Given the following tokens,

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