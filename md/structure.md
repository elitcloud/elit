# Structure

## Table of Contents

* [NLPNode](#nlpgraph).
* [NLPNode](#nlpnode).

## NLPGraph

The following code reads the first dependency graph from [sample.tsv](../resources/sample/sample.tsv):

```
from elit.reader import TSVReader
from elit.structure import NLPNode

reader = TSVReader(1, 2, 3, 4, 5, 6, 7, 8)
reader.open('sample.tsv')

graph = reader.next()


 
```
* See [TSVReader](reader.md#tsvreader) for more details about the reader.