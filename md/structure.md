# Structure

## Table of Contents

* [NLPGraph](#nlpgraph).
* [NLPNode](#nlpnode).

## NLPGraph

The following code reads the first dependency graph from [sample.tsv](../resources/sample/sample.tsv) (see [TSVReader](reader.md#tsvreader) for more details):

```python
from elit.reader import TSVReader
from elit.structure import *

reader = TSVReader(1, 2, 3, 4, 5, 6, 7, 8)
reader.open('sample.tsv')
graph = reader.next()
```

The graph consists of a list of nodes, where 0th node represents the root, 1st node represents the first token in the document, and so on.

```python
root = graph.nodes[0]
first_node = graph.nodes[1]
last_node = graph.nodes[-1]
```

The following code iterates through all nodes including the root:

```python
for node in graph.nodes:
    print(str(node))
```

The following code skips the root and iterates from the first node to the last node in the graph:

```python
for node in graph:
    print(str(node))
```

The following code gives the number of tokens in the document represented in this graph:

```python
print(len(graph))
```

## NLPNode

The following code prints each field in the 4th node:

```python
node = graph.nodes[4]

print(node.node_id)  # node ID
print(node.word)     # word form
print(node.lemma)    # lemma
print(node.pos)      # part-of-speech-tag
print(node.nament)   # named entity tag
```

The extra features are stored in a dictionary:

```python
for k, v in node.feats.items():
    print(k+':'+v)
```

The following code retrieves various dependencies:

```python
# primary parent
parent = node.parent
print(parent.word+' -'+node.get_dependency_label()+'-> '+node.word)

# secondary parents
for parent in node.secondary_parents:
    print(parent.word + ' -' + node.get_dependency_label() + '-> ' + node.word)

# primary children
for child in node.children:
    print(node.word+' -'+child.get_dependency_label()+'-> '+child.word)

# secondary children
for child in node.secondary_children:
    print(node.word+' -'+child.get_dependency_label()+'-> '+child.word)

# various
print(node.grandparent)
print(node.get_leftmost_child())
print(node.get_rightmost_child())
print(node.get_left_nearest_child())
print(node.get_right_nearest_child())
print(node.get_leftmost_sibling())
print(node.get_rightmost_sibling())
print(node.get_left_nearest_sibling())
print(node.get_right_nearest_sibling())
```

Note that `left*` or `right*` must be on the lefthand or righthand side of `node`, respectively.

