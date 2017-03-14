# Part-of-Speech Tagger

## POSLexicon

Create a lexicon inheriting [NLPLexicon](lexicon.md) that contains embeddings for words and ambiguity classes:

```python
def __init__(self, word_embeddings: KeyedVectors, ambiguity_classes: KeyedVectors=None):
    super().__init__(word_embeddings)
    self.ambiguity_classes: KeyedVectors = init_vectors(ambiguity_classes)
```

Override the `populate` function to add appropriate embeddings to each node in the input graph:

```python
def populate(self, graph: NLPGraph):
    super().populate(graph)
    root = graph.nodes[0]
    
    if self.ambiguity_classes and not hasattr(root, 'ambiguity_class'):
        root.ambiguity_class = get_vector(self.ambiguity_classes)
        
        for node in graph:
            node.ambiguity_class = get_vector(self.ambiguity_classes, node.word)
```