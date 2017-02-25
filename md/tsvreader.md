# TSVReader

[TSVReader](../python/elit/reader.py) reads dependency graphs from a file in the TSV (Tab Separated Values) format.
The `tsv` format expects columns delimited by `\t` and graphs separated by `\n`.
The following shows two dependency graphs from [sample.tsv](../../resources/sample/sample.tsv):


```tsv
1   John        john        NNP  _        2   nsbj   4:nsbj           U-PERSON
2   came        come        VBD  _        0   root   _                O
3   to          to          TO   _        4   aux    _                O
4   visit       visit       VB   srl=prp  2   advcl  _                O
5   Emory       emory       NNP  _        6   com    _                B-ORG
6   University  university  NNP  _        4   obj    _                L-ORG
7   yesterday   yesterday   NN   srl=tmp  4   advnp  _                O

1   John        john        NNP  _        3   nsbj   5:nsbj;7:nsbj    U-PERSON
2   had         have        VBD  _        3   aux    _                _
3   found       find        VBN  _        0   root   _                _
4   ,           ,           ,    _        3   p      _                _
5   bought      buy         VBN  _        3   conj   _                _
6   and         and         CC   _        5   cc     _                _
7   read        read        VBN  _        5   conj   _                _
8   the         the         DT   _        9   det    _                _
9   book        book        NN   _        7   obj    3:obj;5:obj      _
10  last        last        JJ   _        11  attr   _                _
11  year        year        NN   sem=TMP  7   advnp  3:advnp;5:advnp  _
```

* `0`: node ID.
* `1`: word form.
* `2`: lemma.
* `3`: part-of-speech tag.
* `4`: extra features; features are delimited by `|`, and keys and values are delimited by `=` (e.g., `k1=v1|k2=v2`).
* `5`: head ID.
* `6`: dependency label.
* `7`: secondary heads; heads are delimited by `;`, and head IDs and labels are delimited by `:`.
* `8`: named entity tags in the BILOU notaiton.

The following code sequentially reads dependency graphs from [sample.tsv](../../resources/sample/sample.tsv):

```python
from elit.reader import TSVReader 

reader = TSVReader(word_index=1, lemma_index=2, pos_index=3, feats_index=4, head_index=5, deprel_index=6, sheads_index=7, nament_index=8)
reader.open('sample.tsv')

for graph in reader:
    print(str(graph)+'\n')
```
* `word_index	`: the column index of the word form.
* `lemma_index`: the column index of the lemma.
* `pos_index`: the column index of the part-of-speech tag.
* `feats_index`: the column index of the extra features.
* `head_index`: the column index of the head ID.
* `deprel_index`: the column index of the dependency label.
* `sheads_index`: the column index of the secondary heads.
* `nament_index`: the column index of the named entity tag.

You can read a graph at a time:

```python
from elit.reader import TSVReader 

reader = TSVReader(1, 2, 3, 4, 5, 6, 7, 8)
reader.open('sample.tsv')

graph = reader.next()
print(str(graph) + '\n')
```

You can also read all graphs at once as a list:

```python
from elit.reader import TSVReader 

reader = TSVReader(1, 2, 3, 4, 5, 6, 7, 8)
reader.open('sample.tsv')

graphs = reader.next_all()
print(len(graphs))
```
