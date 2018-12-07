# Coreference Resolution

## Output Format

The key `coref` is inserted to the [document](../documentation/output_format.html#document), where the value is a list of entity clusters.
The example below shows output for the input text, "_Mr. Johnson bought a truck. He likes it very much!_":

```json
{
  "sens": [
    {
      "sid": 0,
      "tok": ["Mr.", "Johnson", "bought", "a", "truck", "."]
    },
    {
      "sid": 1,
      "tok": ["He", "likes", "it", "very", "much", "!"]
    }],
  "coref": [
    [[0, 0, 2], [1, 0, 1]],
    [[0, 3, 5], [1, 2, 3]]]
}
```

Each entity cluster is represented by a list of mentions, where a mention is represented by a tuple of the following three fields:

* `sentence_id`: the index of the sentence that contains the mention.
* `begin_token_id`: the index of the first token in the mention (inclusive).
* `end_token_id`: the index of the last token in the mention (exclusive).

All indices start with 0 such that `[0, 3, 5]` in the above example indicates a mention in the `0`th sentence that begins with the `3`rd token ("_a_") and ends before the `5`th token ("_._").
Thus, the first cluster contains two mentions, "_Mr. Johnson_" and "_He_", whereas the second cluster contains two mentions, "_a truck_" and "_it_".

## End-to-End System

This is ELIT's adaptation of the higher-order coreference resolution system developed by the University of Washington.

* Original source: [https://github.com/kentonl/e2e-coref](https://github.com/kentonl/e2e-coref)
* Source: [https://github.com/elitcloud/e2e-coref](https://github.com/elitcloud/e2e-coref)
* Associated models: `uw_coref_e2e_en_ontonotes`
* Decode parameters:
  * `genre`: the genre of the input text; `bc` (broadcasting conversation), `bn` (broadcasting news), `mz` (news magazine), `nw` (newswire; _default_), `pt` (pivot text), `tc` (telephone conversation), or `wb` (weblog)


### Web-API

```json
{"model": "uw_coref_e2e_en_ontonotes", "args": {"genre": "nw"}}
```

### Citation

```text
@InProceedings{lee-he-zettlemoyer:NAACL:2018,
  author    = {Lee, Kenton and He, Luheng and Zettlemoyer, Luke},
  title     = {Higher-Order Coreference Resolution with Coarse-to-Fine Inference},
  booktitle = {Proceedings of the Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  year      = {2018},
  series    = {NAACL'18},
  pages     = {687--692},
  url       = {http://www.aclweb.org/anthology/N18-2108}
}
```