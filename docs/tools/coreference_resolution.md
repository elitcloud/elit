# Coreference Resolution

## End-to-End System

This is ELIT's adaptation of the higher-order coreference resolution system developed by the University of Washington.

* Original source: [https://github.com/kentonl/e2e-coref](https://github.com/kentonl/e2e-coref)
* Source: [https://github.com/elitcloud/uw-coref-e2e](https://github.com/elitcloud/uw-coref-e2e)
* Associated models: `uw-coref-e2e-en-ontonotes`
* Decode parameters:
  * `genre`: the genre of the input text; `bc` (broadcasting conversation), `bn` (broadcasting news), `mz` (news magazine), `nw` (newswire; _default_), `pt` (pivot text), `tc` (telephone conversation), or `wb` (weblog)


### Web-API

```json
{"model": "uw-coref-e2e-en-ontonotes", "args": {"genre": "nw"}}
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