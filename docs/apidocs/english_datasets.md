# English Datasets

* sc = sentence count.
* tc = token count.


## CRAFT

* [Colorado Richly Annotated Full Text Corpus](http://bionlp-corpora.sourceforge.net/CRAFT/)
  * Biomedical journal articles (sc = 19,792, tc = 554,539)


## BOLT

* [Broad Operational Language Translation](https://www.ldc.upenn.edu/collaborations/current-projects/bolt)
  * Conversational telephone speech (sc = 11,552, tc = 160,319)
  * Discussion forum (sc = 17,382, tc = 396,584)
  * SMS message (sc = 22,883, tc = 260,431)


## EWT

* [English Web Treebank](https://catalog.ldc.upenn.edu/LDC2012T13)
  * Question-answer (sc = 3,089, tc = 50,404)
  * Email (sc = 3,436, tc = 51,504)
  * Newsgroup (sc = 2,122, tc = 41,891)
  * Review (sc = 2,951, tc = 45,864)
  * Weblog (sc = 1,886, tc = 42,988)


## OntoNotes

* [OntoNotes 5.0](https://catalog.ldc.upenn.edu/LDC2013T19)
  * Broadcasting conversation (sc = 14,648, tc = 239,940)
  * Broadcasting news (sc = 11,867, tc = 240,241)
  * News magazine  (sc = 7,960, tc = 194,926)
  * Newswire (sc = 40,491, tc = 1,038,190)
  * Pivot text (sc = 24,386, tc = 339,013)
  * Telephone conversation (sc = 10,955, tc = 112,847)
  * Weblog (sc = 11,800, tc = 262,049)


## QuestionBank

* [QuestionBank Revised](https://nlp.stanford.edu/data/QuestionBank-Stanford.shtml)
  * Question (sc = 3,989, tc = 38,100)


## MiPACQ

* [Multi-source Integrated Platform for Answering Clinical Questions](http://clear.colorado.edu/compsem/index.php?page=endendsystems&sub=mipacq)
  * Clinical note (sc = 9,706, tc = 132,235)
  * Clinical question (sc = 1,980, tc = 37,178)
  * Medpedia (sc = 2,921, tc = 49,252)
  * Pathological note (sc = 1,182, tc = 22,088)


## SHARP

* [Strategic Health IT Advanced Research Projects](http://informatics.mayo.edu/sharp/index.php/Main_Page)
  * Clinical note (sc = 7,841, tc = 111,789)
  * Seattle group health note (sc = 8,268, tc = 110,208)
  * Stratified (sc = 5,022, tc = 51,629)
  * Stratified Seattle group health note (sc = 15,948, tc = 165,960)


## THYME

* [Temporal History of Your Medical Events](http://clear.colorado.edu/compsem/index.php?page=endendsystems&sub=temporal)
 * Brain cancer note (sc = 21,284, tc = 263,011)
 * Clinical/Pathological note (sc = 30,090, tc = 448,603)


## Mixed

A combined dataset consisting of [CRAFT](CRAFT), [BOLT](BOLT), [EWT](EWT), [OntoNotes](OntoNotes), [QuestionBank](QuestionBank), [MiPACQ](MiPACQ), [SHARP](#SHARP), and [THYME](#THYME).  

### Part-of-Speech Tags

Words:

| Tag     | Description              | Tag     | Description              |
|:-------:|--------------------------|:-------:|--------------------------|
| `ADD`   | Email                    | `PDT`   | Predeterminer |
| `AFX`   | Affix                    | `POS`   | Possessive ending |
| `CC`    | Coordinating conjunction | `RB`    | Adverb |
| `CD`    | Cardinal number          | `RBR`   | Adverb, comparative |
| `DT`    | Determiner               | `RBS`   | Adverb, superlative |
| `EX`    | Existential _there_      | `RP`    | Particle |
| `FW`    | Foreign word             | `TO`    | To |
| `GW`    | Go with                  | `UH`    | Interjection |
| `IN`    | Preposition              | `VB`    | Verb, base form |
| `JJ`    | Adjective                | `VBD`   | Verb, past tense |
| `JJR`   | Adjective, comparative   | `VBG`   | Verb, gerund or present participle |
| `JJS`   | Adjective, superlative   | `VBN`   | Verb, past participle |
| `LS`    | List item                | `VBP`   | Verb, non-3rd person singular present |
| `MD`    | Modal                    | `VBZ`   | Verb, 3rd person singular present |
| `NN`    | Noun, singular or mass   | `WDT`   | _Wh_-determiner |
| `NNS`   | Noun, plural             | `WP`    | _Wh_-pronoun |
| `NNP`   | Proper noun, singular    | `WP$`   | _Wh_-pronoun, possessive |
| `NNPS`  | Proper noun, plural      | `WRB`   | _Wh_-adverb |
| `PRP`   | Pronoun                  | `XX`    | Unknown |
| `PRP$`  | Pronoun, possessive      | | |

Punctuation:

| Tag  | Description | Tag     | Description   | Tag    | Description             |
|:----:|-------------|:-------:|---------------|:------:|-------------------------|
| `$`  | Currency    | ` `` `  | Left quote    | `HYPH` | Hyphen                  |
| `:`  | Colon       | `‘’`    | Right quote   | `NFP`  | Superfluous punctuation |
| `,`  | Comma       | `-LRB-` | Left bracket  | `SYM`  | Symbol                  |
| `.`  | Period      | `-RRB-` | Right bracket | | |


### Named Entity Tags

Named entities:

| Tag           | Description              |
|:-------------:|--------------------------|
| `PERSON`      | People, including fictional. |
| `NORP`        | Nationalities or religious or political groups. |
| `FAC`         | Buildings, airports, highways, bridges, etc. |
| `ORG`         | Companies, agencies, institutions, etc. |
| `GPE`         | Countries, cities, states. |
| `LOC`         | Non-GPE locations, mountain ranges, bodies of water. |
| `PRODUCT`     | Vehicles, weapons, foods, etc. (not services) |
| `EVENT`       | Named hurricanes, battles, wars, sports events, etc. |
| `WORK_OF_ART` | Titles of books, songs, etc. |
| `LAW`         | Named documents made into laws. |
| `LANGUAGE`    | Any named language. |

Other entities:

| Tag           | Description              |
|:-------------:|--------------------------|
| `DATE`     | Absolute or relative dates or periods. | 
| `TIME`     | Times smaller than a day. |
| `PERCENT`  | Percentage (including "%"). |
| `MONEY`    | Monetary values, including unit. |
| `QUANTITY` | Measurements, as of weight or distance. |
| `ORDINAL`  | "first", "second", etc. |
| `CARDINAL` | Numerals that do not fall under another type |


### Dependency Labels

See the [Deep Dependency Guidelines](https://emorynlp.github.io/ddr/doc/) for more details:

| Label   | Description              | Label   | Description              |
|:-------:|--------------------------|:-------:|--------------------------|
| `acl` | clausal modifier of noun | `lv` | light verb |
| `adv` | adverbial | `mark` | clausal marker |
| `advcl` | adverbial clause | `meta` | meta element |
| `advnp` | adverbial noun phrase | `modal` | modal |
| `appo` | apposition | `neg` | negation |
| `attr` | attribute | `nsbj` | nominal subject |
| `aux` | auxiliary verb | `num` | numeric modifier |
| `case` | case marker | `obj` | object |
| `cc` | coordinating conjunction | `p` | punctuation or symbol |
| `com` | compound word | `poss` | possessive modifier |
| `comp` | complement | `ppmod` | prepositional phrase |
| `conj` | conjunct | `prn` | parenthetical notation |
| `cop` | copula | `prt` | verb particle |
| `csbj` | clausal subject | `raise` | raising predicate |
| `dat` | dative | `r-*` | referential |
| `dep` | unclassified dependency |  `relcl` | relative clause |
| `det` | determiner | `root` | root |
| `disc` | discourse element | `voc` | vocative |
| `expl` | expletive | | |