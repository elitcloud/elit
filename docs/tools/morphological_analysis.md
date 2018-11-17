# Morphological Analysis

## English Analyzer

This analyzer takes an input token and its part-of-speech tag in the [Penn Treebank style](../documentation/english_datasets.html#mixed), 
and splits it into morphemes using [inflection](https://en.wikipedia.org/wiki/Inflection), [derivation](https://en.wikipedia.org/wiki/Morphological_derivation), and [prefix](https://en.wikipedia.org/wiki/Prefix) rules.

* Associated models: `elit-morph-lexrule-en`.
* API reference: [EnglishMorphAnalyzer](../apidocs/morphological_analyzers.html#elit.nlp.morph_analyzer.EnglishMorphAnalyzer).
* Decode parameters:
  * `derivation`: `True` (_default_) or `False`.
  * `prefix`: `0` (no prefix; _default_), `1` (shortest prefix), `2` (longest prefix).

The followings show key features of the inflection rules (*: irregular):

| Feature | (Token, POS Tag) | Morphemes |
|---------|------------------|-----------|
| Noun, plural            | `('studies', 'NNS')` | `[('study', 'NN'), ('+ies', 'I_PLR')]` |
| Noun, plural            | `('quizzes', 'NNS')` | `[('quiz', 'NN'), ('+es', 'I_PLR')]` |
| Noun, plural*           | `('women', 'NNS')`   | `[('woman', 'NN'), ('+men', 'I_PLR')]` |
| Noun, plural*           | `('wolves', 'NNS')`  | `[('wolf', 'NN'), ('+ves', 'I_PLR')]` |
| Verb, 3rd               | `('studies', 'VBZ')` | `[('study', 'VB'), ('+ies', 'I_3PS')]` |
| Verb, 3rd               | `('pushes', 'VBZ')`  | `[('push', 'VB'), ('+es', 'I_3PS')]` |
| Verb, gerund            | `('taking', 'VBG')`  | `[('take', 'VB'), ('+ing', 'I_GRD')]` |
| Verb, gerund            | `('running', 'VBZ')` | `[('run', 'VB'), ('+ing', 'I_GRD')]` |
| Verb, past              | `('studied', 'VBD')` | `[('study', 'VB'), ('+ied', 'I_PST')]` |
| Verb, past*             | `('bound', 'VBD')`   | `[('bind', 'VB'), ('+ou+', 'I_PST')]` |
| Verb, past*             | `('bit', 'VBD')`     | `[('bite', 'VB'), ('-e', 'I_PST')]` |
| Verb, past*             | `("'d", 'VBD')`      | `[[('have', 'VB'), ('+d', 'I_PST')]` |
| Verb, 3rd, past*        | `('was', 'VBD')`     | `[('be', 'VB'), ('', 'I_3PS'), ('', 'I_PST')]` |
| Adjective, comparative  | `('easier', 'JJR')`  | `[('easy', 'JJ'), ('+ier', 'I_COM')]` |
| Adjective, comparative* | `('worse', 'JJR')`   | `[('bad', 'JJ'), ('', 'I_COM')]` |
| Adjective, superlative  | `('biggest', 'JJS')` | `[('big', 'JJ'), ('+est', 'I_SUP')]` |
| Adjective, superlative* | `('worst', 'JJS')`   | `[[('bad', 'JJ'), ('', 'I_SUP')]` |
| Adverb, comparative     | `('earlier', 'RBR')` | `[('early', 'RB'), ('+ier', 'I_COM')]` |
| Adverb, comparative*    | `('further', 'RBR')` | `[('far', 'RB'), ('+urthe+', 'I_COM')]` |
| Adverb, superlative     | `('soonest', 'RBS')` | `[('soon', MT.RB), ('+est', MT.I_SUP)]` |
| Adverb, superlative*    | `('best', 'RBS')`    | `[('well', 'RB'), ('', 'I_SUP')]` |
| Modal                   | `("'d", 'MD')`       | `[('would', 'MD')]` |

The followings show key features of the derivation rules:

| Feature | (Token, POS Tag) | Morphemes |
|---------|------------------|-----------|
| Verb &rarr;      | `('glorify', 'VB')`          | `[('glory', 'NN'), ('+ify', 'V_FY')]` |
| Verb &rarr;      | `('simplify', 'VB')`         | `[('simple', 'JJ'), ('+ify', 'V_FY')]` |
| Verb &rarr;      | `('theorize', 'VB')`         | `[('theory', 'NN'), ('+ize', 'V_IZE')]` |
| Verb &rarr;      | `('dramatize', 'VB')`        | `[('drama', 'NN'), ('+tic', 'J_IC'), ('+ize', 'V_IZE')]` |
| Verb &rarr;      | `('strengthen', 'VB')`       | `[('strength', 'NN'), ('+en', 'V_EN')]` |
| Noun &rarr;      | `('marriage', 'NN')`         | `[('marry', 'VB'), ('+iage', 'N_AGE')]` |
| Noun &rarr;      | `('mileage', 'NN')`          | `[('mile', 'NN'), ('+age', 'N_AGE')]` |
| Noun &rarr;      | `('approval', 'NN')`         | `[('approve', 'VB'), ('+al', 'N_AL')]` |
| Noun &rarr;      | `('difference', 'NN')`       | `[('differ', 'VB'), ('+ent', 'J_ANT'), ('+ence', 'N_ANCE')]` |
| Noun &rarr;      | `('fluency', 'NN')`          | `[('fluent', 'VB'), ('+ency', 'N_ANCE')]` |
| Noun &rarr;      | `('applicant', 'NN')`        | `[('apply', 'VB'), ('+icant', 'N_ANT')]` |
| Noun &rarr;      | `('freedom', 'NN')`          | `[('free', 'JJ'), ('+dom', 'N_DOM')]` |
| Noun &rarr;      | `('kingdom', 'NN')`          | `[('king', 'NN'), ('+dom', 'N_DOM')]` |
| Noun &rarr;      | `('employee', 'NN')`         | `[('employ', 'VB'), ('+ee', 'N_EE')]` |
| Noun &rarr;      | `('runner', 'NN')`           | `[('run', 'VB'), ('+er', 'N_ER')]` |
| Noun &rarr;      | `('lawyer', 'NN')`           | `[('law', 'NN'), ('+yer', 'N_ER')]` |
| Noun &rarr;      | `('likelihood', 'NN')`       | `[('like', 'NN'), ('+ly', 'J_LY'), ('+hood', 'N_HOOD')]` |
| Noun &rarr;      | `('building', 'NN')`         | `[('build', 'VB'), ('+ing', 'N_ING')]` |
| Noun &rarr;      | `('baptism', 'NN')`          | `[('baptize', 'NN'), ('+ism', 'N_ISM')]` |
| Noun &rarr;      | `('capitalism', 'NN')`       | `[('capital', 'NN'), ('+ize', 'V_IZE'), ('+ism', 'N_ISM')]` |
| Noun &rarr;      | `('environmentalist', 'NN')` | `[('environ', 'VB'), ('+ment', 'N_MENT'), ('+al', 'J_AL'), ('+ist', 'N_IST')]` |
| Noun &rarr;      | `('variety', 'NN')`          | `[('vary', 'VB'), ('+ious', 'J_OUS'), ('+ety', 'N_ITY')]` |
| Noun &rarr;      | `('normality', 'NN')`        | `[('norm', 'NN'), ('+al', 'J_AL'), ('+ity', 'N_ITY')]` |
| Noun &rarr;      | `('chairman', 'NN')`             | `[('chair', 'NN'), ('+man', 'N_MAN')]` |
| Noun &rarr;      | `('development', 'NN')`      | `[('develop', 'VB'), ('+ment', 'N_MENT')]` |
| Noun &rarr;      | `('thinness', 'NN')`         | `[('thin', 'JJ'), ('+ness', 'N_NESS')]` |
| Noun &rarr;      | `('friendship', 'NN')`       | `[('friend', 'JJ'), ('+ship', 'N_SHIP')]` |
| Noun &rarr;      | `('analysis', 'NN')`         | `[('analyze', 'VB'), ('+sis', 'N_SIS')]` |
| Noun &rarr;      | `('verification', 'NN')`     | `[('verify', 'VB'), ('+ication', 'N_TION')]` |
| Noun &rarr;      | `('decision', 'NN')`         | `[('decide', 'VB'), ('+sion', 'N_TION')]` |
| Adjective &rarr; | `('certifiable', 'JJ')`      | `[('cert', 'NN'), ('+ify', 'V_FY'), ('+iable', 'J_ABLE')]` |
| Adjective &rarr; | `('visible', 'JJ')`          | `[('vision', 'NN'), ('+ible', 'J_ABLE')]` |
| Adjective &rarr; | `('economical', 'JJ')`       | `[('economy', 'NN'), ('+ic', 'J_IC'), ('+al', 'J_AL')]` |
| Adjective &rarr; | `('focal', 'JJ')`            | `[('focus', 'NN'), ('+al', 'J_AL')]` |
| Adjective &rarr; | `('pleasant', 'JJ')`         | `[('please', 'VB'), ('+ant', 'J_ANT')]` |
| Adjective &rarr; | `('adherent', 'JJ')`         | `[('adhere', 'VB'), ('+ent', 'J_ANT')]` |
| Adjective &rarr; | `('imaginary', 'JJ')`        | `[('imagine', 'VB'), ('+ary', 'J_ARY')]` |
| Adjective &rarr; | `('monetary', 'JJ')`         | `[('money', 'NN'), ('+tary', 'J_ARY')]` |
| Adjective &rarr; | `('diffused', 'JJ')`         | `[('diffuse', 'VB'), ('+d', 'J_ED')]` |
| Adjective &rarr; | `('helpful', 'JJ')`          | `[('help', 'VB'), ('+ful', 'J_FUL')]` |
| Adjective &rarr; | `('realistic', 'JJ')`        | `[('real', 'NN'), ('+ize', 'V_IZE'), ('+stic', 'J_IC')]` |
| Adjective &rarr; | `('diagnostic', 'JJ')`       | `[('diagnose', 'VB'), ('+sis', 'N_SIS'), ('+tic', 'J_IC')]` |
| Adjective &rarr; | `('dignifying', 'JJ')`       | `[('dignity', 'NN'), ('+ify', 'V_FY'), ('+ing', 'J_ING')]` |
| Adjective &rarr; | `('ticklish', 'JJ')`         | `[('tickle', 'VB'), ('+ish', 'J_ISH')]` |
| Adjective &rarr; | `('boyish', 'JJ')`           | `[('boy', 'NN'), ('+ish', 'J_ISH')]` |
| Adjective &rarr; | `('talkative', 'JJ')`        | `[('talk', 'VB'), ('+ative', 'J_IVE')]` |
| Adjective &rarr; | `('speechless', 'JJ')`       | `[('speech', 'NN'), ('+less', 'J_LESS')]` |
| Adjective &rarr; | `('childlike', 'JJ')`        | `[('child', 'NN'), ('+like', 'J_LIKE')]` |
| Adjective &rarr; | `('daily', 'JJ')`            | `[('day', 'NN'), ('+ily', 'J_LY')]` |
| Adjective &rarr; | `('innermost', 'JJ')`        | `[('inner', 'JJ'), ('+most', 'J_MOST')]` |
| Adjective &rarr; | `('courteous', 'JJ')`        | `[('court', 'NN'), ('+eous', 'J_OUS')]` |
| Adjective &rarr; | `('worrisome', 'JJ')`        | `[('worry', 'NN'), ('+isome', 'J_SOME')]` |
| Adjective &rarr; | `('fulsome', 'JJ')`          | `[('full', 'JJ'), ('+some', 'J_SOME')]` |
| Adjective &rarr; | `('clockwise', 'JJ')`        | `[('clock', 'NN'), ('+wise', 'J_WISE')]` |
| Adjective &rarr; | `('likewise', 'JJ')`         | `[('like', 'JJ'), ('+wise', 'J_WISE')]` |
| Adjective &rarr; | `('rumbly', 'JJ')`           | `[('rumble', 'VB'), ('+y', 'J_Y')]` |
| Adverb &rarr;    | `('beautifully', 'RB')`      | `[('beauty', 'NN'), ('+iful', 'J_FUL'), ('+ly', 'R_LY')]` |


### Web-API

```json
{"model": "elit-morph-lexrule-en", "args": {"derivation": true, "prefix": 0}}
```

### Python API

```python
from elit.structure import Document, Sentence, TOK, POS, MORPH
from elit.tools import EnglishMorphAnalyzer

tokens = ['dramatized', 'ownerships', 'environmentalists', 'certifiable', 'realistically']
postags = ['VBD', 'NNS', 'NNS', 'JJ', 'RB']
doc = Document()
doc.add_sentence(Sentence({TOK: tokens, POS: postags}))

morph = EnglishMorphAnalyzer()
morph.decode([doc], derivation=True, prefix=0)
print(doc.sentences[0][MORPH])
```

### Output

```json
[
  [["drama", "NN"], ["+tic", "J_IC"], ["+ize", "V_IZE"], ["+d", "I_PST"]], 
  [["own", "VB"], ["+er", "N_ER"], ["+ship", "N_SHIP"], ["+s", "I_PLR"]], 
  [["environ", "VB"], ["+ment", "N_MENT"], ["+al", "J_AL"], ["+ist", "N_IST"], ["+s", "I_PLR"]], 
  [["cert", "NN"], ["+ify", "V_FY"], ["+iable", "J_ABLE"]], 
  [["real", "NN"], ["+ize", "V_IZE"], ["+stic", "J_IC"], ["+ally", "R_LY"]]
]
```