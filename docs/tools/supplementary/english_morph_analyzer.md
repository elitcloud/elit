# English Morphological Analyzer

The part-of-speech tags follow the [Penn Treebank style](../../documentation/english_datasets.html#mixed).

## Inflection Tagset

| Inflection | POS Tags | Description |
|:----------:|----------|-------------|
| `I_3PS` | `VBZ`         | 3rd-person singular, present tense |
| `I_GRD` | `VBG`         | Gerund |
| `I_PST` | `VBD`, `VBN`  | Past tense/participle |
| `I_PLR` | `NNS`, `NNPS` | Plural |
| `I_COM` | `JJR`, `RBR`  | Comparative |
| `I_SUP` | `JJS`, `RBS`  | Superlative |

The followings show examples of each inflection type (*: irregular):

| Inflection | (Token, POS Tag) | Morphemes |
|------------|------------------|-----------|
| `I_3PS`  | `('studies', 'VBZ')` | `[('study', 'VB'), ('+ies', 'I_3PS')]` |
| `I_3PS`  | `('pushes', 'VBZ')`  | `[('push', 'VB'), ('+es', 'I_3PS')]` |
| `I_GRD`  | `('taking', 'VBG')`  | `[('take', 'VB'), ('+ing', 'I_GRD')]` |
| `I_GRD`  | `('running', 'VBZ')` | `[('run', 'VB'), ('+ing', 'I_GRD')]` |
| `I_PST`  | `('studied', 'VBD')` | `[('study', 'VB'), ('+ied', 'I_PST')]` |
| `I_PST`* | `('bound', 'VBD')`   | `[('bind', 'VB'), ('+ou+', 'I_PST')]` |
| `I_PST`* | `('bit', 'VBD')`     | `[('bite', 'VB'), ('-e', 'I_PST')]` |
| `I_PST`* | `("'d", 'VBD')`      | `[[('have', 'VB'), ('+d', 'I_PST')]` |
| `I_PST`* | `('was', 'VBD')`     | `[('be', 'VB'), ('', 'I_3PS'), ('', 'I_PST')]` |
| `I_PLR`  | `('studies', 'NNS')` | `[('study', 'NN'), ('+ies', 'I_PLR')]` |
| `I_PLR`  | `('quizzes', 'NNS')` | `[('quiz', 'NN'), ('+es', 'I_PLR')]` |
| `I_PLR`* | `('women', 'NNS')`   | `[('woman', 'NN'), ('+men', 'I_PLR')]` |
| `I_PLR`* | `('wolves', 'NNS')`  | `[('wolf', 'NN'), ('+ves', 'I_PLR')]` |
| `I_COM`  | `('easier', 'JJR')`  | `[('easy', 'JJ'), ('+ier', 'I_COM')]` |
| `I_COM`* | `('worse', 'JJR')`   | `[('bad', 'JJ'), ('', 'I_COM')]` |
| `I_COM`  | `('earlier', 'RBR')` | `[('early', 'RB'), ('+ier', 'I_COM')]` |
| `I_COM`* | `('further', 'RBR')` | `[('far', 'RB'), ('+urthe+', 'I_COM')]` |
| `I_SUP`  | `('biggest', 'JJS')` | `[('big', 'JJ'), ('+est', 'I_SUP')]` |
| `I_SUP`* | `('worst', 'JJS')`   | `[[('bad', 'JJ'), ('', 'I_SUP')]` |
| `I_SUP`  | `('soonest', 'RBS')` | `[('soon', MT.RB), ('+est', MT.I_SUP)]` |
| `I_SUP`* | `('best', 'RBS')`    | `[('well', 'RB'), ('', 'I_SUP')]` |


## Derivation Tagset

| Type | Derivation Tags |
|:----:|-----------------|
| `VB` &rarr; | `V_EN`, `V_FY`, `V_IZE` |
| `NN` &rarr; | `N_AGE`, `N_AL`, `N_ANCE`, `N_ANT`, `N_DOM`, `N_EE`, `N_ER`, `N_HOOD`, `N_ING`, `N_ISM`  |
|             | `N_IST`, `N_ITY`, `N_MAN`, `N_MENT`, `N_NESS`, `N_SHIP`, `N_SIS`, `N_TION`  |
| `JJ` &rarr; | `J_ABLE`, `J_AL`, `J_ANT`, `J_ARY`, `J_ED`, `J_FUL`, `J_IC`, `J_ING`, `J_ISH`, `J_IVE`  |
|             | `J_LESS`, `J_LIKE`, `J_LY`, `J_MOST`, `J_OUS`, `J_SOME`, `J_WISE`, `J_Y`  |
| `RB` &rarr; | `R_LY` |

The followings show examples of each derivation type:

| Derivation | (Token, POS Tag) | Morphemes |
|------------|------------------|-----------|
| `V_EN`   | `('strengthen', 'VB')`       | `[('strength', 'NN'), ('+en', 'V_EN')]` |
| `V_FY`   | `('glorify', 'VB')`          | `[('glory', 'NN'), ('+ify', 'V_FY')]` |
| `V_FY`   | `('simplify', 'VB')`         | `[('simple', 'JJ'), ('+ify', 'V_FY')]` |
| `V_IZE`  | `('theorize', 'VB')`         | `[('theory', 'NN'), ('+ize', 'V_IZE')]` |
| `V_IZE`  | `('dramatize', 'VB')`        | `[('drama', 'NN'), ('+tic', 'J_IC'), ('+ize', 'V_IZE')]` |
| `N_AGE`  | `('marriage', 'NN')`         | `[('marry', 'VB'), ('+iage', 'N_AGE')]` |
| `N_AGE`  | `('mileage', 'NN')`          | `[('mile', 'NN'), ('+age', 'N_AGE')]` |
| `N_AL`   | `('approval', 'NN')`         | `[('approve', 'VB'), ('+al', 'N_AL')]` |
| `N_ANCE` | `('difference', 'NN')`       | `[('differ', 'VB'), ('+ent', 'J_ANT'), ('+ence', 'N_ANCE')]` |
| `N_ANCE` | `('fluency', 'NN')`          | `[('fluent', 'VB'), ('+ency', 'N_ANCE')]` |
| `N_ANT`  | `('applicant', 'NN')`        | `[('apply', 'VB'), ('+icant', 'N_ANT')]` |
| `N_DOM`  | `('freedom', 'NN')`          | `[('free', 'JJ'), ('+dom', 'N_DOM')]` |
| `N_DOM`  | `('kingdom', 'NN')`          | `[('king', 'NN'), ('+dom', 'N_DOM')]` |
| `N_EE`   | `('employee', 'NN')`         | `[('employ', 'VB'), ('+ee', 'N_EE')]` |
| `N_ER`   | `('runner', 'NN')`           | `[('run', 'VB'), ('+er', 'N_ER')]` |
| `N_ER`   | `('lawyer', 'NN')`           | `[('law', 'NN'), ('+yer', 'N_ER')]` |
| `N_HOOD` | `('likelihood', 'NN')`       | `[('like', 'NN'), ('+ly', 'J_LY'), ('+hood', 'N_HOOD')]` |
| `N_ING`  | `('building', 'NN')`         | `[('build', 'VB'), ('+ing', 'N_ING')]` |
| `N_ISM`  | `('baptism', 'NN')`          | `[('baptize', 'NN'), ('+ism', 'N_ISM')]` |
| `N_ISM`  | `('capitalism', 'NN')`       | `[('capital', 'NN'), ('+ize', 'V_IZE'), ('+ism', 'N_ISM')]` |
| `N_IST`  | `('environmentalist', 'NN')` | `[('environ', 'VB'), ('+ment', 'N_MENT'), ('+al', 'J_AL'), ('+ist', 'N_IST')]` |
| `N_ITY`  | `('variety', 'NN')`          | `[('vary', 'VB'), ('+ious', 'J_OUS'), ('+ety', 'N_ITY')]` |
| `N_ITY`  | `('normality', 'NN')`        | `[('norm', 'NN'), ('+al', 'J_AL'), ('+ity', 'N_ITY')]` |
| `N_MAN`  | `('chairman', 'NN')`             | `[('chair', 'NN'), ('+man', 'N_MAN')]` |
| `N_MENT` | `('development', 'NN')`      | `[('develop', 'VB'), ('+ment', 'N_MENT')]` |
| `N_NESS` | `('thinness', 'NN')`         | `[('thin', 'JJ'), ('+ness', 'N_NESS')]` |
| `N_SHIP` | `('friendship', 'NN')`       | `[('friend', 'JJ'), ('+ship', 'N_SHIP')]` |
| `N_SIS`  | `('analysis', 'NN')`         | `[('analyze', 'VB'), ('+sis', 'N_SIS')]` |
| `N_TION` | `('verification', 'NN')`     | `[('verify', 'VB'), ('+ication', 'N_TION')]` |
| `N_TION` | `('decision', 'NN')`         | `[('decide', 'VB'), ('+sion', 'N_TION')]` |
| `J_ABLE` | `('certifiable', 'JJ')`      | `[('cert', 'NN'), ('+ify', 'V_FY'), ('+iable', 'J_ABLE')]` |
| `J_ABLE` | `('visible', 'JJ')`          | `[('vision', 'NN'), ('+ible', 'J_ABLE')]` |
| `J_AL`   | `('economical', 'JJ')`       | `[('economy', 'NN'), ('+ic', 'J_IC'), ('+al', 'J_AL')]` |
| `J_AL`   | `('focal', 'JJ')`            | `[('focus', 'NN'), ('+al', 'J_AL')]` |
| `J_ANT`  | `('pleasant', 'JJ')`         | `[('please', 'VB'), ('+ant', 'J_ANT')]` |
| `J_ANT`  | `('adherent', 'JJ')`         | `[('adhere', 'VB'), ('+ent', 'J_ANT')]` |
| `J_ARY`  | `('imaginary', 'JJ')`        | `[('imagine', 'VB'), ('+ary', 'J_ARY')]` |
| `J_ARY`  | `('monetary', 'JJ')`         | `[('money', 'NN'), ('+tary', 'J_ARY')]` |
| `J_ED`   | `('diffused', 'JJ')`         | `[('diffuse', 'VB'), ('+d', 'J_ED')]` |
| `J_FUL`  | `('helpful', 'JJ')`          | `[('help', 'VB'), ('+ful', 'J_FUL')]` |
| `J_IC`   | `('realistic', 'JJ')`        | `[('real', 'NN'), ('+ize', 'V_IZE'), ('+stic', 'J_IC')]` |
| `J_IC`   | `('diagnostic', 'JJ')`       | `[('diagnose', 'VB'), ('+sis', 'N_SIS'), ('+tic', 'J_IC')]` |
| `J_ING`  | `('dignifying', 'JJ')`       | `[('dignity', 'NN'), ('+ify', 'V_FY'), ('+ing', 'J_ING')]` |
| `J_ISH`  | `('ticklish', 'JJ')`         | `[('tickle', 'VB'), ('+ish', 'J_ISH')]` |
| `J_ISH`  | `('boyish', 'JJ')`           | `[('boy', 'NN'), ('+ish', 'J_ISH')]` |
| `J_IVE`  | `('talkative', 'JJ')`        | `[('talk', 'VB'), ('+ative', 'J_IVE')]` |
| `J_LESS` | `('speechless', 'JJ')`       | `[('speech', 'NN'), ('+less', 'J_LESS')]` |
| `J_LIKE` | `('childlike', 'JJ')`        | `[('child', 'NN'), ('+like', 'J_LIKE')]` |
| `J_LY`   | `('daily', 'JJ')`            | `[('day', 'NN'), ('+ily', 'J_LY')]` |
| `J_MOST` | `('innermost', 'JJ')`        | `[('inner', 'JJ'), ('+most', 'J_MOST')]` |
| `J_OUS`  | `('courteous', 'JJ')`        | `[('court', 'NN'), ('+eous', 'J_OUS')]` |
| `J_SOME` | `('worrisome', 'JJ')`        | `[('worry', 'NN'), ('+isome', 'J_SOME')]` |
| `J_SOME` | `('fulsome', 'JJ')`          | `[('full', 'JJ'), ('+some', 'J_SOME')]` |
| `J_WISE` | `('clockwise', 'JJ')`        | `[('clock', 'NN'), ('+wise', 'J_WISE')]` |
| `J_WISE` | `('likewise', 'JJ')`         | `[('like', 'JJ'), ('+wise', 'J_WISE')]` |
| `J_Y`    | `('rumbly', 'JJ')`           | `[('rumble', 'VB'), ('+y', 'J_Y')]` |
| `R_LY`   | `('beautifully', 'RB')`      | `[('beauty', 'NN'), ('+iful', 'J_FUL'), ('+ly', 'R_LY')]` |
