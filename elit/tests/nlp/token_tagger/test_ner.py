# ========================================================================
# Copyright 2018 ELIT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========================================================================
from elit.nlp.token_tagger.ner import ner_formatter
from elit.structure import Document, Sentence

__author__ = "Gary Lai"

def test_ner():
    docs = [{"sens":[{"tok":["Jinho","Choi","is","a","professor","at","Emory","University","in","Atlanta",",","GA","."],"ner":["B-PERSON","L-PERSON","O","O","O","O","B-ORG","L-ORG","O","U-GPE","O","U-GPE","O"],"off":[[0,5],[6,10],[11,13],[14,15],[16,25],[26,28],[29,34],[35,45],[46,48],[49,56],[56,57],[58,60],[60,61]],"sid":0},{"tok":["Dr.","Choi","started","the","Emory","NLP","Research","Group","in","2014","."],"ner":["O","U-PERSON","O","B-ORG","I-ORG","I-ORG","I-ORG","L-ORG","O","U-DATE","O"],"off":[[62,65],[66,70],[71,78],[79,82],[83,88],[89,92],[93,101],[102,107],[108,110],[111,115],[115,116]],"sid":1},{"tok":["He","is","the","founder","of","the","ELIT","project","."],"ner":["O","O","O","O","O","O","U-ORG","O","O"],"off":[[117,119],[120,122],[123,126],[127,134],[135,137],[138,141],[142,146],[147,154],[154,155]],"sid":2}]}]
    expected_docs = [{'sens': [{'tok': ['Jinho', 'Choi', 'is', 'a', 'professor', 'at', 'Emory', 'University', 'in', 'Atlanta', ',', 'GA', '.'], 'ner': [(0, 2, 'PERSON'), (6, 8, 'ORG'), (9, 10, 'GPE'), (11, 12, 'GPE')], 'morph': [[['jinho', 'NN']], [['choi', 'NN']], [['be', 'VB'], ['', 'I_3PS']], [['a', 'DT']], [['profess', 'VB'], ['+or', 'N_ER']], [['at', 'IN']], [['emory', 'NN']], [['university', 'NN']], [['in', 'IN']], [['atlanta', 'NN']], [[',', 'PU']], [['ga', 'NN']], [['.', 'PU']]], 'pos': ['NNP', 'NNP', 'VBZ', 'DT', 'NN', 'IN', 'NNP', 'NNP', 'IN', 'NNP', ',', 'NNP', '.'], 'sid': 0, 'off': [[0, 5], [6, 10], [11, 13], [14, 15], [16, 25], [26, 28], [29, 34], [35, 45], [46, 48], [49, 56], [56, 57], [58, 60], [60, 61]]}, {'tok': ['Dr.', 'Choi', 'started', 'the', 'Emory', 'NLP', 'Research', 'Group', 'in', '2014', '.'], 'ner': [(1, 2, 'PERSON'), (3, 8, 'ORG'), (9, 10, 'DATE')], 'morph': [[['dr.', 'NN']], [['choi', 'NN']], [['start', 'VB'], ['+ed', 'I_PST']], [['the', 'DT']], [['emory', 'NN']], [['nlp', 'NN']], [['research', 'NN']], [['group', 'NN']], [['in', 'IN']], [['2014', 'CD']], [['.', 'PU']]], 'pos': ['NNP', 'NNP', 'VBD', 'DT', 'NNP', 'NNP', 'NNP', 'NNP', 'IN', 'CD', '.'], 'sid': 1, 'off': [[62, 65], [66, 70], [71, 78], [79, 82], [83, 88], [89, 92], [93, 101], [102, 107], [108, 110], [111, 115], [115, 116]]}, {'tok': ['He', 'is', 'the', 'founder', 'of', 'the', 'ELIT', 'project', '.'], 'ner': [(6, 7, 'ORG')], 'morph': [[['he', 'PR']], [['be', 'VB'], ['', 'I_3PS']], [['the', 'DT']], [['found', 'VB'], ['+er', 'N_ER']], [['of', 'IN']], [['the', 'DT']], [['elit', 'NN']], [['project', 'NN']], [['.', 'PU']]], 'pos': ['PRP', 'VBZ', 'DT', 'NN', 'IN', 'DT', 'NNP', 'NN', '.'], 'sid': 2, 'off': [[117, 119], [120, 122], [123, 126], [127, 134], [135, 137], [138, 141], [142, 146], [147, 154], [154, 155]]}]}]
    docs = [Document(sens=[Sentence(sen) for sen in doc['sens']]) for doc in docs]
    expected_docs = [Document(sens=[Sentence(sen) for sen in doc['sens']]) for doc in expected_docs]
    ner_formatter(docs)
    expected_docs == docs