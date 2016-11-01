# ========================================================================
# Copyright 2016 Emory University
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
__author__ = 'Jinho D. Choi'


# ==================================== NLPNode ====================================

class NLPNode:
    def __init__(self, token_id: int = -1, word_form: str = None, lemma: str = None, syntactic_tag: str = None,
                 named_entity_tag: str = None):
        # initial
        self._word_form = None
        self.simplified_word_form = None

        # fields
        self.token_id = token_id
        self.word_form = word_form
        self.lemma = lemma
        self.syntactic_tag = syntactic_tag
        self.named_entity_tag = named_entity_tag

        # structure
        self.children_list = list()  # list of ConstituencyNode
        self.parent = None           # ConstituencyNode
        self.left_sibling = None     # ConstituencyNode
        self.right_sibling = None    # ConstituencyNode
        self.antecedent = None       # ConstituencyNode

    @property
    def word_form(self):
        return self._word_form

    @word_form.setter
    def word_form(self, form):
        self._word_form = form
        self.simplified_word_form = form