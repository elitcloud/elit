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
from typing import Sequence

from elit.nlp.lemmatization import SuffixRule

__author__ = "Liyan Xu"


class SuffixGroup:
    def __init__(self, affix_canonical_form: str, pos_types: str, rules: Sequence[SuffixRule]):
        """
        :param affix_canonical_form:
        :param pos_types: a string of pos types separated by '|'
        :param rules:
        """
        self.affix_canonical_form = affix_canonical_form
        self.pos_types = pos_types
        self.rules = rules

    def get_base_form(self, lower: str, pos: str):
        """
        If pos_types contains the input pos, get base form by enumerating each rule.
        :param lower:
        :param pos:
        :return:
        """
        # Check if pos matches this group
        if pos not in self.pos_types:
            return None

        # Enumerate each rule
        for rule in self.rules:
            base = rule.get_base_form(lower)
            if base is not None:
                return base

        return None
