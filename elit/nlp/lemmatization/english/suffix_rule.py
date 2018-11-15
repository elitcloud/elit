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

__author__ = "Liyan Xu"


class SuffixRule:
    def __init__(self, suffix_form: str, replacements: Sequence[str], double_consonants: bool, set_base: set):
        """
        :param suffix_form:
        :param replacements:
        :param double_consonants:
        :param set_base:
        """
        self.suffix_form = suffix_form
        self.replacements = replacements
        self.double_consonants = double_consonants
        self.set_base = set_base

    def get_base_form(self, lower: str):
        """
        Get base form by enumerating each token_affixes.
        :param lower:
        :return:
        """
        # Check if input matches this rule's suffix form
        if not lower.endswith(self.suffix_form):
            return None

        # Get base form without considering double consonants
        stem = lower[:len(lower) - len(self.suffix_form)]
        base = self.__get_valid_base_by_rules__(stem)

        # If applicable, consider double consonants
        if base is None and self.double_consonants and self.__is_double_consonants__(stem):
            base = self.__get_valid_base_by_rules__(stem[:len(stem) - 1])

        return base

    def __get_valid_base_by_rules__(self, stem: str):
        """
        Given a stem, try to get a matched base form by enumerating each replacement.
        :param stem:
        :return:
        """
        for replacement in self.replacements:
            base = self.__get_valid_base__(stem, replacement)
            if base is not None:
                return base
        return None

    def __get_valid_base__(self, stem: str, new_suffix: str):
        """
        :param stem:
        :param new_suffix:
        :return:
        """
        base = stem + new_suffix
        return base if base in self.set_base else None

    @classmethod
    def __is_double_consonants__(cls, stem: str):
        """
        :param stem:
        :return:
        """
        length = len(stem)
        return len(stem) >= 2 and stem[length - 1] == stem[length - 2]