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

from elit.nlp.lemmatization.english.suffix_group import SuffixGroup

__author__ = "Liyan Xu"


class Inflection:
    def __init__(self, base_pos: str, set_base: set, dict_exc: dict, suffix_groups: Sequence[SuffixGroup]):
        """
        :param base_pos:
        :param set_base:
        :param dict_exc:
        :param suffix_groups:
        """
        self.base_pos = base_pos
        self.set_base = set_base
        self.dict_exc = dict_exc
        self.suffix_groups = suffix_groups

    def get_base_form(self, lower: str, pos: str):
        """
        Get base form by looking at exceptions and enumerating each suffix group.
        :param lower:
        :param pos:
        :return:
        """
        # Get base form from exceptions
        base = self.__get_base_from_exceptions__(lower)

        # Get base form from suffixes
        if base is None:
            base = self.__get_base_from_suffixes__(lower, pos)

        return base

    def __get_base_from_exceptions__(self, lower: str):
        """

        :param lower:
        :return:
        """
        return self.dict_exc.get(lower, None)

    def __get_base_from_suffixes__(self, lower: str, pos: str):
        """
        Enumerate each suffix group.
        :param lower:
        :param pos:
        :return:
        """
        for suffix_group in self.suffix_groups:
            base = suffix_group.get_base_form(lower, pos)
            if base is not None:
                return base

        return None
