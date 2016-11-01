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
from enum import Enum
from elit.structure.constituency import *
from typing import Dict
import re
__author__ = 'Jinho D. Choi'


class Direction(Enum):
    left = 'l'
    right = 'r'


class HeadGroup:
    _DELIM_PIPE = re.compile('\|')

    def __init__(self, tags: str):
        self.function_tagset = set()
        l = list()

        for tag in self._DELIM_PIPE.split(tags):
            if tag.startswith('-'):
                self.function_tagset.add(tag[1:])
            else:
                l.append(tag)

        self.syntactic_tag_pattern = re.compile('^('+'|'.join(l)+')$') if l else None

    def match(self, node: ConstituencyNode) -> bool:
        if self.syntactic_tag_pattern and self.syntactic_tag_pattern.match(node.syntactic_tag):
            return True
        if self.function_tagset and self.function_tagset.intersection(node.function_tagset):
            return True
        return False


class HeadRule:
    _DELIM_SEMICOLON = re.compile(';')

    def __init__(self, direction: Direction, rule: str='.*'):
        self.direction = direction
        self.group_list = [HeadGroup(tags) for tags in self._DELIM_SEMICOLON.split(rule)]

    def __getitem__(self, index: int):
        return self.group_list[index]


def read_head_dict(inputstream) -> Dict[str,HeadRule]:
    d = dict()
    for line in inputstream:
        l = line.split()
        d[l[0]] = HeadRule(Direction(l[1]), l[2])
    return d
