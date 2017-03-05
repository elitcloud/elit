# ========================================================================
# Copyright 2017 Emory University
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
from bisect import *

__author__ = 'Jinho D. Choi'


def bisect_index(a, x, lo=0, hi=None) -> int:
    """
    :param a: a sorted list.
    :param x: the element to search for.
    :param lo: the lower-bound for search (inclusive).
    :param hi: the upper-bound for search (exclusive).
    :return: the index of the node in the sorted list of tuples if exists; otherwise, -1.
    """
    if hi is None: hi = len(a)
    idx = bisect_left(a, x, lo, hi)
    return idx if idx != len(a) and a[idx] == x else -1


def bisect_remove(a, x, lo=0, hi=None) -> int:
    """
    :param a: a sorted list.
    :param x: the element to search for.
    :param lo: the lower-bound for search (inclusive).
    :param hi: the upper-bound for search (exclusive).
    :return: the index of the removed item in the sorted list of tuples if exists; otherwise, -1.
    """
    idx = bisect_index(a, x, lo, hi)
    if idx >= 0: del a[idx]
    return idx

