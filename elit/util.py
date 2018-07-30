# ========================================================================
# Copyright 2018 Emory University
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
import inspect

from typing import List, Tuple

__author__ = 'Jinho D. Choi'


# ======================================== Structure ========================================

def to_gold(key: str) -> str:
    return key + '-gold'


def to_out(key: str) -> str:
    return key + '-out'


class BILOU:
    B = 'B'  # beginning
    I = 'I'  # inside
    L = 'L'  # last
    O = 'O'  # outside
    U = 'U'  # unit

    @classmethod
    def to_chunks(cls, tags: List[str], fix: bool = False) -> List[Tuple[int, int, str]]:
        """
        :param tags: a list of tags encoded by BILOU.
        :param fix: if True, fixes potential mismatches in BILOU (see :meth:`heuristic_fix`).
        :return: a list of tuples where each tuple contains (begin index (inclusive), end index (exclusive), label) of the chunk.
        """
        if fix: cls.heuristic_fix(tags)
        chunks = []
        begin = -1

        for i, tag in enumerate(tags):
            t = tag[0]

            if t == cls.B:
                begin = i
            elif t == cls.I:
                pass
            elif t == cls.L:
                if begin >= 0: chunks.append((begin, i + 1, tag[2:]))
                begin = -1
            elif t == cls.O:
                begin = -1
            elif t == cls.U:
                chunks.append((i, i + 1, tag[2:]))
                begin = -1

        return chunks

    @classmethod
    def heuristic_fix(cls, tags):
        """
        Use heuristics to fix potential mismatches in BILOU.
        :param tags: a list of tags encoded by BLIOU.
        """

        def fix(i, pt, ct, t1, t2):
            if pt == ct:
                tags[i][0] = t1
            else:
                tags[i - 1][0] = t2

        def aux(i):
            p = tags[i - 1][0]
            c = tags[i][0]
            pt = tags[i - 1][1:]
            ct = tags[i][1:]

            if p == cls.B:
                if c == cls.B:
                    fix(i, pt, ct, cls.I, cls.U)  # BB -> BI or UB
                elif c == cls.U:
                    fix(i, pt, ct, cls.L, cls.U)  # BU -> BL or UU
                elif c == cls.O:
                    tags[i - 1][0] = cls.U  # BO -> UO
            elif p == cls.I:
                if c == cls.B:
                    fix(i, pt, ct, cls.I, cls.L)  # IB -> II or LB
                elif c == cls.U:
                    fix(i, pt, ct, cls.I, cls.L)  # IU -> II or LU
                elif c == cls.O:
                    tags[i - 1][0] = cls.L  # IO -> LO
            elif p == cls.L:
                if c == cls.I:
                    fix(i, pt, ct, cls.I, cls.B)  # LI -> II or LB
                elif c == cls.L:
                    fix(i, pt, ct, cls.I, cls.B)  # LL -> IL or LB
            elif p == cls.O:
                if c == cls.I:
                    tags[i][0] = cls.B  # OI -> OB
                elif c == cls.L:
                    tags[i][0] = cls.B  # OL -> OB
            elif p == cls.U:
                if c == cls.I:
                    fix(i, pt, ct, cls.B, cls.B)  # UI -> BI or UB
                elif c == cls.L:
                    fix(i, pt, ct, cls.B, cls.B)  # UL -> BL or UB

        for idx in range(1, len(tags)): aux(idx)
        prev = tags[-1][0]

        if prev == cls.B:
            tags[-1][0] = cls.U
        elif prev == cls.I:
            tags[-1][0] = cls.L


# ======================================== More ========================================

def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }
