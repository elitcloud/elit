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

__author__ = "Gary Lai"


class BILOU:
    B = 'B'  # beginning
    I = 'I'  # inside
    L = 'L'  # last
    O = 'O'  # outside
    U = 'U'  # unit

    @classmethod
    def collect(cls, tags):
        """
        :param tags: a list of tags encoded by the BILOU format.
        :type tags: list of str
        :return: a dictionary where the keys represent the chunk spans and the values represent the tags
        """
        entities = {}
        begin = -1

        for i, tag in enumerate(tags):
            t = tag[0]

            if t == cls.B:
                begin = i
            elif t == cls.I:
                pass
            elif t == cls.L:
                if begin >= 0: entities[(begin, i + 1)] = tag[2:]
                begin = -1
            elif t == cls.O:
                begin = -1
            elif t == cls.U:
                entities[(i, i + 1)] = tag[2:]
                begin = -1

        return entities

    def quick_fix(self, tags):
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

            if p == self.B:
                if c == self.B:
                    fix(i, pt, ct, self.I, self.U)  # BB -> BI or UB
                elif c == self.U:
                    fix(i, pt, ct, self.L, self.U)  # BU -> BL or UU
                elif c == self.O:
                    tags[i - 1][0] = self.U  # BO -> UO
            elif p == self.I:
                if c == self.B:
                    fix(i, pt, ct, self.I, self.L)  # IB -> II or LB
                elif c == self.U:
                    fix(i, pt, ct, self.I, self.L)  # IU -> II or LU
                elif c == self.O:
                    tags[i - 1][0] = self.L  # IO -> LO
            elif p == self.L:
                if c == self.I:
                    fix(i, pt, ct, self.I, self.B)  # LI -> II or LB
                elif c == self.L:
                    fix(i, pt, ct, self.I, self.B)  # LL -> IL or LB
            elif p == self.O:
                if c == self.I:
                    tags[i][0] = self.B  # OI -> OB
                elif c == self.L:
                    tags[i][0] = self.B  # OL -> OB
            elif p == self.U:
                if c == self.I:
                    fix(i, pt, ct, self.B, self.B)  # UI -> BI or UB
                elif c == self.L:
                    fix(i, pt, ct, self.B, self.B)  # UL -> BL or UB

        for i in range(1, len(tags)):
            aux(i)
        p = tags[-1][0]
        if p == self.B:
            tags[-1][0] = self.U
        elif p == self.I:
            tags[-1][0] = self.L
