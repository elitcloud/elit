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

from mxnet import nd
from mxnet.gluon.loss import _apply_weighting, Loss

__author__ = 'Jinho D. Choi'



class BILOU:
    B = 'B'     # beginning
    I = 'I'     # inside
    L = 'L'     # last
    O = 'O'     # outside
    U = 'U'     # unit

    def collect(self, tags):
        """
        :param tags: a list of tags encoded by the BILOU format.
        :type tags: list of str
        :return: a dictionary where the keys represent the chunk spans and the values represent the tags
        """
        entities = {}
        begin = -1

        for i, tag in enumerate(tags):
            c = tag[0]

            if tag == self.B:
                begin = i
            elif tag == self.I:
                pass
            elif tag == self.L:
                if begin >= 0: entities[(begin, i+1)] = tags[2:]
                begin = -1
            elif tag == self.O:
                begin = -1
            elif tag == self.U:
                entities[(i, i+1)] = tags[2:]
                begin = -1

        return entities

    def quick_fix(self, tags):
        def fix(i, pt, ct, t1, t2):
            if pt == ct: tags[i][0] = t1
            else: tags[i-1][0] = t2

        def aux(i):
            p = tags[i-1][0]
            c = tags[i][0]
            pt = tags[i-1][1:]
            ct = tags[i][1:]

            if p == self.B:
                if   c == self.B: fix(i, pt, ct, self.I, self.U)  # BB -> BI or UB
                elif c == self.U: fix(i, pt, ct, self.L, self.U)  # BU -> BL or UU
                elif c == self.O: tags[i-1][0] = self.U           # BO -> UO
            elif p == self.I:
                if   c == self.B: fix(i, pt, ct, self.I, self.L)  # IB -> II or LB
                elif c == self.U: fix(i, pt, ct, self.I, self.L)  # IU -> II or LU
                elif c == self.O: tags[i-1][0] = self.L           # IO -> LO
            elif p == self.L:
                if   c == self.I: fix(i, pt, ct, self.I, self.B)  # LI -> II or LB
                elif c == self.L: fix(i, pt, ct, self.I, self.B)  # LL -> IL or LB
            elif p == self.O:
                if   c == self.I: tags[i][0] = self.B             # OI -> OB
                elif c == self.L: tags[i][0] = self.B             # OL -> OB
            elif p == self.U:
                if   c == self.I: fix(i, pt, ct, self.B, self.B)  # UI -> BI or UB
                elif c == self.L: fix(i, pt, ct, self.B, self.B)  # UL -> BL or UB

        for i in range(1, len(tags)): aux(i)
        p = tags[-1][0]
        if   p == self.B: tags[-1][0] = self.U
        elif p == self.I: tags[-1][0] = self.L


class MultiLabelSoftmaxCrossEntropyLoss(Loss):
    def __init__(self, axis=-1, sparse_label=True, from_logits=False, weight=None,
                 batch_axis=0, **kwargs):
        super(MultiLabelSoftmaxCrossEntropyLoss, self).__init__(weight, batch_axis, **kwargs)
        self._axis = axis
        self._sparse_label = sparse_label
        self._from_logits = from_logits

    def hybrid_forward(self, F, output, label, sample_weight=None):
        if not self._from_logits:
            output = F.log_softmax(output)
        if self._sparse_label:
            # loss = -F.pick(output, label, axis=self._axis, keepdims=True)
            l = -F.pick(output, label, axis=self._axis, keepdims=True)
            d = nd.array([0 if i.asscalar() < 0 else 1 for i in label]).reshape((-1, 1))
            loss = l * d
        else:
            loss = -F.sum(output*label, axis=self._axis, keepdims=True)
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        return F.mean(loss, axis=self._batch_axis, exclude=True)


