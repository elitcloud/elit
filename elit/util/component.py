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
                if begin >= 0: entities[(begin, i+1)] = tag[2:]
                begin = -1
            elif t == cls.O:
                begin = -1
            elif t == cls.U:
                entities[(i, i+1)] = tag[2:]
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




# class ConvBlock(gluon.Block):
#     def __init__(self):
#         super(ConvBlock, self).__init__()
#         with self.name_scope():
#             self.fc0 = gluon.nn.Dense(5)
#             self.fc1 = gluon.nn.Dense(5)
#             self.fc2 = gluon.nn.Dense(5)
#             self.fcn = [self.fc0, self.fc1, self.fc2]
#
#     def forward(self, x):
#         t = [fc(x) for fc in self.fcn]
#         x = nd.concat(*t, dim=0)
#         return x
#
# X = nd.array([
#     [[1,0,0,0,0], [2,0,0,0,0], [3,0,0,0,0], [4,0,0,0,0]],
#     [[0,1,0,0,0], [0,2,0,0,0], [0,3,0,0,0], [0,4,0,0,0]],
#     [[0,0,1,0,0], [0,0,2,0,0], [0,0,3,0,0], [0,0,4,0,0]],
#     [[0,0,0,1,0], [0,0,0,2,0], [0,0,0,3,0], [0,0,0,4,0]],
#     [[0,0,0,0,1], [0,0,0,0,2], [0,0,0,0,3], [0,0,0,0,4]]
# ])
#
# Y = nd.array([[0,1,2],[0,1,2],[0,1,2],[0,-1,-1],[0,1,-1]])
#
# ctx = mx.cpu()
# net = ConvBlock()
# net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
# trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
#
# batch_size = 2
# L = MultiLabelSoftmaxCrossEntropyLoss()
# data = gluon.data.DataLoader(gluon.data.ArrayDataset(X, Y), batch_size=batch_size)
#
# for x, y in data:
#     x = x.as_in_context(ctx)
#     y = y.as_in_context(ctx)
#     y = nd.transpose(y).reshape((-1,1))
#     with autograd.record():
#         output = net(x)
#         print(output)
#         print(y)
#         loss = L(output, y)
#         loss.backward()
#
#
#     trainer.step(x.shape[0])
#
#
# @property
#     def x(self):
#         """
#         If the total number of words in the document is greater than max_len, the rest gets discarded.
#         :param document: a sentence or a list of sentences, where each sentence is represented by a dictionary.
#         :type document: Union[list of dict, dict]
#         :return: a list of feature vectors for the corresponding words across sentences.
#         :rtype: list of numpy.array -> max_len * (word_vsm.dim + ambi_vsm.dim + num_class)
#         """
#         def position(i, size):
#             return self.fst_word if i == 0 else self.lst_word if i+1 == size else self.mid_word
#
#         def aux(sentence):
#             word_emb = sentence.setdefault(WORD_EMB, self.word_vsm.get_list(sentence[TOKEN]))
#             ambi_emb = sentence.setdefault(AMBI_EMB, self.ambi_vsm.get_list(sentence[TOKEN]))
#             pos_scores = sentence.get(POS_OUT, None)
#
#             return [np.concatenate((
#                 position(i, len(sentence)),
#                 word_emb[i],
#                 ambi_emb[i],
#                 self.zero_scores if pos_scores is None else pos_scores[i]))
#                 for i in range(len(sentence))]
#
#         matrix = [v for s in document for v in aux(s)]
#
#         # zero padding
#         if len(matrix) < self.max_len:
#             matrix.extend([np.zeros(len(matrix[0]))] * (self.max_len - len(matrix)))
#         elif len(matrix) > self.max_len:
#             matrix = matrix[:self.max_len]
#
#         return np.array(matrix)
#
#     def gold_labels(self, document):
#         labels = np.concatenate([d[POS_GOLD] for d in document])
#
#         # zero padding
#         if len(labels) < self.max_len:
#             return np.append(labels, np.full(self.max_len - len(labels), -1))
#         elif len(labels) > self.max_len:
#             return labels[:self.max_len]
#         else:
#             return labels
#
#     def set_scores(self, document, output):
#         """
#         :param document: a sentence or a list of sentences, where each sentence is represented by a dictionary.
#         :type document: Union[list of dict, dict]
#         :param output: the output of the part-of-speech tag predictions.
#         :param output: numpy.array -> max_len * num_class
#         """
#         def index(i):
#             return (begin + i) * self.num_class
#
#         def get(sentence):
#             return [output[index(i):index(i+1)] for i in range(0, len(sentence))]
#
#         begin = 0
#
#         if isinstance(document, dict):
#             document[POS_OUT] = get(document)
#         else:
#             for d in document:
#                 sc = get(d)
#                 d[POS_OUT] = sc
#                 begin += sc
#
# def trunc_pads(labels, output):
#     idx = next((i for i, label in enumerate(labels) if label.asscalar() == -1), None)
#     return (labels[:idx], output[:idx]) if idx else (labels, output)

# def evaluate(dat_path, word_vsm, ambi_vsm, model_path):
#     word_vsm = FastText(word_vsm)
#     ambi_vsm = Word2Vec(ambi_vsm) if ambi_vsm else None
#     comp = POSTagger(mx.gpu(0), word_vsm, ambi_vsm, model_path=model_path)
#
#     cols = {TOKEN: 0, POS: 1}
#     dev_states = read_tsv(dat_path, cols, comp.create_state)
#     dev_eval = comp.evaluate(dev_states, 512, Accuracy())
#     print(dev_eval)

