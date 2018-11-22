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

import mxnet as mx

__author__ = "Gary Lai"


def mx_loss(s: str) -> mx.gluon.loss:
    s = s.lower()

    if s == 'softmaxcrossentropyloss':
        return mx.gluon.loss.SoftmaxCrossEntropyLoss()
    if s == 'sigmoidbinarycrossentropyloss':
        return mx.gluon.loss.SigmoidBinaryCrossEntropyLoss()
    if s == 'l2loss':
        return mx.gluon.loss.L2Loss()
    if s == 'l2loss':
        return mx.gluon.loss.L1Loss()
    if s == 'kldivloss':
        return mx.gluon.loss.KLDivLoss()
    if s == 'huberloss':
        return mx.gluon.loss.HuberLoss()
    if s == 'hingeloss':
        return mx.gluon.loss.HingeLoss()
    if s == 'squaredhingeloss':
        return mx.gluon.loss.SquaredHingeLoss()
    if s == 'logisticloss':
        return mx.gluon.loss.LogisticLoss()
    if s == 'tripletloss':
        return mx.gluon.loss.TripletLoss()
    if s == 'ctcloss':
        return mx.gluon.loss.CTCLoss()

    raise TypeError("Unsupported loss: " + s)

