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
import pickle
import time

__author__ = 'Jinho D. Choi'

from hyperopt import fmin, tpe, hp, STATUS_OK, base


def tmp(l):
    return l[0] * l[1]

space = [hp.choice('x', [-5, 5]), hp.choice('y', [0, 10])]
trials = base.Trials()

best = fmin(fn=tmp,
    space=space,
    algo=tpe.suggest,
    max_evals=100,
    trials=trials)
print(best)
