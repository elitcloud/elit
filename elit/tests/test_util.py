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
import pytest

__author__ = "Gary Lai"


def test_util_accuracy(accuracy):
    accuracy.correct = 9
    accuracy.total = 10
    assert accuracy.get() == 90.0
    accuracy.init()
    assert accuracy.correct == 0
    assert accuracy.total == 0


def test_util_f1(f1):
    f1.correct = 5
    f1.p_total = 6
    f1.r_total = 6
    assert f1.get() == (83.33333333333333, 83.33333333333333, 83.33333333333333)


@pytest.mark.parametrize('filepath', ["emory"])
def test_util_file(filepath):
    from ..util import pkl, gln
    assert pkl(filepath) == 'emory.pkl'
    assert gln(filepath) == 'emory.gln'