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

from elit.component import Component

__author__ = "Gary Lai"


def test_component():
    class TestComponent(Component):
        pass

    with pytest.raises(TypeError):
        TestComponent()


def test_abstract_class():
    class TestComponent(Component):

        def __init__(self):
            super().__init__()

        def init(self):
            super().init()

        def decode(self, input_data, **kwargs):
            super().decode(input_data, **kwargs)

        def load(self, model_path, **kwargs):
            super().load(model_path, **kwargs)

        def train(self, trn_data, dev_data, model_path, **kwargs):
            super().train(trn_data, dev_data, model_path, **kwargs)

        def save(self, model_path, **kwargs):
            super().save(model_path, **kwargs)

    with pytest.raises(NotImplementedError):
        test_task = TestComponent()
        test_task.decode("test")


def test_space_tokenizer(space_tokenizer):
    result = space_tokenizer.decode("Hello, world")
    assert result['tok'] == ['Hello,', 'world']
    assert result['off'] == [(0, 6), (7, 12)]
