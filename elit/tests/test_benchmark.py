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
from elit.benchmark import Timer

__author__ = "Gary Lai"


def test_benchmark(space_tokenizer):
    with Timer() as time1:
        space_tokenizer.decode("Hello, world")
    with Timer() as time2:
        space_tokenizer.decode(
            "This module implements specialized container datatypes providing alternatives to Python’s general purpose built-in containers.")

    assert isinstance(time1.runtime, float)
    assert isinstance(time2.runtime, float)
    assert time1.runtime > 0.0
    assert time2.runtime > 0.0
