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
# -*- coding: utf-8 -*-
"""
.. module:: elitsdk.benchmark
   :synopsis: The ELIT software development kit.
.. moduleauthor:: Gary Lai
"""
import timeit

__author__ = "Gary Lai"


class Timer(object):
    """
    Timer is an context manager measures elapsed time of running function or process.

    Example::

        from elitsdk.benchmark import Timer
        from example.example import SpaceTokenizer
        space_tokenizer = SpaceTokenizer()
        with Timer() as t:
            space_tokenizer.decode("hello world")
        print(t.runtime)

    """

    def __init__(self, timer=timeit.default_timer):
        self.timer = timer
        self.start = None
        self.end = None
        self.runtime = None

    def __enter__(self):
        self.start = self.timer()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = self.timer()
        self.runtime = self.end - self.start
