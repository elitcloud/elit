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
__author__ = 'Jinho D. Choi'
import ujson as json


class Document:
    def __init__(self, sentences):
        self.do


class Data:
    def __init__(self, js):
        """
        :param js: either a JSON stream (e.g., StringIO('[...]'), open('tmp.json')) or a list of documents
        """
        self.documents = js if isinstance(js, list) else json.load(js)
        documents = None

        for document in documents:
            for sentence in document:













