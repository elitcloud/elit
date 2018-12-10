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

from elit.structure import NER

__author__ = "Gary Lai"


def ner_formatter(docs):
    for doc in docs:
        for sen in doc.sentences:
            start_idx = -1
            idx = 0
            tags = []
            ent = ""
            for tag in sen[NER]:
                BILOU = tag.split("-")[0]
                if BILOU == 'B':
                    if start_idx != -1:
                        tags.append((start_idx, idx, ent))
                    start_idx = idx
                    ent = tag.split("-")[1]
                elif BILOU == 'U' or BILOU == 'O':
                    if start_idx != -1:
                        tags.append((start_idx, idx, ent))
                        start_idx = -1
                        ent = ""
                    if BILOU == 'U':
                        tags.append((idx, idx + 1, tag.split("-")[1]))
                elif BILOU == 'L':
                    if start_idx != -1:
                        tags.append((start_idx, idx + 1, ent))
                        start_idx = -1
                        ent = ""
                idx += 1
            sen[NER] = tags
