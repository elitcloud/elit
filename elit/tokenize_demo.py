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
from elit.tokenizer import SpaceTokenizer, EnglishTokenizer

__author__ = 'Jinho D. Choi'
import sys

seg = sys.argv[3] == 'space'
fin = open(sys.argv[1])
fout = open(sys.argv[2], 'w')
tok = SpaceTokenizer() if seg else EnglishTokenizer()

for i,line in enumerate(fin):
    if i%100000 == 0: print(i)
    d = tok.decode(line)
    for s in d.sentences:
        fout.write(' '.join(s.tokens)+'\n')
