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
import glob
import os

def count(filename):
    sc, wc = 0, 0
    tree = []
    for line in open(filename):
        line = line.strip()
        if line:
            tree.append(line)
        elif tree:
            sc += 1
            wc += len(tree)
            tree = []
    return sc, wc

dat_dir = '/Users/jdchoi/Documents/Data/english/zzz/tsv'
dsc, dwc = {}, {}

for filename in glob.glob(os.path.join(dat_dir, '*/*')):
    key = os.path.basename(filename)[:-4]
    sc, wc = count(filename)
    dsc[key] = dsc.get(key, 0) + sc
    dwc[key] = dwc.get(key, 0) + wc


for k, sc in sorted(dsc.items(), key=lambda x: x[0]):
    print('%s\t%d\t%d' % (k, sc, dwc[k]))