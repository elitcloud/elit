# ========================================================================
# Copyright 2016 Emory University
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

import argparse
import glob
import sys
import os
import re

# ==================================== Arguments ====================================

parser = argparse.ArgumentParser(description='Generate byte indices for constituency trees.')
parser.add_argument('byte_file', type=str, help='filename of the output')
parser.add_argument('tree_path', type=str, help='filepath to the directory containing input files')
parser.add_argument('tree_ext', type=str, help='extension of the input files', default='parse')
args = parser.parse_args()

# ==================================== ConstituencyByte ====================================




print(args.byte_file)
print(args.tree_path)
print(args.tree_ext)