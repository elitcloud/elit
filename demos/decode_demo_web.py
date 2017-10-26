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

import requests

r = requests.post('https://elit.cloud/public/decode/',
    data={
        # required
        'text': 'I watched "The Sound of Music" last night. The ending could have been better. It is my favorite movie though.',
        # optional (default: 'raw')
        'input_format': 'raw',
        # optional (default: '1')
        'tokenize': '1',
        # optional (default: '1')
        'segment': '1',
        # optional (default: 'mov')
        'sentiment': 'mov'})

print(r.text)