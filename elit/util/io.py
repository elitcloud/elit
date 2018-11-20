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

import codecs
import hashlib
import json
import logging
import os
import pathlib
import re
import sys
from typing import Set

import requests
from tqdm import tqdm

__author__ = "Jinho D. Choi, Gary Lai"


def pkl(filepath):
    return filepath + '.pkl'


def gln(filepath):
    return filepath + '.gln'


def params(filepath):
    return filepath + '.params'


def read_word_set(filename) -> Set[str]:
    """
    :param filename: the name of the file containing one key per line.
    :return: a set containing all keys in the file.
    """
    fin = codecs.open(filename, mode='r', encoding='utf-8')
    s = set(line.strip() for line in fin)
    logging.info('Init: %s (keys = %d)' % (filename, len(s)))
    return s


def read_concat_word_dict(filename):
    """
    :param filename: the name of the file containing one key per line.
    :return: a dictionary whose key is the concatenated word and value is the list of split points.
    """

    def key_value(line):
        l = [i for i, c in enumerate(line) if c == ' ']
        l = [i - o for o, i in enumerate(l)]
        line = line.replace(' ', '')
        l.append(len(line))
        return line, l

    fin = codecs.open(filename, mode='r', encoding='utf-8')
    d = dict(key_value(line.strip()) for line in fin)
    logging.info('Init: %s (keys = %d)' % (filename, len(d)))
    return d


class NoIndent(object):
    def __init__(self, value):
        self.value = value


class NoIndentEncoder(json.JSONEncoder):
    REGEX = re.compile(r'@@@(\d+)@@@')

    def __init__(self, *args, **kwargs):
        super(NoIndentEncoder, self).__init__(*args, **kwargs)
        self.kwargs = dict(kwargs)
        del self.kwargs['indent']
        self._replacements = {}

    def default(self, o):
        if isinstance(o, NoIndent):
            key = len(self._replacements)
            self._replacements[key] = json.dumps(o.value, **self.kwargs)
            return "@@@%d@@@" % key
        else:
            return super(NoIndentEncoder, self).default(o)

    def encode(self, o):
        result = super(NoIndentEncoder, self).encode(o)
        out = []
        m = self.REGEX.search(result)
        while m:
            key = int(m.group(1))
            out.append(result[:m.start(0) - 1])
            out.append(self._replacements[key])
            result = result[m.end(0) + 1:]
            m = self.REGEX.search(result)
        return ''.join(out)


def check_resource_dir(directory):
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)


def file_exist(filename):
    return os.path.isfile(filename)


def remove_file(filename):
    os.remove(filename)


def sha1sum(filename):
    buf_size = 65536
    sha1 = hashlib.sha1()
    with open(filename, 'rb') as f:
        while True:
            data = f.read(buf_size)
            if not data:
                break
            sha1.update(data)
    return sha1.hexdigest()


def download(source, filename):
    try:
        with open(filename, "wb") as f:
            response = requests.get(source, stream=True)
            total_size = response.headers.get('content-length', 0)

            if total_size is None:
                f.write(response.content)
            else:
                block_size = 32 * 1024
                total_size = int(total_size) // block_size
                with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
                    for data in response.iter_content(chunk_size=block_size):
                        f.write(data)
                        pbar.set_description('Downloading: {}'.format(filename))
                        pbar.update(len(data))
    except KeyboardInterrupt:
        if file_exist(filename):
           os.remove(filename)
        sys.exit()