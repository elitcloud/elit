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

import hashlib
import json
import logging
import math
import os
import pathlib
import platform
import re
import sys
import time
import zipfile
from typing import Set
from urllib.parse import urlparse
from urllib.request import urlretrieve

import numpy as np

import requests
from tqdm import tqdm

from elit.resources.pre_trained_models import RESOURCE_URL_PREFIX

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
    with open(filename, encoding='utf-8') as fin:
        s = set(line.strip() for line in fin)
    logging.info('Init: %s (keys = %d)' % (filename, len(s)))
    return s


def read_concat_word_dict(filename) -> dict:
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

    with open(filename, encoding='utf-8') as fin:
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


def file_exist(filename) -> bool:
    return os.path.isfile(filename)


def remove_file(filename):
    if file_exist(filename):
        os.remove(filename)


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


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


def elit_data_dir_default():
    """

    :return: default data directory depending on the platform and environment variables
    """
    system = platform.system()
    if system == 'Windows':
        return os.path.join(os.environ.get('APPDATA'), 'elit')
    else:
        return os.path.join(os.path.expanduser("~"), '.elit')


def elit_data_dir():
    """

    :return: data directory in the filesystem for storage, for example when downloading models
    """
    return os.getenv('ELIT_HOME', elit_data_dir_default())


def merge_args_with_config(args) -> dict:
    args = vars(args)
    if args.get('config_path', None):
        with open(args.get('config_path')) as src:
            json_args = json.load(src)  # type: dict
            for k, v in json_args.items():
                if k not in args:
                    args[k] = v
            # args = parser.parse_args(' '.join('--{} {}'.format(k, v) for k, v in args.items()))
    return args


def save_json(item: dict, path: str, ensure_ascii=False):
    with open(path, 'w') as out:
        json.dump(item, out, ensure_ascii=ensure_ascii, indent=2)


def load_json(path):
    with open(path) as src:
        return json.load(src)


if __name__ == '__main__':
    print(elit_data_dir())


class Progbar(object):
    """Progbar class copied from keras (https://github.com/fchollet/keras/)

    Displays a progress bar.
    Small edit : added strict arg to update
    # Arguments
        target: Total number of steps expected.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1):
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def update(self, current, values=[], exact=[], strict=[]):
        """
        Updates the progress bar.
        # Arguments
            current: Index of current step.
            values: List of tuples (name, value_for_last_step).
                The progress bar will display averages for these values.
            exact: List of tuples (name, value_for_last_step).
                The progress bar will display these values directly.
        """

        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far), current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)

        for cells in exact:
            k, v, w = cells[0], cells[1], 4
            if len(cells) == 3:
                w = cells[2]
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = [v, 1, w]

        for k, v in strict:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = v

        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = 0 if self.target == 0 or math.isnan(self.target) else int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = 0 if self.target == 0 else float(current) / self.target
            prog_width = int(self.width * prog)
            if prog_width > 0:
                bar += ('=' * (prog_width - 1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.' * (self.width - prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit * (self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                if type(self.sum_values[k]) is list:
                    info += (' - %s: %.' + str(self.sum_values[k][2]) + 'f') % (
                        k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                else:
                    info += ' - %s: %s' % (k, self.sum_values[k])

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width - self.total_width) * " ")

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                sys.stdout.write(info + "\n")

    def add(self, n, values=[]):
        self.update(self.seen_so_far + n, values)


def make_sure_path_exists(path):
    os.makedirs(path, exist_ok=True)


def download_friendly(url, path=None, model_root=elit_data_dir(), prefix=RESOURCE_URL_PREFIX):
    if not path:
        path = path_from_url(url, prefix, model_root=model_root)
        os.makedirs(path, exist_ok=True)
    if os.path.isfile(path):
        print('Using local {}, ignore {}'.format(path, url))
        return path
    else:
        if os.path.isdir(path):
            path = os.path.join(path, url.split('/')[-1])
        print('Downloading {} to {}'.format(url, path))
        tmp_path = '{}.downloading'.format(path)
        remove_file(tmp_path)
        try:
            def reporthook(count, block_size, total_size):
                global start_time, progress_size
                if count == 0:
                    start_time = time.time()
                    progress_size = 0
                    return
                duration = time.time() - start_time
                duration = max(1e-8, duration)
                progress_size = int(count * block_size)
                if progress_size > total_size:
                    progress_size = total_size
                speed = int(progress_size / (1024 * duration))
                ratio = progress_size / total_size
                ratio = max(1e-8, ratio)
                percent = ratio * 100
                eta = duration / ratio * (1 - ratio)
                minutes = eta / 60
                seconds = eta % 60
                sys.stdout.write("\r%.2f%%, %d MB, %d KB/s, ETA %d min %d s" %
                                 (percent, progress_size / (1024 * 1024), speed, minutes, seconds))
                sys.stdout.flush()

            import socket
            socket.setdefaulttimeout(10)
            urlretrieve(url, tmp_path, reporthook)
            print()
        except Exception as e:
            remove_file(tmp_path)
            raise e
        remove_file(path)
        os.rename(tmp_path, path)
    return path


def path_from_url(url, prefix=RESOURCE_URL_PREFIX, parent=True, model_root=elit_data_dir()):
    path = model_root
    parsed = urlparse(url[len(prefix):] if url.startswith(prefix) else url)
    if parsed.path:
        path = os.path.join(path, *parsed.path.strip('/').split('/'))
        if parent:
            path = os.path.dirname(path)
    return path


def unzip(path, folder=None, remove_zip=True):
    if folder is None:
        folder = os.path.dirname(path)
    print('Extracting {} to {}'.format(path, folder))
    with zipfile.ZipFile(path, "r") as archive:
        archive.extractall(folder)
    if remove_zip:
        remove_file(path)
    return folder


def fetch_resource(path: str, model_root=os.path.join(elit_data_dir(), 'models'), auto_unzip=True):
    """
    Fetch real path for a resource (model, corpus, whatever)
    :param path: the general path (can be a url or a real path)
    :param auto_unzip: whether to unzip it if it's a zip file
    :param model_root:
    :return: the real path to the resource
    """
    if not model_root:
        model_root = os.path.join(elit_data_dir(), 'models')
    if os.path.isdir(path):
        return path
    elif os.path.isfile(path):
        pass
    elif path.startswith('http:') or path.startswith('https:'):
        realpath = path_from_url(path, parent=False, model_root=model_root)
        if not os.path.isfile(realpath):
            if realpath.endswith('.zip'):
                realpath = realpath[:-len('.zip')]
            if os.path.isdir(realpath) or os.path.isfile(realpath):
                return realpath
            path = download_friendly(url=path, model_root=model_root)
        else:
            path = realpath
    if auto_unzip and path.endswith('.zip'):
        unzip(path)
        path = path[:-len('.zip')]
    return path