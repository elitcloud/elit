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
import argparse
import json
import logging
import sys

from pkg_resources import resource_filename

from elit import EMB_PATH
from elit.cli import BaseCLI
from elit.util.io import check_resource_dir, file_exist, download, sha1sum

__author__ = "Gary Lai"


class DownloadCLI(BaseCLI):

    def __init__(self):
        name = 'download'
        usage = """elit {} <command> [<args>]

       commands:
           emb: word embedding file
       """.format(name)
        super().__init__(name=name, usage=usage)

    @classmethod
    def emb(cls):
        parser = argparse.ArgumentParser(description='Download word embeddings',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('model', type=str)
        args = parser.parse_args(sys.argv[3:])
        meta_file = resource_filename('elit.resources.meta.emb', '{}.json'.format(args.model))
        with open(meta_file) as f:
            meta = json.load(f)
        check_resource_dir(EMB_PATH)
        filename = '{}/{}'.format(EMB_PATH, meta['name'])
        if not file_exist(filename):
            download(meta['source'], filename)
        if sha1sum(filename) != meta['checksum']:
            logging.error("{}: checksum is invalid. Please remove it and and run download command it again.".format(filename))
        else:
            logging.error("{} exist.".format(filename))

