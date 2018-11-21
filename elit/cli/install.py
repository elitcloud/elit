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
import os
import subprocess
import sys

from git import Repo
from pkg_resources import resource_filename

from elit import MODEL_PATH, EMB_PATH, ELITNLP_PATH
from elit.cli import BaseCLI
from elit.util.io import check_resource_dir, file_exist, download, sha1sum

__author__ = "Gary Lai"


def install(filename, source, checksum):
    if not file_exist(filename):
        download(source, filename)
    if sha1sum(filename) != checksum:
        logging.error("Invalid checksum. Please remove {} and run elit install again.".format(filename))


class InstallCLI(BaseCLI):

    def __init__(self):
        name = 'install'
        usage = """elit {} <command> [<args>]

       commands:
           model: pre-train model
       """.format(name)
        super().__init__(name=name, usage=usage)

    @classmethod
    def model(cls):
        parser = argparse.ArgumentParser(description='Download word embeddings',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('model', type=str)
        args = parser.parse_args(sys.argv[3:])
        meta_file = resource_filename('elit.resources.meta.model', '{}.json'.format(args.model))

        with open(meta_file) as f:
            meta = json.load(f)

        check_resource_dir(MODEL_PATH)
        check_resource_dir(EMB_PATH)
        check_resource_dir(ELITNLP_PATH)

        if meta['github']:
            dest = '{}/{}'.format(ELITNLP_PATH, meta['name'])
            Repo.clone_from(url=meta['github'], to_path=dest)
            pip_args = ['-r', '--no-cache-dir']
            cmd = [sys.executable, '-m', 'pip', 'install'] + pip_args + '{}/requirements.txt'.format(dest)
            return subprocess.call(cmd, env=os.environ.copy())
        for model in meta['model']:
            filename = '{}/{}'.format(MODEL_PATH, model['name'])
            source = model['source']
            checksum = model['checksum']
            install(filename, source, checksum)
        for emb in meta['emb']:
            filename = '{}/{}'.format(EMB_PATH, emb['name'])
            source = emb['source']
            checksum = emb['checksum']
            install(filename, source, checksum)
