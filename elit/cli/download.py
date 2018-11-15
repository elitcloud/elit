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
import logging
import sys

import requests

from elit.cli import BaseCLI

__author__ = "Gary Lai"

BUCKET = 'https://s3-us-west-2.amazonaws.com/elitcloud-public-data/'


class DownloadCLI(BaseCLI):

    def __init__(self):
        name = 'download'
        usage = """elit {} <command> [<args>]

        commands:
            emb: word embedding data
            model: pre-trained model
        """.format(name)

        super().__init__(name=name, usage=usage)

    @classmethod
    def emb(cls):
        parser = argparse.ArgumentParser(description='Download word embeddings file',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # parser.add_argument('-a', '--all', action='store_true')
        parser.add_argument('model', type=str, help='word embedding model, ex: fasttext')
        parser.add_argument('dim', type=int, help='word embedding dim, ex: 400')
        args = parser.parse_args(sys.argv[3:])
        if args.model == 'fasttext':
            pass

    @classmethod
    def model(cls):
        pass


def download(url, filename):
    with open(filename, "wb") as f:
        logging.info("Downloading {}".format(filename))
        response = requests.get(url, stream=True)
        total_length = response.headers.get('content-length')

        if total_length is None:  # no content length header
            f.write(response.content)
        else:
            dl = 0
            total_length = int(total_length)
            for data in response.iter_content(chunk_size=4096):
                dl += len(data)
                f.write(data)
                done = int(50 * dl / total_length)
                sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50 - done)))
                sys.stdout.flush()