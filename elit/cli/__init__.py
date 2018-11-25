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
import sys

__author__ = "Gary Lai"


class ElitCli(object):
    def __init__(self):
        # set_logger()
        usage = '''elit <command> [<args>]

        commands:
            token_tagger: use token tagger
            install: Install model
            download: download pre-trained model or word embedding
        '''
        parser = argparse.ArgumentParser(usage=usage)
        parser.add_argument('command', help='command to run')
        args = parser.parse_args(sys.argv[1:2])
        if args.command == 'token_tagger':
            from elit.nlp.token_tagger import TokenTaggerCLI
            TokenTaggerCLI()
        elif args.command == 'install':
            from elit.cli.install import InstallCLI
            InstallCLI()
        elif args.command == 'download':
            from elit.cli.download import DownloadCLI
            DownloadCLI()
        else:
            print('Unrecognized command')
            parser.print_help()
            exit(1)


from .base import BaseCLI
from .component import ComponentCLI
