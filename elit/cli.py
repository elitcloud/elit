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
import argparse
import sys

__author__ = "Gary Lai"


class ElitCli(object):

    def __init__(self):
        parser = argparse.ArgumentParser(
            usage='''
    elit <command> [<args>]

commands:
    pos     part-of-speech tagger
'''
        )
        parser.add_argument('command', help='command to run')
        args = parser.parse_args(sys.argv[1:2])
        # TODO
        # print('Unrecognized command')
        # parser.print_help()
        # exit(1)


if __name__ == '__main__':
    ElitCli()
