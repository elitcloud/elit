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
import abc
import argparse
import logging
import sys

__author__ = "Gary Lai"


class BaseCLI(abc.ABC):

    def __init__(self, name: str, usage: str = None, description: str = None):
        """
        :param name: the name of the component.
        :param description: the description of this component; if ``None``, the name is used instead.
        """
        if not description:
            description = name

        parser = argparse.ArgumentParser(usage=usage, description=description)
        parser.add_argument('command', help='{} command to run'.format(name))
        args = parser.parse_args(sys.argv[2:3])

        if not hasattr(self, args.command):
            logging.info('Unrecognized command: ' + args.command)
            parser.print_help()
            exit(1)
        getattr(self, args.command)()