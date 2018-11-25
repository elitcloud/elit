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

import inspect

__author__ = "Gary Lai, Jinho D. Choi"

from elit.cli import BaseCLI


class ComponentCLI(BaseCLI):
    """
    :class:`ComponentCLI` is an abstract class to implement a command-line interface for a component.

    Abstract methods to be implemented:
      - :meth:`ComponentCLI.train`
      - :meth:`ComponentCLI.decode`
      - :meth:`ComponentCLI.evaluate`
    """

    def __init__(self, name: str, description: str = None):
        """
        :param name: the name of the component.
        :param description: the description of this component; if ``None``, the name is used instead.
        """
        usage = """elit {} <command> [<args>]
        
        commands:
            train: train a model (gold labels required)
            decode: predict labels (gold labels not required)
            evaluate: evaluate the pre-trained model (gold labels required)
        """.format(name)

        super().__init__(name=name, usage=usage, description=description)

    @classmethod
    @abc.abstractmethod
    def train(cls):
        """
        :param args: the command-line arguments to be parsed by :class:`argparse.ArgumentParser`.

        Trains a model for this component.
        """
        raise NotImplementedError('%s.%s()' % (cls.__class__.__name__, inspect.stack()[0][3]))

    @classmethod
    @abc.abstractmethod
    def decode(cls):
        """
        :param args: the command-line arguments to be parsed by :class:`argparse.ArgumentParser`.

        Predicts labels using this component.
        """
        raise NotImplementedError('%s.%s()' % (cls.__class__.__name__, inspect.stack()[0][3]))

    @classmethod
    @abc.abstractmethod
    def evaluate(cls):
        """
        :param args: the command-line arguments to be parsed by :class:`argparse.ArgumentParser`.

        Evaluates the current model of this component.
        """
        raise NotImplementedError('%s.%s()' % (cls.__class__.__name__, inspect.stack()[0][3]))
