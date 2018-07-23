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
# -*- coding: utf-8 -*-
"""
.. module:: elitsdk.sdk
   :synopsis: The ELIT software development kit.
.. moduleauthor:: Gary Lai
"""

import abc

__author__ = "Gary Lai"


class Component(abc.ABC):
    """
    Component is an abstract class. To deploy your method to the ELIT service,
    your class must be implemented by inheriting this class.

    Example::

        class Example(Component):
            def init(self, **kwargs):
                pass
            def decode(self, input_data, **kwargs):
                pass
            def load(self, model_path, **kwargs):
                pass
            def save(self, model_path, **kwargs):
                pass
            def train(self, trn_data, dev_data, **kwargs):
                pass


    """
    @abc.abstractmethod
    def init(self):
        """
        Implement the init method.

        """
        raise NotImplementedError("Not implemented")

    @abc.abstractmethod
    def decode(self, input_data, *args, **kwargs):
        """
        Implement the decode method.

        :param input_data: expect input for the decode function
        :param args: args for the decode method if needed
        :param kwargs: kwargs for the decode method if needed
        :return: result of your decode method
        :rtype: dict
        """
        raise NotImplementedError("Not implemented")

    @abc.abstractmethod
    def load(self, model_path, **kwargs):
        """
        Implement the method of how to load your model(s).
        If you don't need this method, leave it as pass.

        :type model_path: str
        :param model_path: root path of the model
        :param args: args for the load_model method if needed
        :param kwargs: kwargs for the load_model method if needed
        """
        raise NotImplementedError("Not implemented")

    @abc.abstractmethod
    def save(self, model_path, **kwargs):
        """
        Implement the method of how to save your model(s).
        If you don't need this method, leave it as pass.

        :type model_path: str
        :param model_path: root path of the model
        :param args: args for the save_model method if needed
        :param kwargs: kwargs for the save_model method if needed
        """
        raise NotImplementedError("Not implemented")

    @abc.abstractmethod
    def train(self, trn_data, dev_data, model_path, **kwargs):
        """
        Implement the train method.
        If you don't need this method, leave it as pass.

        :param trn_data: training data
        :param dev_data: developing data
        :param args: args for the load_model method if needed
        :param kwargs: kwargs for the load_model method if needed
        """
        raise NotImplementedError("Not implemented")
