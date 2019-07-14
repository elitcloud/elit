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
import logging
import os
import sys

__author__ = "Gary Lai"


def set_logger(filename: str = None, level: int = logging.INFO, formatter: logging.Formatter = None):
    log = logging.getLogger()
    log.setLevel(level)
    ch = logging.StreamHandler(sys.stdout) if filename is None else logging.FileHandler(filename)
    if formatter is not None:
        ch.setFormatter(formatter)
    log.addHandler(ch)


def init_logger(root_dir, name="train.log"):
    """Initialize a logger

    Parameters
    ----------
    root_dir : str
        directory for saving log
    name : str
        name of logger

    Returns
    -------
    logger : logging.Logger
        a logger
    """
    os.makedirs(root_dir, exist_ok=True)
    log_formatter = logging.Formatter("%(message)s")
    logger = logging.getLogger(name)
    file_handler = logging.FileHandler("{0}/{1}".format(root_dir, name), mode='w')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    return logger