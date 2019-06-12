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
import sys
from os.path import expanduser

__author__ = "Gary Lai"

HOME = expanduser("~")
CONFIG_PATH = '{HOME}/.elit'.format(HOME=HOME)
EMB_PATH = '{CONFIG_PATH}/emb'.format(CONFIG_PATH=CONFIG_PATH)
ELITNLP_PATH = '{CONFIG_PATH}/elitnlp'.format(CONFIG_PATH=CONFIG_PATH)
MODEL_PATH = '{CONFIG_PATH}/model'.format(CONFIG_PATH=CONFIG_PATH)
REPO_PATH = '{CONFIG_PATH}/repo'.format(CONFIG_PATH=CONFIG_PATH)

sys.path.insert(1, ELITNLP_PATH)