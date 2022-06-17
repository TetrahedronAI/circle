# Copyright 2021 Neuron-AI GitHub Authors. All Rights Reserved.
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
# ==============================================================================
import logging
from typing import Optional


def get_logger(filename: str, log_format: Optional[str] = None):
    if log_format is None:
        log_format = "%(asctime)s \t [%(levelname)s] \t %(message)s"

    logging.basicConfig(filename=filename, level=logging.DEBUG, format=log_format)
    return logging.getLogger()
