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
import json
from os import listdir
from sandboxai.logging import get_logger
from sandboxai._testutils.gh_actions import notRunningInGitHubActions

SHOULD_LOG_ERRORS = True
if ".devconfig.json" in listdir():
    with open(".devconfig.json", "r") as file:
        SHOULD_LOG_ERRORS = json.load(file)["log_errors"]

if SHOULD_LOG_ERRORS:

    def log_errors(func):
        def wrapper(*args, **kwargs):
            if notRunningInGitHubActions():
                logger = get_logger(f"logs/{func.__module__}.log")

                try:
                    func(*args, **kwargs)
                    logger.info(f"{func.__qualname__} - Test Success")
                except Exception as e:
                    message = (
                        " ".join(str(i) for i in e.args)
                        .replace("\n", " ")
                        .replace("  ", "")
                    )
                    logger.error(
                        f'"{str(type(e))[8:-2]}" @ {func.__qualname__} - {message}'
                    )
                    raise e from e
            else:
                func(*args, **kwargs)

        return wrapper

else:

    def log_errors(func):
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)

        return wrapper
