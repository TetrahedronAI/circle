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
from os import environ


def notRunningInGitHubActions() -> bool:
    """Ensures that the process is not running in a GitHub action

    Returns
    -------
    bool
            True if running locally, False if in a GitHub action
    """
    return environ.get("GITHUB_ACTIONS") not in [
        "true",
        "True",
        "TRUE",
        True,
    ]  # I was not sure if the environment variable would be which of these
