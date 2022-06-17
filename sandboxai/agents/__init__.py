"""easyneuron.agents is a suite of reinforcement learning tools for quick development, production and research.

Current Tools Available
-----------------------

Environments
	+ Environment - an abstract base class for environments
	+ SimpleLateralMover - a test environment for debugging. Get your agent to go right (constant policy).
"""
# Copyright 2022 Neuron-AI GitHub Authors. All Rights Reserved.
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

from sandboxai.agents.envs._classes import Environment
from sandboxai.agents.envs.examples import SimpleLateralMover
