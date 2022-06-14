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
from typing import Any, Callable
import numpy as np

# Value Updaters
QTableUpdaterFunction = Callable[[float, float, float, float, Any], float]

def bellman_updater(
		reward: float, old_reward: float = 0,
		learning_rate: float = 1, discount_rate: float = 1,
		state_rewards: Any = None
) -> float:
	"""Q-Table updater for Bellman's equation.

	Parameters
	----------
	reward : float
		The current reward.
	old_reward : float, optional
		The previous reward of the current state/action pair, by default 0
	learning_rate : float, optional
		The learning rate, by default 1
	discount_rate : float, optional
		The discount rate, by default 1
	state_rewards : SupportsIndex, optional
		An iterable of rewards that contains the rewards for the current state, by default None

	Returns
	-------
	float
		The updated reward.
	"""

	if state_rewards is None:
		state_rewards = [0]

	return old_reward * (1 - learning_rate) + learning_rate * (reward + discount_rate * np.max(state_rewards))
