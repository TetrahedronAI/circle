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
import functools
from typing import Any, List, Union

import numpy as np
from sandboxai.agents.qlearn.value import QTableUpdaterFunction, bellman_updater


class QTable(object):
	"""A Q Table object for organising the states, action and values."""
	__slots__ = "actions", "states", "table", "__updater_function"

	def __init__(self, n_actions: int, updater: QTableUpdaterFunction = bellman_updater) -> None:
		self.actions: List[int] = list(range(n_actions))
		self.states: List[Any] = []
		self.table: np.ndarray = np.zeros((0, n_actions))

		self.__updater_function = updater

	@functools.singledispatchmethod
	def __getitem__(self, index) -> np.ndarray:
		return self.table[index]

	@__getitem__.register(list)
	def _(self, index: list) -> Union[np.ndarray, float]:
		return self.table[index]

	@__getitem__.register(int)
	def _(self, index: int) -> np.ndarray:
		return self.table[index]

	def add_state(self, state: Any) -> None:
		self.states.append(state)
		self.table = np.vstack((self.table, np.zeros(len(self.actions))))

	def update(self, state: Any, action: int, reward: float):
		self.table[self.states.index(state)][self.actions.index(action)] += self.__updater_function(reward)
