import functools
from typing import Any, Callable, List, Union

import numpy as np
from easyneuron.agents.qlearn.value import QTableUpdaterFunction, bellman_updater


class QTable(object):
	"""A Q Table object for organising the states, action and values."""
	__slots__ = "actions", "states", "table"

	def __init__(self, n_actions: int, updater: QTableUpdaterFunction = bellman_updater) -> None:
		self.actions: List[int] = list(range(n_actions))
		self.states: List[Any] = []
		self.table: np.ndarray = np.zeros((0, n_actions))

		self.__updater_function = updater
	
	@functools.singledispatchmethod
	def __getitem__(self, index) -> np.ndarray:
		return self.table[index]
	
	@__getitem__.register(List)
	def _(self, index: List) -> Union[np.ndarray, float]:
		return self.table[index]
	
	@__getitem__.register(int)
	def _(self, index: int) -> np.ndarray:
		return self.table[index]
	
	def add_state(self, state: Any) -> None:
		self.states.append(state)
		self.table = np.vstack((self.table, np.zeros(len(self.actions))))

	def update(self, state: Any, action: int, reward: float):
		if updater_function is None:
			updater_function = lambda x: x

		self.table[self.states.index(state)][self.actions.index(action)] += self.__updater_function(reward)
