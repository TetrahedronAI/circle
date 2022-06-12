import functools
from typing import Any, Callable, List, Union

import numpy as np
from numpy import array


class QTable(object):
	__slots__ = "actions", "states", "table"

	def __init__(self, n_actions: int) -> None:
		self.actions: List[int] = list(range(n_actions))
		self.states: List[Any] = []
		self.table: np.ndarray = np.zeros((0, n_actions))
	
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

	def update(self, state: Any, action: int, reward: float, updater_function: Callable[[int]] = None):
		if updater_function is None:
			updater_function = lambda x: x

		self.table[self.states.index(state)][self.actions.index(action)] += updater_function(reward)
