from abc import ABC, abstractmethod
from functools import total_ordering
from typing import Any, Sequence

from easyneuron._classes import Model
from easyneuron.types import X_Data
from easyneuron.types.types import Int


@total_ordering
class _KNN(Model, ABC):
	K: Int

	def __lt__(self, other) -> Any:
		return self.K < other.K

	@abstractmethod
	def fit(self, X: X_Data, y: Sequence[Any], *args, **kwargs) -> Model: ...

	@abstractmethod
	def predict(self, X: X_Data, *args, **kwargs) -> Sequence[Any]: ...
