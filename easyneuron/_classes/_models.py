from abc import ABC, abstractmethod
from typing import Any

class Model(ABC):
	@abstractmethod
	def fit(self, X,*args, **kwargs) -> Any: ...

	@abstractmethod
	def predict(self, X, *args, **kwargs) -> Any: ...
