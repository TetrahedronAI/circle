from abc import ABC, abstractmethod
from typing import Any

class Model(ABC):
	@abstractmethod
	def __init__(self) -> None: ...

	@abstractmethod
	def fit(self, X, **kwargs) -> Any: ...

	@abstractmethod
	def predict(self, X, **kwargs) -> Any: ...