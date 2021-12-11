from dataclasses import dataclass
from typing import Any, Iterable
from functools import total_ordering
from easyneuron._classes import Model
from easyneuron.types import Numerical

@dataclass
class _KNNParams(object):
	k: int

@total_ordering
class _KNN(_KNNParams, Model):
	def __lt__(self, other: _KNNParams) -> Any:
		return self.K < other.K

