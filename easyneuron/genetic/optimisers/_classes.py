from typing import Callable, Sequence
from easyneuron.types import X_Data


class GeneticOptimiser(object):
	def __init__(self, loss_fn: Callable, population_size: int = 10000,  *args, **kwargs) -> None:
		self.loss_fn = loss_fn
	
	def __call__(self, X: X_Data, y: Sequence, *args, **kwargs):
		self.optimise(X, y, *args, **kwargs)
	
	def optimise(self, X: X_Data, y: Sequence):	...
