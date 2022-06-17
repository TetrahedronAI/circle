from dataclasses import dataclass
from functools import total_ordering
from typing import List, Optional, Union, overload

from sandboxai.exceptions.exceptions import EmptyDataError
from sandboxai.genetic.genome.mutate import GenomeMutatorFunc, mutate

@dataclass(unsafe_hash=True, eq=True, repr=True)
@total_ordering
class BaseGenome:
	"""Base class for all genomes. Can be used alone."""
	genes: List[float]
	fitness: float = 0.0

	def __init__(self, genes: List[float]=None):
		"""Create a new genome.

		Parameters
		----------
		genes : List[float], optional
			The genes for the genome, by default None

		Raises
		------
		EmptyDataError
			Thrown if no genes are given.
		"""
		if genes is None:
			raise EmptyDataError("genomes cannot be empty.")

		self.genes = genes or []

	def __gt__(self, other):
		return self.fitness > other.fitness
	
	@overload
	def fitness(self):
		"""Returns the fitness.""" 

	@overload
	def fitness(self, fitness: float):
		"""Set the fitness to the given float."""

	def fitness(self, fitness: Optional[float] = None) -> Union[float, None]:
		"""Returns or sets the fitness.

		Parameters
		----------
		fitness : Optional[float], optional
			The new fitness, by default None

		Returns
		-------
		Union[float, None]
			Returns the fitness if no fitness is given, otherwise returns None.
		"""
		if fitness is None:
			return self.fitness

		self.fitness = fitness
