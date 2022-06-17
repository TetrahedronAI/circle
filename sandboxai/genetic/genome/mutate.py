from typing import Callable, Iterable, Union

import numpy as np
from sandboxai.genetic.genome.base import BaseGenome

GenomeMutatorFunc = Union[str, Callable[[BaseGenome, bool], BaseException]]
GenomeMutatorSelection = Union[float, Iterable[int], Callable[[float], float]]

def mutate(genome: BaseGenome, method: GenomeMutatorFunc, **kwargs) -> BaseGenome:
	"""Mutate a genome with the given method.

	Parameters
	----------
	genome : BaseGenome
		The genome to mutate.
	method : GenomeMutatorFunc
		The method to mutate the genome with.

	Returns
	-------
	BaseGenome
		The mutated genome.
	"""
	if isinstance(method, str):
		return mutator_functions[method](genome)
	elif isinstance(method, Callable):
		return method(genome, **kwargs)

def normal_mutate(genome: BaseGenome, mean: float = 0, stddev: float = 0.1, inplace: bool = False) -> BaseGenome:
	"""Mutate every item in the genome by the normal distribution.

	Parameters
	----------
	genome : BaseGenome
		The genome to mutate.
	mean : float, optional
		The mean for the normal distribution, by default 0
	stddev : float, optional
		The standard deviation of the normal distribution, by default 0.1
	inplace : bool, optional
		If True, mutate the genome inplace, by default False

	Returns
	-------
	BaseGenome
		The mutated genome.
	"""
	mutated_genes = (np.array(genome.genes) + np.random.normal(mean, stddev, len(genome.genes))).tolist()

	if inplace:
		genome.genes = mutated_genes

	return BaseGenome(genes=mutated_genes)

def normal_mutate_selective(genome: BaseGenome, selection: GenomeMutatorSelection = 0.1, mean: float = 0, stddev: float = 0.1, inplace: bool = False) -> BaseGenome:
	"""Mutate every item in the genome by the normal distribution, but only mutate the selected items.

	Parameters
	----------
	genome : BaseGenome
		The genome to mutate.
	mean : float, optional
		The mean for the normal distribution, by default 0
	stddev : float, optional
		The standard deviation of the normal distribution, by default 0.1
	inplace : bool, optional
		If True, mutate the genome inplace, by default False

	Returns
	-------
	BaseGenome
		The mutated genome.
	"""
	if isinstance(selection, float):
		selector = lambda _: np.random.random() < selection
	elif isinstance(selection, Iterable):
		selector = lambda x: x in selection
	elif isinstance(selection, Callable):
		selector = selection
	else:
		raise ValueError("The selection method must be a float, iterable, or callable.")
	
	mutated_genes = [i + np.random.normal(mean, stddev) if selector(i) else i for i in range(len(genome.genes))]

	if inplace:
		genome.genes = mutated_genes
	
	return BaseGenome(genes=mutated_genes)

mutator_functions = {
	"normal": normal_mutate,
	"normal_selective": normal_mutate_selective
}