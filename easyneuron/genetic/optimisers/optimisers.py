# Copyright 2021 Neuron-AI GitHub Authors. All Rights Reserved.
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

from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple
from warnings import warn

from easyneuron.genetic.genomes import Genome, child_of
from easyneuron.types.types import Loss, X_Data
from numpy.random import uniform
from numpy.core import array

@dataclass(repr=True, eq=True, order=True, unsafe_hash=True)
class BasicOptimiser(object):
	def __init__(self, loss: Callable[[X_Data, Sequence, Sequence], float], population_size: int = 1000, genome_shape: Tuple[int] = (1,), **kwargs) -> None:
		warn(FutureWarning("genetic optimisers run, and work, but convergence is presently difficult. Use at your own risk."))
		self.population_size = population_size
		self._loss = loss

		self._genome_shape = genome_shape

		# boundaries for random genome contents
		self._bounds = (kwargs.get("lower_bound") or -5,
						kwargs.get("upper_bound") or 5)

	def _sort_genomes(self, X: X_Data, y: Sequence, population: Sequence[Genome]) -> List[Genome]:
		losses = {self._loss(X, y, genome.genome): genome for genome in population} # create a dict with loss:genome
		losses = [losses[key] for key in sorted(list(losses.keys()))] # get the genomes from the losses sorted by key
		return losses

	def optimise(self, X: X_Data, y: Sequence, child_loss: Loss = "mae", child_max_loss: float = 0.1, mutation_rate: float = 0.1, mutation_magnitude: float = 1, **kwargs):
		sort_by_loss = kwargs.get("sort_by_loss") if kwargs.get("sort_by_loss") is not None else True

		population = [Genome(
			uniform(self._bounds[0], self._bounds[1], size=self._genome_shape)
		) for _ in range(self.population_size)]

		while len(population) > 1:
			if sort_by_loss:
				population = self._sort_genomes(X, y, population) # sort by losses

			if ((len(population) % 2) != 0) and (len(population) != 1):
				population.pop() # to ensure that there are no extras at the end of pairing up

			population = array(population).reshape((-1, 2)).tolist() # turn into pairs
			population = [
				child_of(
				i.mutate(mutation_rate, mutation_magnitude),
				j.mutate(mutation_rate, mutation_magnitude),

				method=child_loss,
				max_loss=child_max_loss,
				fill_value = kwargs.get("fill_value"))
				for i, j in population
			] # generate children

		return population[0]

