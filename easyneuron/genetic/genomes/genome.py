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

from copy import copy
from dataclasses import dataclass
from typing import SupportsIndex

from easyneuron.exceptions import DimensionsError
from easyneuron.metrics import losses
from easyneuron.types.types import Loss
from numpy import array, mean, ndarray
from numpy.random import randint


@dataclass(eq=True, order=True, unsafe_hash=True)
class Genome(object):
    """A genome for genetic algorithms, with mutation and support for easyneuron.genetic.child_of child generation."""

    def __init__(self, genome: SupportsIndex) -> None:
        """A genome for genetic algorithms, with mutation and support for easyneuron.genetic.child_of child generation.

        Parameters
        ----------
        genome : SupportsIndex
            The genome to use.
        """

        self.genome = genome
    
    def __repr__(self) -> str:
        return f"Genome({self.genome.tolist()})"

    def mutate(self, rate: float = 0.1, magnitude: float = 0.1, **kwargs) -> object:
        # how many elements in genome are changed
        num_changes = max(1, int(len(self._genome) * rate))

        bounds = (kwargs.get("lower_bound") or -5,
                  kwargs.get("upper_bound") or 5)

        shape = copy(tuple(self.genome.shape))
        self._genome.reshape(1, -1)

        for _ in range(num_changes):
            random_index = randint(
                0, len(self._genome) - 1) if len(self._genome) != 1 else 0
            self._genome[random_index] += randint(
                bounds[0], bounds[1]) * magnitude

        self._genome.reshape(shape)

        return self

    @property
    def genome(self) -> ndarray:
        return self._genome

    @genome.setter
    def genome(self, value):
        self._genome = array(value, dtype=float)


def child_of(g1: Genome, g2: Genome, method: Loss = "mse", max_loss: float = 0.1, **kwargs) -> Genome:
    """child_of returns the child of 2 genomes.

    Parameters
    ----------
    g1 : Genome
        The first parent genome
    g2 : Genome
        The second parent genome
    method : Loss, optional
        The loss function to use to tell between the 2 parents, by default "mse"
    max_loss : float, optional
        Maximum the loss can be for the child to be created, by default 0.1

    Returns
    -------
    Genome
        The child genome generated

    Raises
    ------
    DimensionsError
        If the 2 parent genomes have different lengths
    """
    loss_fn = losses.get(method) or (lambda x: x)
    fill_value = kwargs.get("fill_value") or 0

    g1 = g1.genome.reshape(1, -1).tolist()[0]
    g2 = g2.genome.reshape(1, -1).tolist()[0]

    if len(g1) != len(g2):
        raise DimensionsError(
            f"the 2 genomes must have the same number of items, not {len(g1)} and {len(g2)}.")
    child = [
        mean([i, j])
        if loss_fn([i], [j]) < max_loss
        else fill_value
        for i, j in zip(g1, g2)
    ]

    return Genome(child)
