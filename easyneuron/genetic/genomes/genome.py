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

from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from typing import SupportsIndex, get_args

from easyneuron.exceptions import DimensionsError
from easyneuron.metrics import losses
from easyneuron.types.types import Loss, Numerical
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

    def mutate(self, rate: float = 0.1, magnitude: float = 0.1, **kwargs) -> Genome:
        """Mutates the genome
        Pass "lower_bound" and/or "upper_bound" to the method to  change the bounds for the random numbers to be used for updating genome elements

        Parameters
        ----------
        rate : float, optional
            The proportion of the genome, out of 1, to change, by default 0.1
        magnitude : float, optional
            A multiplier to increase the magnitude of the mutations, by default 0.1

        Returns
        -------
        object
            The mutated version of itself
        """
        # how many elements in genome are changed
        num_changes = max(1, int(len(self._genome) * rate))

        bounds = (kwargs.get("lower_bound") or -5,
                  kwargs.get("upper_bound") or 5) # in case users have changed this

        shape = copy(tuple(self.genome.shape)) # copied to prevent changes when reshaped in the line below
        self._genome.reshape(1, -1) # so it can be accessed with a single index

        for _ in range(num_changes):
            random_index = randint(0, len(self._genome) - 1) if len(self._genome) != 1 else 0 # choose a random index to edit

            self._genome[random_index] += randint(
                bounds[0], bounds[1]) * magnitude # update it with random(lower, upper) * magnitude

        self._genome.reshape(shape) # reshape to original shape

        return self

    @property
    def genome(self) -> ndarray:
        """The genome of the Genome instance

        Returns
        -------
        ndarray
            The genome
        """
        return self._genome

    @genome.setter
    def genome(self, value):
        """Set the genome of the Genome instance - auto changes it to a NumPy array

        Parameters
        ----------
        value : Any
            The value to set it to
        """
        self._genome = array(value, dtype=float)


def child_of(g1: Genome, g2: Genome, method: Loss = "mse", max_loss: float = 0.1, **kwargs) -> Genome:
    """child_of returns the child of 2 genomes.
    Pass "fill_value" to specify the value to insert if parents are not similar enough, or pass "fill_value_factory" to use a callable that returns the fill value.

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
    loss_fn = losses.get(method) or (lambda x: x)  # the loss function to use

    fill_value = kwargs.get("fill_value") or kwargs.get("fill_value_factory") or 0 # the fill value to use if the 2 parents are too different on an attribute
    def fill():
        return fill_value if isinstance(fill_value, get_args(Numerical)) else fill_value() # in case the fill value is callable

    # reshape for compatability
    g1 = g1.genome.reshape(1, -1).tolist()[0]
    g2 = g2.genome.reshape(1, -1).tolist()[0]

    if len(g1) != len(g2):
        raise DimensionsError(
            f"the 2 genomes must have the same number of items, not {len(g1)} and {len(g2)}.")

    child = [
        mean([i, j]) # in between the 2 parent's elements
        if loss_fn([i], [j]) < max_loss # if it is less than the max_loss
        else fill()
        for i, j in zip(g1, g2) # zip them together
    ]

    return Genome(child)
