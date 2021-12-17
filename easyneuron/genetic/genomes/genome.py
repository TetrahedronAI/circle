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

from copy import deepcopy
from typing import SupportsIndex

from easyneuron.exceptions import DimensionsError
from numpy import array, ndarray
from numpy.random import randint


class Genome(object):

    def __init__(self, genome: SupportsIndex) -> None:
        if len(array(genome).shape) > 1:
            raise DimensionsError(
                f"the genome should be 1 dimensional, not {len(array(genome).shape)}.\nTry using <arrayName>.reshape(1, -1) on your genome data.")
        self.genome = genome

    def mutate(self, rate, magnitude: float = 0.1, **kwargs) -> ndarray:
        num_changes = max(1, int(len(self._genome) * rate))
        bounds = (kwargs.get("lower_bound") or -5,
                  kwargs.get("upper_bound") or 5)

        shape = deepcopy(tuple(self.genome.shape))
        self._genome.reshape(1, -1)

        for _ in range(num_changes):
            self._genome[randint(0, len(self._genome) - 1)
                        ] += randint(bounds[0], bounds[1]) * magnitude

        self._genome.reshape(shape)

        return self.genome

    @property
    def genome(self) -> ndarray:
        return self._genome

    @genome.setter
    def genome(self, value):
        self._genome = array(value, dtype=float)
