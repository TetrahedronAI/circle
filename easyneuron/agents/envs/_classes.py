# Copyright 2022 Neuron-AI GitHub Authors. All Rights Reserved.
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

from abc import ABC, abstractmethod
from typing import Any, Sequence, Union


class Environment(ABC):
    """Base class for RL environments."""

    def __init__(self, *args, **kwargs) -> None:
        """Create an instance of the environment."""
        self.reset(*args, **kwargs)

    def reset(self, *args, **kwargs) -> None:
        """Reset the environment to its original state."""
        ...

    @abstractmethod
    def get_obs(self, *args, **kwargs) -> Sequence:
        """Get the observation of the environment.

        Returns
        -------
        Sequence
                The environment, as a sequence
        """
        return []

    @abstractmethod
    def get_obs_shape(self, *args, **kwargs) -> Sequence:
        """Returns the shape of the observation array. e.g. (100,)

        Returns
        -------
        Sequence
                The observation's shape.
        """
        return []

    @abstractmethod
    def get_all_actions(self, *args, **kwargs) -> Sequence:
        """Get all of the possible actions. We recommend that you use this as returning a sequence of integers.

        Returns
        -------
        Sequence
                The list of possible actions
        """
        return []

    @abstractmethod
    def get_actions(self, *args, **kwargs) -> Sequence:
        """Get the actions possible at this timestep. We recommend that you use this as returning a sequence of integers.

        Returns
        -------
        Sequence
                The possible actions.
        """
        return []

    @abstractmethod
    def get_reward(self, *args, **kwargs) -> Union[Sequence, float]:
        """Get the reward at the current timestep.

        Returns
        -------
        Sequence | float
                The reward at the specified timestep. Typically a float but can be a sequence
        """
        return []

    @abstractmethod
    def step(self, action: Any, *args, **kwargs) -> None:
        """Increments the timestep and state of the environment.

        Parameters
        ----------
        action : Any
                The action taken just before initialization of this timestep.
        """
        ...
