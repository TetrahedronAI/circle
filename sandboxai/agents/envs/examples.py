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

from dataclasses import dataclass
from typing import Any, List, Sequence

from sandboxai.agents.envs._classes import Environment


@dataclass(init=False, eq=True, order=True, unsafe_hash=True, repr=True)
class SimpleLateralMover(Environment):
    """This is an environment that is for initial debugging of new agents, just to check for errors and that you can complete this.

    The aim is to always output 1 and then stay at agent location 99 (move to the right and stop once reached end).
    This just incentivises the agent to climb towards the right.
    """

    def reset(self, *args, **kwargs) -> None:
        """Sets all of the default variables and starts the environment."""
        self.agent_loc = 50
        self.env = [0 for _ in range(100)]

    def get_all_actions(self, *args, **kwargs) -> Sequence:
        """Returns all of the actions possible in this environment.

        Returns
        -------
        list
            All of the possible actions.
        """
        return [-1, 0, 1]

    def get_obs_shape(self, *args, **kwargs) -> Sequence:
        """Returns an example observation.

        Returns
        -------
        List[int]
            The shape - (100,)
        """
        return (100,)

    def _get_env(self) -> List[int]:
        """Returns the environment. Internal. Not for use of user.

        Returns
        -------
        List[int]
            The current environment state.
        """
        self.env = [0 for _ in range(100)]
        self.env[self.agent_loc] = 1
        return self.env

    def get_actions(self, *args, **kwargs) -> Sequence:
        """Returns the actions possible at the current time.

        Returns
        -------
        Sequence
            The possible actions.
        """
        if self.agent_loc == 0:
            return [0, 1]
        elif self.agent_loc == 99:
            return [-1, 0]
        else:
            return [-1, 0, 1]

    def get_obs(self, *args, **kwargs) -> List[int]:
        """Returns the observation of the environment state.

        Returns
        -------
        List[int]
            The environment state.
        """
        return self.env

    def get_reward(self, *args, **kwargs) -> float:
        """Returns the reward

        Returns
        -------
        float
            The reward.
        """
        return self.agent_loc / 5

    def step(self, action: Any, *args, **kwargs) -> None:
        """Increment parameters (step forwards a timestep).

        Parameters
        ----------
        action : Any
            The action chosen, so that the environment can respond and change state.
        """
        self.agent_loc += action
        self._get_env()
