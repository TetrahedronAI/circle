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

import secrets


def secure_random(low: int, high: int) -> int:
    """Generates a random number from low (inclusive) to high (inclusive) using the secrets module.

    DISCLAIMER: we are not security experts this may not be at all secure.

    Parameters
    ----------
    low : int
            The inclusive lower bound
    high : int
            The inclusive upper bound

    Returns
    -------
    int
            The generated number
    """
    return secrets.randbelow(high) + low


def random_with_float_step(start: int, stop: int, step: float) -> float:
    """Generates a random number between a range with a float step.

    Parameters
    ----------
    start : int
            The inclusive lower bound
    stop : int
            The inclusive upper bound
    step : float
            The step of the range

    Returns
    -------
    float
            The generated float
    """
    return secrets.randbelow(int((stop - start) / step)) * step + start
