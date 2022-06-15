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
from typing import Iterable
from warnings import warn

from numpy import array, sqrt


def _check_distance_params(x, y, kwargs):
    x = array(x).reshape(1, -1)  # to make them 1D
    y = array(y).reshape(1, -1)

    if x.shape != y.shape and kwargs.get("suppress_warnings") != True:
        warn(
            UserWarning(
                "using sequences which do not contain equivalent numbers of items can result in unexpected results."
            )
        )

    return x,y

def euclidean_distance(x: Iterable, y: Iterable, **kwargs) -> float:
    """Euclidean distance function, returns the distance between x and y
    Pass "supress_warnings=True" to the function to avoid warnings when x and y have different lengths.

    Parameters
    ----------
    x : Iterable
            The first item to be calucated with
    y : Iterable
            The second item to be calucated with

    Returns
    -------
    float
            The euclidean distance
    """
    x, y = _check_distance_params(x, y, kwargs)

    return sqrt(sum((x - y) ** 2 for x, y in zip(x[0], y[0])))


def manhattan_distance(x, y, **kwargs):
    """Manhattan distance function, returns the manhattan distance between x and y
    Pass "supress_warnings=True" to the function to avoid warnings when x and y have different lengths.

    Parameters
    ----------
    x : Iterable
            The first item to be calucated with
    y : Iterable
            The second item to be calucated with

    Returns
    -------
    float
            The manhattan distance
    """
    x, y = _check_distance_params(x, y, kwargs)

    return sum(abs(x - y) for x, y in zip(x[0], y[0]))


distance_functions = {"euclidean": euclidean_distance, "manhattan": manhattan_distance}
