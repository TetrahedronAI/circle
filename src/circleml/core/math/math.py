# Copyright 2023 CircleML GitHub Authors. All Rights Reserved.
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

"""Internal math module. Use circleml.core.math instead."""

import numpy as np
from numpy import linalg

from ...log import check_err


def euclidean_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """Returns the euclidean distance between two vectors.

    Args:
        x1 (np.ndarray): first input
        x2 (np.ndarray): second input

    Returns:
        float: Euclidean distance between x1 and x2
    """
    return linalg.norm(x1 - x2)


def manhattan_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """Returns the manhattan distance between two vectors.

    Args:
        x1 (np.ndarray): first input
        x2 (np.ndarray): second input

    Returns:
        float: Manhattan distance between x1 and x2
    """
    return np.sum(np.abs(x1 - x2))


def hamming_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """Returns the hamming distance between two vectors.

    Args:
        x1 (np.ndarray): first input
        x2 (np.ndarray): second input

    Returns:
        float: Hamming distance between x1 and x2
    """
    check_err(
        isinstance(x1, np.ndarray) and isinstance(x2, np.ndarray),
        "Values must be numpy arrays",
        TypeError,
    )
    return np.sum(x1 != x2)
