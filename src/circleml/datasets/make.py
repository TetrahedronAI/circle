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

"""Internal data generation module. Use circleml.datasets instead."""

import typing as t

import numpy as np

from ..log import check


def make_clusters(
    samples: int = 1000,
    features: int = 2,
    classes: int = 3,
    noise: float = 0.3,
    separate_clusters: bool = True,
    separation: float = 1.5,
) -> t.Tuple[np.ndarray, np.ndarray]:
    """Make a simple cluster dataset for testing.

    Args:
        samples (int, optional): Number of samples. Defaults to 1000.
        features (int, optional): Number of features. Defaults to 2.
        classes (int, optional): Number of classes. Defaults to 3.
        noise (float, optional): Standard deviation of each class. Defaults to 0.3.
        separate_clusters (bool, optional): Whether to separate clusters to prevent overlap. Defaults to True.
        separation (float, optional): How far to separate clusters, if separate_clusters is True.
            1 means no alteration, 2 means double the distance between clusters, 0 means clusters overlap.
            Defaults to 1.5.

    Returns:
        X, y: the dataset
    """
    check(samples > 0, "samples must be positive")
    check(features > 0, "features must be positive")
    check(classes > 0, "classes must be positive")

    # generate cluster centres and separate them if necessary
    points = np.random.randn(classes, features)
    if separate_clusters:
        mean_point = np.mean(points, axis=0)
        for i, p in enumerate(points):
            direction = p - mean_point
            points[i] = p + direction * (separation - 0)

    X = np.zeros((samples, features))
    y = np.zeros(samples, dtype=int)

    # generate samples + noise
    for sample in range(samples):
        label = np.random.randint(classes)
        X[sample, :] = points[label] + np.random.normal(0, noise, size=features)
        y[sample] = label

    return X, y
