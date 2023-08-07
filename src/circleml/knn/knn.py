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

"""Internal KNN module. Use circleml.knn instead."""

import typing as t

import numpy as np

from .. import log
from ..core import euclidean_distance
from ..core.base.abcs import SupervisedModelABC


class KNNCla(SupervisedModelABC):
    """K-Nearest Neighbors Classifier."""

    def __init__(
        self,
        k: int = 3,
        distance_func: t.Callable[[t.Sized], float] = euclidean_distance,
    ) -> None:
        """Creates a new KNN instance.

        Args:
            k (int, optional): K-value (number of neighbours to "vote"). Defaults to 3.
            distance_func (Callable[[Sized], float], optional): Distance function. Defaults to euclidean_distance.
        """
        self.k: int = k
        self.dist: t.Callable[[t.Sized], float] = distance_func
        self.__train: t.Sized = []
        self.__labels: t.Sized = []

    def fit(self, X: t.Sized, y: t.Sized, verbose: bool = False) -> "KNNCla":
        """Fit the model to the data.

        Args:
            X (Sized): the samples
            y (Sized): the target labels, encoded as integers
            verbose (bool, optional): whehter to log info. Defaults to False.

        Returns:
            KNN: trained model
        """
        log.check(
            len(X) == len(y),
            f"X and y must have the same length,\n\tnot X: {len(X)} and y: {len(y)}",
        )

        logger = log.create_logger(log.info, verbose=verbose)
        logger(f"Saving {len(X)} samples to memory")
        self.__train: t.Sized = X
        self.__labels: t.Sized = y
        logger("Training complete")
        return self

    def predict(self, X: t.Sized, verbose: bool = False) -> np.ndarray:
        """Predict the class of the data.

        Args:
            X (Sized): an iterable of samples to classify
            verbose (bool, optional): whether to log info. Defaults to False.

        Returns:
            np.ndarray: An array of the predicted classes with shape (-1,)
        """
        logger = log.create_logger(log.debug, verbose=verbose)
        y_pred: t.List[None] = [self._predict(i, self.k, logger) for i in X]
        return np.array(y_pred)

    def _predict(self, x: t.Sized, k: int, logger: t.Callable) -> None:
        """Predict for a single sample. Used recursively in predict(). Not intended to be used by end-users."""
        dists: np.ndarray = np.zeros(len(self.__train))

        for i, sample in enumerate(self.__train):
            dists[i] = self.dist(np.array(x), np.array(sample))

        k_idx: np.ndarray = np.argsort(dists)[:k]
        k_neighbor_labels: t.List[int] = [self.__labels[i] for i in k_idx]

        counts = {
            label: k_neighbor_labels.count(label) for label in set(k_neighbor_labels)
        }

        counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)

        if len(counts) > 1 and counts[0][1] == counts[1][1]:
            logger(f"Breaking tie, with new k={k-1}")
            return self._predict(x, k - 1, logger)

        return counts[0][0]
