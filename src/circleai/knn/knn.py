from __future__ import annotations

from typing import Callable, List, Sized

import numpy as np

from .. import log
from ..core import euclidean_distance
from ..core.base.abcs import ModelABC


class KNNCla(ModelABC):
    """K-Nearest Neighbors Classifier."""

    def __init__(self, k: int=3, distance_func: Callable[[Sized], float]=euclidean_distance) -> None:
        """Creates a new KNN instance.

        Args:
            k (int, optional): K-value (number of neighbours to "vote"). Defaults to 3.
            distance_func (Callable[[Sized], float], optional): Distance function. Defaults to euclidean_distance.
        """        
        self.k: int = k
        self.dist: Callable[[Sized], float] = distance_func

    def fit(self, X: Sized, y: Sized[int]) -> KNNCla:
        """Fit the model to the data.

        Args:
            X (Sized): the samples
            y (Sized[int]): the target labels, encoded as integers
            verbose (bool, optional): whehter to log info. Defaults to False.

        Returns:
            KNN: trained model
        """
        log.check(len(X) == len(y), f"X and y must have the same length,\n\tnot X: {len(X)} and y: {len(y)}")
        self.__train: Sized = X
        self.__labels: Sized = y
        return self

    def predict(self, X: Sized, verbose: bool=False) -> np.ndarray:
        """Predict the class of the data.

        Args:
            X (Sized): an iterable of samples to classify
            verbose (bool, optional): whether to log info. Defaults to False.

        Returns:
            np.ndarray: An array of the predicted classes with shape (-1,)
        """
        logger = log.create_logger(log.debug, verbose=verbose)
        y_pred: List[None] = [self._predict(i, self.k, logger) for i in X]
        return np.array(y_pred)

    def _predict(self, x: Sized, k: int, logger: Callable) -> None:
        """Predict for a single sample. Used recursively in predict(). Not intended to be used by end-users."""
        dists: np.ndarray = np.zeros(len(self.__train))

        for i, sample in enumerate(self.__train):
            dists[i] = self.dist(np.array(x), np.array(sample))

        k_idx: np.ndarray = np.argsort(dists)[:k]
        k_neighbor_labels: List[int] = [self.__labels[i] for i in k_idx]

        counts = {
            label: k_neighbor_labels.count(label) for label in set(k_neighbor_labels)
        }

        counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)

        if len(counts) > 1 and counts[0][1] == counts[1][1]:
            logger(f"Breaking tie, with new k={k-1}")
            return self._predict(x, k - 1, logger)

        return counts[0][0]
