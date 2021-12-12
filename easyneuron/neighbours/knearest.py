from dataclasses import dataclass
from typing import Any, Optional, Sequence

from easyneuron._classes import Model
from easyneuron.neighbours._classes import _KNN
from easyneuron.types import X_Data
from easyneuron.types.types import Distance, Int
from numpy import array
from warnings import warn


@dataclass(init=True, repr=True, unsafe_hash=True, eq=True)
class KNNClassifier(_KNN):

    """KNN Classifier is the K-Nearest Neighbours algorithm for classification, implemented in Python."""

    def __init__(self, K: Optional[Int], distance: Distance = "euclidean") -> None:
        """Create an instance of the K-Nearest-Neighbours classifier

        Parameters
        ----------
        K : Optional[Int]
            The K-Value for the model
        distance : Distance[str], optional
            The distance function to use ("euclidean" or "manhattan"), by default "euclidean"
        """
        if K is None:
            self.K = 7
        elif K == 1:
            warn(FutureWarning(
                "setting K to 1 can result in bad quality predictions later."))

        self.K = K
        self.distance = distance

    def fit(self, X: X_Data, y: Sequence[Any]) -> Model:
        
        """Train (fit the model) to the given data.

        Parameters
        ----------
        X : X_Data
                The samples' data/features
        y : Sequence[Any]
                The labels for each sample

        Returns
        -------
        Model
                The trained model

        Raises
        ------
        ValueError
                If X has a different number of samples to y. They must have equivalent lengths.
        ValueError
                If the data for X has less than 2 dimensions.
        """
        if len(X) != len(y):
            raise ValueError(
                f"parameters X and y should have the same length. Not\nX: {len(X)}\n\ty: {len(y)}.")

        X = array(X)
        y = array(y)

        if len(X.shape) < 2:
            raise ValueError(
                f"the parameter passed for X should have more than 2 dimensions, not {len(X.shape)} dimensions.\nUsing X.reshape(-1, 1) may solve this.")

        self.samples = {s: X[sI] for sI, s in enumerate(y)}

    def predict(self, X: X_Data) -> Sequence[Any]:
        pass
