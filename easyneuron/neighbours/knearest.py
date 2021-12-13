from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple
from warnings import warn

from easyneuron._classes import Model
from easyneuron.math.distance.distance import distance_functions, euclidean_distance, manhattan_distance
from easyneuron.neighbours._classes import _KNN
from easyneuron.types import X_Data
from easyneuron.types.types import Distance, Int, Numerical
from numpy import array


@dataclass(init=True, repr=True, unsafe_hash=True, eq=True)
class KNNClassifier(_KNN):
    """
    KNN Classifier is the K-Nearest Neighbours algorithm for classification, implemented in Python.
    """

    def __init__(self, K: Int =  7, distance: Distance = "euclidean") -> None:
        """Create an instance of the K-Nearest-Neighbours classifier

        Parameters
        ----------
        K : Int
            The K-Value for the model, by default 7
        distance : Distance[str], optional
            The distance function to use ("euclidean" or "manhattan"), by default "euclidean"
        """
        if K == 1:
            warn(FutureWarning(
                "setting K to 1 can result in bad quality predictions later.")) # This is since it'd only analyse the closest one

        self.K: Numerical = K
        self.distance: Callable = distance_functions[distance]

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
                f"parameters X and y should have the same length.\nNot...\n\tX: {len(X)}\n\ty: {len(y)}.") # So that samples align with labels

        X = array(X)
        y = array(y)

        if len(X.shape) < 2:
            raise ValueError(
                f"the parameter passed for X should have 2 or more dimensions, not {len(X.shape)} dimensions.\nUsing <arrayName>.reshape(-1, 1) on your X parameter may solve this.") # Ensuring that the following algorithms work

        self._samples = {s: X[sI] for sI, s in enumerate(y)}

    def predict(self, X: X_Data) -> Sequence[Any]:
        """Generate predictions from the kNN model.

        Parameters
        ----------
        X : X_Data
            The samples to predict from.

        Returns
        -------
        Sequence[Any]
            The predicted labels.

        Raises
        ------
        ValueError
            If the X shape has less than 2 dimensions.
        """
        if len(X.shape) < 2:
            raise ValueError(
                f"the parameter passed for X should have 2 or more dimensions, not {len(X.shape)} dimensions.\nUsing <arrayName>.reshape(-1, 1) on your X parameter may solve this.")

        return [self._choose_label(X, sample) for sample in X]

    def _choose_label(self, sample: Sequence[Any]) -> Any:
        """Choose the label from a sample.

        Parameters
        ----------
        sample : Sequence[Any]
            The sample to use.

        Returns
        -------
        Any
            The label (of any type).
        """
        # Calculate all of the distances
        distances_k = {sK: self.distance(sample, s) for sK, s in self._samples.items()}
        distances_k = self._get_k_distances(distances_k)

        # Vote on all of the choices (by how many of the K are of each)
        votes = {}
        for label in distances_k.keys():
            if label not in list(votes.keys()):
                votes[label] = 1
            else:
                votes[label] += 1

        most_votes = max(list(votes.keys()))
        choices = [j for i, j in votes.items() if i == most_votes] # Get all the voted choices (even if there are multiple with the same key)

        # No tie
        if len(choices) <= 1:
            return choices[0]

        # Recursively search for ties
        self.K -= 1
        return self._choose_label(sample)

    def _get_k_distances(self, distances) -> Dict[Numerical, Any]:
        # Returns items 0, 1 ... K values of a dictionary of the distances
        return dict(list(sorted(distances.items(), key=lambda x:x[1]))[:self.K])
