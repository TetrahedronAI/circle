from typing import Any, Sequence

from easyneuron._classes import Model
from easyneuron.neighbours._classes import _KNN
from easyneuron.types import X_Data
from numpy import array


class KNNClassifier(_KNN):
	def fit(self, X: X_Data, y: Sequence[Any]) -> Model:
		
		if len(X) != len(y):
			raise ValueError(f"parameters X and y should have the same length. Not\nX: {len(X)}\n\ty: {len(y)}.")
		
		X = array(X)
		y = array(y)

		if len(X.shape) not in [2, 3]:
			raise ValueError(f"the parameter passed for X should have 2 or 3 dimensions, not {len(X.shape)}.")

		self.samples = {s: X[sI] for sI, s in enumerate(y)}
