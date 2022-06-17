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
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence
from warnings import warn

from sandboxai._classes import Model
from sandboxai.exceptions.exceptions import UntrainedModelError
from sandboxai.math.distance.distance import distance_functions
from sandboxai.neighbours._classes import _KNN
from sandboxai.types import X_Data, Numerical
from sandboxai.math.distance import Distance
from numpy import array


@dataclass(init=True, repr=True, unsafe_hash=True, eq=True)
class KNNClassifier(_KNN):
	"""
	KNN Classifier is the K-Nearest Neighbours algorithm for classification, implemented in Python.
	"""

	def __init__(self, K: int = 7, distance: Distance = "euclidean") -> None:
		"""Create an instance of the K-Nearest-Neighbours classifier

		Parameters
		----------
		K : int
			The K-Value for the model, by default 7
		distance : Distance[str], optional
			The distance function to use ("euclidean" or "manhattan"), by default "euclidean"
		"""
		if K == 1:
			warn(
				FutureWarning(
					"setting K to 1 can result in bad quality predictions later."
				)
			)  # This is since it'd only analyse the closest one

		self.K: Numerical = K
		self._X: Optional[Sequence] = None

		if distance_functions.get(distance) is None:
			raise ValueError(
				f"the distance function {distance} is not a valid distance. Please use euclidean or manhattan."
			)

		self.distance: Callable = distance_functions[distance]

	def fit(self, X: X_Data, y: Sequence) -> Model:
		"""Train (fit the model) to the given data.

		Parameters
		----------
		X : X_Data
				The samples' data/features
		y : Sequence
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
				f"parameters X and y should have the same length.\nNot...\n\tX: {len(X)}\n\ty: {len(y)}."
			)  # So that samples align with labels

		X = array(X)
		y = array(y)

		if len(X.shape) < 2:
			raise ValueError(
				f"the parameter passed for X should have 2 or more dimensions, not {len(X.shape)} dimensions.\nUsing <arrayName>.reshape(-1, 1) on your X parameter may solve this."
			)  # Ensuring that the following algorithms work

		self._X = [(s, X[sI]) for sI, s in enumerate(y)]

		return self

	def predict(self, X: X_Data) -> Sequence:
		"""Generate predictions from the kNN model.

		Parameters
		----------
		X : X_Data
			The samples to predict from.

		Returns
		-------
		Sequence
			The predicted labels.

		Raises
		------
		ValueError
			If the X shape has less than 2 dimensions.
		"""
		X = array(X)
		if len(X.shape) < 2:
			raise ValueError(
				f"the parameter passed for X should have 2 or more dimensions, not {len(X.shape)} dimensions.\nUsing <arrayName>.reshape(-1, 1) on your X parameter may solve this."
			)

		return [self._choose_label(sample, self.K) for sample in X]

	def _choose_label(self, sample: Sequence, K: int) -> Any:
		"""Choose the label from a sample.

		Parameters
		----------
		sample : Sequence
			The sample to use.

		Returns
		-------
		Any
			The label (of any type).
		"""
		if self._X is None:
			raise UntrainedModelError("model is not trained.")

		# sort distances for each sample from smallest to largest
		distances = self.new_method(sample, K)

		votes = {}
		for i in [
				j[1] for j in distances
		]:  # iterate over the labels in the sorted distances
			if i in votes:  # checks if in the keys
				votes[i] += 1
			else:
				votes[i] = 1  # adds as new key

		choices = []  # values with the most votes

		m_votes = 0
		for i, j in votes.items():
			if j > m_votes:
				m_votes = i
				choices = [
					i
				]  # this has more votes than any other, so no ties yet - replace whole list
			elif j == m_votes:
				# tie. add to the list, rather than replace it
				choices.append(i)

		if len(choices) > 1:
			return self._choose_label(
				sample, K - 1
			)  # recursively avoid ties by decrementing K

		return choices[0]

	def new_method(self, sample, K):
		return sorted(
		    [(self.distance(sample, i), label) for label, i in self._X],
		    key=lambda item: item[0],
		)[:K]  # distance, label
