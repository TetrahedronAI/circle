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

from math import log1p
from typing import Sequence

from easyneuron.exceptions.exceptions import DimensionsError
from numpy import array, sqrt


def mean_squared_error(x: Sequence, y: Sequence) -> float:
	"""Returns the mean squared error between x and y.

	Parameters
	----------
	x : Sequence
		Any sequence, with equivalent total number of items to y
	y : Sequence
		Any sequence, with equivalent total number of items to x

	Returns
	-------
	float
		The mean squared error.

	Raises
	------
	DimensionsError
		If the total number of items in x and y differ.
	"""
	x = array(x).reshape(1, -1)[0]
	y = array(y).reshape(1, -1)[0]

	if x.shape != y.shape:
		raise DimensionsError(
			"x and y must have the same number of items in it.")

	return sum((i - j)**2 for i, j in zip(x, y)) / len(x)

def mean_squared_log_error(x: Sequence, y: Sequence) -> float:
	"""Returns the mean squared logarithmic error between x and y.

	Parameters
	----------
	x : Sequence
		Any sequence, with equivalent total number of items to y
	y : Sequence
		Any sequence, with equivalent total number of items to x

	Returns
	-------
	float
		The mean squared logarithmic error.

	Raises
	------
	DimensionsError
		If the total number of items in x and y differ.
	"""
	x = array(x).reshape(1, -1)[0]
	y = array(y).reshape(1, -1)[0]

	if x.shape != y.shape:
		raise DimensionsError(
			"x and y must have the same number of items in it.")

	return sum((log1p(i) - log1p(j))**2 for i, j in zip(x, y)) / len(x)

def root_mean_squared_log_error(x: Sequence, y: Sequence) -> float:
	"""Returns the root mean squared logarithmic error between x and y.

	Parameters
	----------
	x : Sequence
		Any sequence, with equivalent total number of items to y
	y : Sequence
		Any sequence, with equivalent total number of items to x

	Returns
	-------
	float
		The root mean squared logarithmic error.

	Raises
	------
	DimensionsError
		If the total number of items in x and y differ.
	"""
	return sqrt(mean_squared_log_error(x, y))
	

def mean_absolute_error(x: Sequence, y: Sequence) -> float:
	"""Returns the mean absolute error between x and y.

	Parameters
	----------
	x : Sequence
		Any sequence, with equivalent total number of items to y
	y : Sequence
		Any sequence, with equivalent total number of items to x

	Returns
	-------
	float
		The mean absolute error.

	Raises
	------
	DimensionsError
		If the total number of items in x and y differ.
	"""
	x = array(x).reshape(1, -1)[0]
	y = array(y).reshape(1, -1)[0]

	if x.shape != y.shape:
		raise DimensionsError(
			"x and y must have the same number of items in it.")

	return sum(abs(i - j) for i, j in zip(x, y)) / len(x)

def root_mean_squared_error(x: Sequence, y: Sequence) -> float:
	"""Returns the root mean squared error between x and y.

	Parameters
	----------
	x : Sequence
		Any sequence, with equivalent total number of items to y
	y : Sequence
		Any sequence, with equivalent total number of items to x

	Returns
	-------
	float
		The root mean squared error.

	Raises
	------
	DimensionsError
		If the total number of items in x and y differ.
	"""
	return sqrt(mean_squared_error(x, y))
