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

from numpy import array

from easyneuron.exceptions.exceptions import DimensionsError


def mean_squared_error(x, y) -> float:
	x = array(x).reshape(1, -1)[0]
	y = array(y).reshape(1, -1)[0]

	if x.shape != y.shape:
		raise DimensionsError(
			"x and y must have the same number of items in it.")

	return sum((i - j)**2 for i, j in zip(x, y)) / len(x)

def mean_absolute_error(x, y) -> float:
	x = array(x).reshape(1, -1)[0]
	y = array(y).reshape(1, -1)[0]

	if x.shape != y.shape:
		raise DimensionsError(
			"x and y must have the same number of items in it.")

	return sum(abs(i - j) for i, j in zip(x, y)) / len(x)
