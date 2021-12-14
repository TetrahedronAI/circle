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
from numpy import array, sqrt
from warnings import warn

def euclidean_distance(x, y):
	x = array(x).reshape(1, -1)
	y = array(y).reshape(1, -1)

	if x.shape != y.shape:
		warn(UserWarning("using sequences which do not contain equivalent numbers of items can result in unexpected results."))

	return sqrt(sum((x - y)**2 for x, y in zip(x[0], y[0])))

def manhattan_distance(x, y):
	x = array(x).reshape(1, -1)
	y = array(y).reshape(1, -1)

	if x.shape != y.shape:
		warn(UserWarning("using sequences which do not contain equivalent numbers of items can result in unexpected results."))

	return sum(abs((x - y)) for x, y in zip(x[0], y[0]))

distance_functions = {
    "euclidean": euclidean_distance,
    "manhattan": manhattan_distance
}