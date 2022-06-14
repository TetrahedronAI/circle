"""easyneuron.types provides custom types to be used for machine learning models.

Types
-----
NumpyFloat - any float from Numpy
NumpyInt - any integer from Numpy
Numerical - any value that is or can be converted into a number

Distance - the name of a distance function given by this package
Loss - the name of a loss function given by this package

X_Data - any sequence or number
"""

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
from easyneuron.types.types import (
    Numerical,
    NumpyFloat,
    NumpyInt,
    X_Data,
    ArrayLike
)
