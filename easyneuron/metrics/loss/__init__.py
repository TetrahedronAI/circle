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

from typing import Callable, Literal, Union

from easyneuron.metrics.loss.meanerrors import (mean_absolute_error,
                                                mean_squared_error,
                                                mean_squared_log_error,
                                                root_mean_squared_error,
                                                root_mean_squared_log_error)

losses = {  # for names as strings
    "mse": mean_squared_error,
    "rmse": root_mean_squared_error,
    "msle": mean_squared_log_error,
    "rmsle": root_mean_squared_log_error,
    "mae": mean_absolute_error,
    "mean_squared_error": mean_squared_error,
    "root_mean_squared_error": root_mean_squared_error,
    "mean_squared_log_error": mean_squared_log_error,
    "mean_squared_logarithmic_error": mean_squared_log_error,
    "root_mean_log_error": root_mean_squared_log_error,
    "mean_absolute": mean_squared_error,
}

Loss = Union[
    Literal["mse", "mae", "mean_squared_error",
            "mean_absolute_error"], Callable
]
