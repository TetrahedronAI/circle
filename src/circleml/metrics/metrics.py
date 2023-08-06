# Copyright 2023 CircleML GitHub Authors. All Rights Reserved.
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
import numpy as np
from ..core import check_len

def accuracy(y_true, y_pred) -> np.number:
    """Returns the accuracy score.

    Args:
        y_true : the real labels
        y_pred : the predicted labels

    Returns:
        number: accuracy from 0 to 1
    """
    check_len(y_true, y_pred)
    return np.mean(np.array(y_true) == np.array(y_pred))

def mse(y_true, y_pred) -> np.number:
    """Returns the mean squared error.

    Args:
        y_true: the real values
        y_pred: the predicted values

    Returns:
        number: MSE value
    """
    check_len(y_true, y_pred)
    return np.mean((np.array(y_true) - np.array(y_pred)) ** 2)

def rmse(y_true, y_pred) -> np.number:
    """Returns the root mean squared error.

    Args:
        y_true: the real values
        y_pred: the predicted values

    Returns:
        number: RMSE value
    """
    return np.sqrt(mse(y_true, y_pred))

def mae(y_true, y_pred) -> np.number:
    """Returns the mean absolute error.

    Args:
        y_true: the real values
        y_pred: the predicted values

    Returns:
        number: MAE value
    """
    check_len(y_true, y_pred)
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

def ssr(y_true, y_pred) -> np.number:
    """Returns the sum of squared residuals.

    Args:
        y_true: the real values
        y_pred: the predicted values

    Returns:
        number: SSR value
    """
    check_len(y_true, y_pred)
    return np.sum((np.array(y_true) - np.array(y_pred)) ** 2)
