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

"""Model evaluation metrics, such as accuracy and loss functions.

# Metrics
- accuracy
- cluster_accuracy
- mse
- mae
- rmse
- ssr
"""

from .metrics import accuracy, cluster_accuracy, mae, mse, rmse, ssr
