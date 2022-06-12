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

from __future__ import (
    annotations,
)  # stops errors around the returning of a Model from a Model subclass instance

from abc import ABC, abstractmethod
from typing import Any


class Model(ABC):
    @abstractmethod
    def fit(self, X, *args, **kwargs) -> Model:
        """Fit the model. This must be overwritten by the subclass.

        Parameters
        ----------
        X : Any
                The data samples

        Returns
        -------
        Model
                The fitted version of itself.
        """
        ...

    @abstractmethod
    def predict(self, X, *args, **kwargs) -> Any:
        """Predict from given data X. This must be overwritten by the subclass

        Parameters
        ----------
        X : Any
                The data samples

        Returns
        -------
        Any
                The predictions of the model.
        """
        ...
