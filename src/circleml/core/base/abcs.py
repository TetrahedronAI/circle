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

"""Abstract base classes for use to ensure compatability with pipelines and other modular components.

# Abstract Base Classes
- Module: Abstract base class for pipeline modules
- ModelABC: Abstract base class for models
"""


from abc import ABC, abstractmethod
from typing import Any


class Module(ABC):
    """Abstract base class for pipeline modules."""

    @abstractmethod
    def __call__(self, X: Any) -> Any:
        return X


class SupervisedModelABC(Module):
    """Abstract base class for models."""

    @abstractmethod
    def fit(self, X, y, verbose: bool = False) -> "SupervisedModelABC":
        """Fit the model to the data."""

    @abstractmethod
    def predict(self, X, verbose: bool = False) -> Any:
        """Predict the class of the data."""

    def __call__(self, X: Any, verbose: bool = False) -> Any:
        return self.predict(X, verbose=verbose)

    def fit_predict(self, X, y, verbose: bool = False) -> Any:
        """Fit the model to the data and predict the class of the data."""
        return self.fit(X, y, verbose).predict(X, verbose)


class UnsupervisedModelABC(Module):
    """Abstract base class for models."""

    @abstractmethod
    def fit(self, X, verbose: bool = False) -> "UnsupervisedModelABC":
        """Fit the model to the data."""

    @abstractmethod
    def predict(self, verbose: bool = False) -> Any:
        """Predict the class of the data."""

    def __call__(self, X: Any, verbose: bool = False) -> Any:
        return self.fit_predict(X, verbose=verbose)

    def fit_predict(self, X, verbose: bool = False) -> Any:
        """Fit the model to the data and predict the class of the data."""
        return self.fit(X, verbose=verbose).predict(verbose=verbose)


class TransformationABC(Module):
    """Abstract base class for transformations."""

    @abstractmethod
    def fit(self, X, verbose: bool = False) -> "TransformationABC":
        """Fit the transformation to the data."""

    @abstractmethod
    def transform(self, X, verbose: bool = False) -> Any:
        """Transform the data."""

    @abstractmethod
    def inverse_transform(self, X, verbose: bool = False) -> Any:
        """Inverse transform the data."""

    def __call__(self, X: Any, verbose: bool = False) -> Any:
        return self.transform(X, verbose=verbose)

    def fit_transform(self, X, verbose: bool = False) -> Any:
        """Fit the transformation to the data and transform the data."""
        return self.fit(X, verbose=verbose).transform(X, verbose=verbose)
