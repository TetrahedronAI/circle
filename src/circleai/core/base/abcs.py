from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

class ModelABC(ABC):
    """Abstract base class for all classifiers."""
    @abstractmethod
    def fit(self, X, y, verbose: bool=False) -> ModelABC:
        """Fit the model to the data."""
        pass

    @abstractmethod
    def predict(self, X, verbose: bool=False) -> Any:
        """Predict the class of the data."""
        pass