"""Abstract base for all model wrappers."""
from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    @abstractmethod
    def fit(self, X, y, X_val=None, y_val=None, **kwargs):
        ...

    @abstractmethod
    def predict_proba(self, X) -> np.ndarray:
        """Return (N,) float array of positive-class probabilities."""
        ...

    @abstractmethod
    def save(self, path: str):
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseModel":
        ...
