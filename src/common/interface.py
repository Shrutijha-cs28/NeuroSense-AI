"""Common interface definitions for NeuroSenseAI modality branches."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
import numpy as np
import pandas as pd


class ModalityBranch(ABC):
    """Abstract base class for all modality branches."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the modality branch (e.g. 'handwriting', 'voice', 'gait')."""
        pass

    @property
    @abstractmethod
    def disease(self) -> str:
        """Associated disease category ('alzheimer' or 'parkinsons')."""
        pass

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, groups: pd.Series | None = None) -> ModalityBranch:
        """Fit the branch model inside CV folds."""
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return binary class probabilities of shape (n_samples, 2)."""
        pass

    @abstractmethod
    def explain(self, X: pd.DataFrame) -> dict[str, Any]:
        """Compute feature importance or SHAP explanations."""
        pass

    @abstractmethod
    def metrics(self) -> dict[str, Any]:
        """Return cross-validated benchmark performance metrics."""
        pass
