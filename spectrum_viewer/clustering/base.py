"""Base class for clustering algorithms."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np


class BaseClusterer(ABC):
    """Abstract base class for clustering algorithms."""

    def __init__(self, name: str):
        """
        Initialize the clusterer.

        Args:
            name: Display name of the clustering method
        """
        self.name = name
        self.labels_: Optional[np.ndarray] = None
        self.n_clusters_: int = 0
        self.silhouette_score_: Optional[float] = None

    @abstractmethod
    def fit(self, data: np.ndarray) -> np.ndarray:
        """
        Fit the clustering model to data and return cluster labels.

        Args:
            data: 2D numpy array of shape (n_samples, n_features)

        Returns:
            1D numpy array of cluster labels (-1 for noise/outliers in DBSCAN)
        """
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """
        Get current parameter values.

        Returns:
            Dictionary of parameter names to values
        """
        pass

    @abstractmethod
    def set_params(self, **params) -> None:
        """
        Set parameter values.

        Args:
            **params: Parameter name-value pairs
        """
        pass

    @abstractmethod
    def get_param_specs(self) -> Dict[str, Dict[str, Any]]:
        """
        Get parameter specifications for UI generation.

        Returns:
            Dictionary of parameter specs, each containing:
                - type: 'int', 'float', 'choice'
                - min, max: for numeric types
                - default: default value
                - choices: for 'choice' type
                - label: display label
        """
        pass

    def optimize(self, data: np.ndarray, param_ranges: Dict[str, Tuple]) -> Dict[str, Any]:
        """
        Optimize parameters using silhouette score.

        Args:
            data: 2D numpy array of shape (n_samples, n_features)
            param_ranges: Dictionary mapping parameter names to (min, max) tuples

        Returns:
            Dictionary of optimal parameter values
        """
        from sklearn.metrics import silhouette_score

        best_score = -1
        best_params = self.get_params()

        # Default implementation: simple grid search
        # Subclasses can override for more sophisticated optimization
        return best_params

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance scores if available.

        Returns:
            1D numpy array of importance scores, or None if not available
        """
        return None
