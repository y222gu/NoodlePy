"""Decision Tree based clustering implementation."""

from typing import Dict, Any, Tuple, Optional
import numpy as np
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import silhouette_score

from .base import BaseClusterer


class DecisionTreeClusterer(BaseClusterer):
    """
    Decision Tree clustering using K-Means initialization.

    This method:
    1. Clusters data with K-Means to get initial labels
    2. Trains a Decision Tree classifier on these labels
    3. Provides interpretable feature importance (which wavenumbers matter)
    """

    def __init__(self, n_clusters: int = 3, max_depth: int = 5, min_samples_split: int = 2):
        """
        Initialize Decision Tree clusterer.

        Args:
            n_clusters: Number of clusters (for initial K-Means)
            max_depth: Maximum depth of the decision tree
            min_samples_split: Minimum samples required to split a node
        """
        super().__init__("Decision Tree")
        self.n_clusters = n_clusters
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self._kmeans: KMeans = None
        self._tree: DecisionTreeClassifier = None
        self._feature_importance: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray) -> np.ndarray:
        """
        Fit Decision Tree clustering to data.

        Args:
            data: 2D numpy array of shape (n_samples, n_features)

        Returns:
            1D numpy array of cluster labels
        """
        # Ensure n_clusters doesn't exceed sample count
        n_clusters = min(self.n_clusters, len(data))

        # Step 1: Cluster with K-Means
        self._kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans_labels = self._kmeans.fit_predict(data)

        # Step 2: Train Decision Tree on K-Means labels
        self._tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=42
        )
        self._tree.fit(data, kmeans_labels)

        # Use tree predictions as final labels (may differ slightly from K-Means)
        self.labels_ = self._tree.predict(data)
        self.n_clusters_ = len(set(self.labels_))

        # Store feature importance
        self._feature_importance = self._tree.feature_importances_

        # Calculate silhouette score
        self.silhouette_score_ = None
        if self.n_clusters_ > 1 and len(data) > self.n_clusters_:
            try:
                self.silhouette_score_ = silhouette_score(data, self.labels_)
            except Exception:
                pass

        return self.labels_

    def get_params(self) -> Dict[str, Any]:
        return {
            'n_clusters': self.n_clusters,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
        }

    def set_params(self, **params) -> None:
        if 'n_clusters' in params:
            self.n_clusters = int(params['n_clusters'])
        if 'max_depth' in params:
            self.max_depth = int(params['max_depth'])
        if 'min_samples_split' in params:
            self.min_samples_split = int(params['min_samples_split'])

    def get_param_specs(self) -> Dict[str, Dict[str, Any]]:
        return {
            'n_clusters': {
                'type': 'int',
                'min': 2,
                'max': 20,
                'default': 3,
                'label': 'Number of Clusters',
            },
            'max_depth': {
                'type': 'int',
                'min': 1,
                'max': 20,
                'default': 5,
                'label': 'Max Tree Depth',
            },
            'min_samples_split': {
                'type': 'int',
                'min': 2,
                'max': 20,
                'default': 2,
                'label': 'Min Samples to Split',
            },
        }

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance scores from the decision tree.

        Returns:
            1D numpy array of importance scores (one per wavenumber)
        """
        return self._feature_importance

    def get_top_features(self, n: int = 10, wavenumbers: np.ndarray = None) -> list:
        """
        Get indices (and optionally wavenumber values) of top important features.

        Args:
            n: Number of top features to return
            wavenumbers: Optional wavenumber array to map indices to values

        Returns:
            List of (index, importance) or (wavenumber, importance) tuples
        """
        if self._feature_importance is None:
            return []

        # Get indices sorted by importance
        sorted_indices = np.argsort(self._feature_importance)[::-1][:n]

        results = []
        for idx in sorted_indices:
            importance = self._feature_importance[idx]
            if wavenumbers is not None and idx < len(wavenumbers):
                results.append((wavenumbers[idx], importance))
            else:
                results.append((idx, importance))

        return results

    def optimize(self, data: np.ndarray, param_ranges: Dict[str, Tuple] = None) -> Dict[str, Any]:
        """
        Optimize parameters using silhouette score.

        Args:
            data: 2D numpy array
            param_ranges: Optional parameter ranges

        Returns:
            Dictionary of optimal parameters
        """
        if param_ranges is None:
            param_ranges = {
                'n_clusters': (2, 10),
                'max_depth': (3, 10),
            }

        k_min, k_max = param_ranges.get('n_clusters', (2, 10))
        k_max = min(k_max, len(data) - 1)

        depth_min, depth_max = param_ranges.get('max_depth', (3, 10))

        best_score = -1
        best_params = self.get_params()

        for k in range(k_min, k_max + 1):
            for depth in range(depth_min, depth_max + 1):
                # Quick K-Means clustering
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans_labels = kmeans.fit_predict(data)

                # Train tree
                tree = DecisionTreeClassifier(
                    max_depth=depth,
                    min_samples_split=self.min_samples_split,
                    random_state=42
                )
                tree.fit(data, kmeans_labels)
                labels = tree.predict(data)

                try:
                    score = silhouette_score(data, labels)
                    if score > best_score:
                        best_score = score
                        best_params = {'n_clusters': k, 'max_depth': depth}
                except Exception:
                    pass

        print(f"Optimal: n_clusters={best_params['n_clusters']}, "
              f"max_depth={best_params['max_depth']}, silhouette={best_score:.3f}")

        self.set_params(**best_params)
        return best_params
