"""K-Means clustering implementation."""

from typing import Dict, Any, Tuple
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .base import BaseClusterer


class KMeansClusterer(BaseClusterer):
    """K-Means clustering with silhouette-based optimization."""

    def __init__(self, n_clusters: int = 3, init: str = 'k-means++', max_iter: int = 300):
        """
        Initialize K-Means clusterer.

        Args:
            n_clusters: Number of clusters
            init: Initialization method ('k-means++', 'random')
            max_iter: Maximum iterations
        """
        super().__init__("K-Means")
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self._model: KMeans = None

    def fit(self, data: np.ndarray) -> np.ndarray:
        """
        Fit K-Means to data.

        Args:
            data: 2D numpy array of shape (n_samples, n_features)

        Returns:
            1D numpy array of cluster labels
        """
        # Ensure n_clusters doesn't exceed sample count
        n_clusters = min(self.n_clusters, len(data))

        self._model = KMeans(
            n_clusters=n_clusters,
            init=self.init,
            max_iter=self.max_iter,
            random_state=42,
            n_init=10
        )

        self.labels_ = self._model.fit_predict(data)
        self.n_clusters_ = n_clusters

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
            'init': self.init,
            'max_iter': self.max_iter,
        }

    def set_params(self, **params) -> None:
        if 'n_clusters' in params:
            self.n_clusters = int(params['n_clusters'])
        if 'init' in params:
            self.init = str(params['init'])
        if 'max_iter' in params:
            self.max_iter = int(params['max_iter'])

    def get_param_specs(self) -> Dict[str, Dict[str, Any]]:
        return {
            'n_clusters': {
                'type': 'int',
                'min': 2,
                'max': 20,
                'default': 3,
                'label': 'Number of Clusters (k)',
            },
            'init': {
                'type': 'choice',
                'choices': ['k-means++', 'random'],
                'default': 'k-means++',
                'label': 'Initialization',
            },
            'max_iter': {
                'type': 'int',
                'min': 100,
                'max': 1000,
                'default': 300,
                'label': 'Max Iterations',
            },
        }

    def optimize(self, data: np.ndarray, param_ranges: Dict[str, Tuple] = None) -> Dict[str, Any]:
        """
        Find optimal k using silhouette score.

        Args:
            data: 2D numpy array
            param_ranges: Optional dict with 'n_clusters' range

        Returns:
            Dictionary with optimal n_clusters
        """
        if param_ranges is None:
            param_ranges = {'n_clusters': (2, 10)}

        k_min, k_max = param_ranges['n_clusters']
        k_max = min(k_max, len(data) - 1)

        best_score = -1
        best_k = self.n_clusters

        for k in range(k_min, k_max + 1):
            kmeans = KMeans(n_clusters=k, init=self.init, max_iter=self.max_iter, random_state=42, n_init=10)
            labels = kmeans.fit_predict(data)

            try:
                score = silhouette_score(data, labels)
                print(f"  k={k}: silhouette={score:.3f}")
                if score > best_score:
                    best_score = score
                    best_k = k
            except Exception:
                pass

        print(f"Optimal k={best_k} with silhouette={best_score:.3f}")

        best_params = {'n_clusters': best_k}
        self.set_params(**best_params)
        return best_params

    def get_cluster_centers(self) -> np.ndarray:
        """Get cluster centers after fitting."""
        if self._model is not None:
            return self._model.cluster_centers_
        return None

    def get_inertia(self) -> float:
        """Get inertia (sum of squared distances to cluster centers)."""
        if self._model is not None:
            return self._model.inertia_
        return None
