"""DBSCAN clustering implementation."""

from typing import Dict, Any, Tuple
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from .base import BaseClusterer


class DBSCANClusterer(BaseClusterer):
    """DBSCAN clustering with iterative outlier detection."""

    def __init__(self, eps: float = 0.5, min_samples: int = 5, n_iterations: int = 1):
        """
        Initialize DBSCAN clusterer.

        Args:
            eps: Maximum distance between samples in a neighborhood
            min_samples: Minimum samples in a neighborhood for core points
            n_iterations: Number of iterative outlier detection passes
        """
        super().__init__("DBSCAN")
        self.eps = eps
        self.min_samples = min_samples
        self.n_iterations = n_iterations
        self.outlier_indices_: set = set()

    def fit(self, data: np.ndarray) -> np.ndarray:
        """
        Fit DBSCAN to data.

        Args:
            data: 2D numpy array of shape (n_samples, n_features)

        Returns:
            1D numpy array of cluster labels (-1 for outliers)
        """
        # Apply PCA for dimensionality reduction if needed
        if data.shape[1] > 10:
            pca_data = PCA(n_components=min(10, data.shape[0], data.shape[1])).fit_transform(data)
        else:
            pca_data = data

        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.labels_ = dbscan.fit_predict(pca_data)

        # Count clusters (excluding noise label -1)
        unique_labels = set(self.labels_)
        self.n_clusters_ = len(unique_labels - {-1})

        # Calculate silhouette score if valid
        self.silhouette_score_ = None
        if self.n_clusters_ > 1:
            # Only use non-noise points for silhouette
            non_noise_mask = self.labels_ != -1
            if non_noise_mask.sum() > 1:
                try:
                    self.silhouette_score_ = silhouette_score(
                        pca_data[non_noise_mask],
                        self.labels_[non_noise_mask]
                    )
                except Exception:
                    pass

        # Track outliers
        self.outlier_indices_ = {i for i, label in enumerate(self.labels_) if label == -1}

        return self.labels_

    def fit_iterative(self, data: np.ndarray) -> np.ndarray:
        """
        Fit DBSCAN with iterative outlier detection using different PCA ranges.

        Args:
            data: 2D numpy array of shape (n_samples, n_features)

        Returns:
            1D numpy array of cluster labels
        """
        self.outlier_indices_ = set()

        for iteration in range(self.n_iterations):
            print(f"Running PCA + DBSCAN Outlier Detection - Iteration {iteration+1}/{self.n_iterations}")

            pca_data = PCA().fit_transform(data)

            # Use different PCA components for each iteration
            if iteration == 0:
                pca_subset = pca_data[:, :2]
            else:
                pcstart, pcend = 3, 5
                pcend = min(pcend, pca_data.shape[1])
                pcstart = min(pcstart, pcend - 1)
                pca_subset = pca_data[:, pcstart:pcend]

            dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
            labels = dbscan.fit_predict(pca_subset)

            iter_outliers = {idx for idx, label in enumerate(labels) if label == -1}
            self.outlier_indices_.update(iter_outliers)
            print(f"Iteration {iteration+1} found {len(iter_outliers)} outliers.")

        print(f"Total outliers found: {len(self.outlier_indices_)}")

        # Create final labels array
        self.labels_ = np.zeros(len(data), dtype=int)
        for idx in self.outlier_indices_:
            self.labels_[idx] = -1

        return self.labels_

    def get_params(self) -> Dict[str, Any]:
        return {
            'eps': self.eps,
            'min_samples': self.min_samples,
            'n_iterations': self.n_iterations,
        }

    def set_params(self, **params) -> None:
        if 'eps' in params:
            self.eps = float(params['eps'])
        if 'min_samples' in params:
            self.min_samples = int(params['min_samples'])
        if 'n_iterations' in params:
            self.n_iterations = int(params['n_iterations'])

    def get_param_specs(self) -> Dict[str, Dict[str, Any]]:
        return {
            'eps': {
                'type': 'float',
                'min': 0.01,
                'max': 10.0,
                'default': 0.5,
                'step': 0.1,
                'label': 'Epsilon (eps)',
            },
            'min_samples': {
                'type': 'int',
                'min': 2,
                'max': 50,
                'default': 5,
                'label': 'Min Samples',
            },
            'n_iterations': {
                'type': 'int',
                'min': 1,
                'max': 10,
                'default': 1,
                'label': 'Outlier Detection Iterations',
            },
        }

    def optimize(self, data: np.ndarray, param_ranges: Dict[str, Tuple] = None) -> Dict[str, Any]:
        """
        Optimize DBSCAN parameters using silhouette score.

        Args:
            data: 2D numpy array
            param_ranges: Optional dict with 'eps' and 'min_samples' ranges

        Returns:
            Dictionary of optimal parameters
        """
        if param_ranges is None:
            param_ranges = {
                'eps': (0.1, 2.0),
                'min_samples': (2, 20),
            }

        # Apply PCA for optimization
        pca_data = PCA(n_components=min(10, data.shape[0], data.shape[1])).fit_transform(data)

        eps_range = np.linspace(param_ranges['eps'][0], param_ranges['eps'][1], 10)
        min_samples_range = range(param_ranges['min_samples'][0], param_ranges['min_samples'][1] + 1, 2)

        best_score = -1
        best_params = self.get_params()

        for eps in eps_range:
            for min_samples in min_samples_range:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(pca_data)

                # Need at least 2 clusters and no noise for valid silhouette
                unique_labels = set(labels)
                if len(unique_labels) > 1 and -1 not in unique_labels:
                    try:
                        score = silhouette_score(pca_data, labels)
                        if score > best_score:
                            best_score = score
                            best_params = {'eps': eps, 'min_samples': min_samples}
                    except Exception:
                        pass

        print(f"Optimized DBSCAN: eps={best_params['eps']:.2f}, "
              f"min_samples={best_params['min_samples']}, silhouette={best_score:.3f}")

        self.set_params(**best_params)
        return best_params
