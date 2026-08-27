"""Clustering controls tab widget with multi-method support."""

import numpy as np
from typing import Dict, Any
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QComboBox,
    QPushButton, QDoubleSpinBox, QSpinBox, QStackedWidget, QFormLayout
)
from PyQt5.QtCore import pyqtSignal

from ..clustering import BaseClusterer, DBSCANClusterer, KMeansClusterer, DecisionTreeClusterer


class ClusteringTab(QWidget):
    """Tab widget for clustering controls with multiple methods."""

    # Signals
    clustering_complete = pyqtSignal(np.ndarray)  # Emits cluster labels
    outliers_found = pyqtSignal(set)  # Emits outlier indices

    def __init__(self, parent=None):
        super().__init__(parent)

        self.clusterers: Dict[str, BaseClusterer] = {
            'K-Means': KMeansClusterer(),
            'DBSCAN': DBSCANClusterer(),
            'Decision Tree': DecisionTreeClusterer(),
        }
        self.current_clusterer: BaseClusterer = self.clusterers['K-Means']
        self.current_labels: np.ndarray = None
        self._data_matrix: np.ndarray = None
        self._embeddings_matrix: np.ndarray = None

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        group = QGroupBox("Clustering")
        group_layout = QVBoxLayout(group)
        group_layout.setSpacing(8)

        # Data source selection
        group_layout.addWidget(QLabel("Data Source:"))
        self.data_source_combo = QComboBox()
        self.data_source_combo.addItems(['Preprocessed Spectra', 'Dim Reduced Embeddings'])
        group_layout.addWidget(self.data_source_combo)

        # Method selection
        group_layout.addWidget(QLabel("Method:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(list(self.clusterers.keys()))
        self.method_combo.currentTextChanged.connect(self._on_method_changed)
        group_layout.addWidget(self.method_combo)

        # Stacked widget for method-specific parameters
        self.params_stack = QStackedWidget()
        self._create_param_widgets()
        group_layout.addWidget(self.params_stack)

        # Action buttons
        btn_row = QHBoxLayout()
        self.btn_run = QPushButton("Run Clustering")
        self.btn_run.clicked.connect(self._run_clustering)
        btn_row.addWidget(self.btn_run)

        self.btn_optimize = QPushButton("Optimize")
        self.btn_optimize.clicked.connect(self._optimize_params)
        btn_row.addWidget(self.btn_optimize)
        group_layout.addLayout(btn_row)

        # Results display
        self.results_label = QLabel("Clusters: -  Silhouette: -")
        self.results_label.setStyleSheet("font-size: 10px; color: #aaaaaa;")
        group_layout.addWidget(self.results_label)

        # DBSCAN-specific outlier selection
        self.btn_select_outliers = QPushButton("Select All Outliers")
        self.btn_select_outliers.setEnabled(False)
        self.btn_select_outliers.clicked.connect(self._select_outliers)
        group_layout.addWidget(self.btn_select_outliers)

        layout.addWidget(group)
        layout.addStretch(1)

        # Initial state
        self._on_method_changed('K-Means')

    def _create_param_widgets(self):
        """Create parameter widgets for each clustering method."""
        # K-Means parameters
        kmeans_widget = QWidget()
        kmeans_layout = QFormLayout(kmeans_widget)
        kmeans_layout.setContentsMargins(0, 5, 0, 5)

        self.kmeans_k_spin = QSpinBox()
        self.kmeans_k_spin.setRange(2, 20)
        self.kmeans_k_spin.setValue(3)
        kmeans_layout.addRow("Clusters (k):", self.kmeans_k_spin)

        self.kmeans_init_combo = QComboBox()
        self.kmeans_init_combo.addItems(['k-means++', 'random'])
        kmeans_layout.addRow("Init:", self.kmeans_init_combo)

        self.kmeans_iter_spin = QSpinBox()
        self.kmeans_iter_spin.setRange(100, 1000)
        self.kmeans_iter_spin.setValue(300)
        kmeans_layout.addRow("Max Iter:", self.kmeans_iter_spin)

        self.params_stack.addWidget(kmeans_widget)

        # DBSCAN parameters
        dbscan_widget = QWidget()
        dbscan_layout = QFormLayout(dbscan_widget)
        dbscan_layout.setContentsMargins(0, 5, 0, 5)

        self.dbscan_eps_spin = QDoubleSpinBox()
        self.dbscan_eps_spin.setRange(0.01, 10.0)
        self.dbscan_eps_spin.setSingleStep(0.1)
        self.dbscan_eps_spin.setValue(0.5)
        dbscan_layout.addRow("Epsilon:", self.dbscan_eps_spin)

        self.dbscan_min_spin = QSpinBox()
        self.dbscan_min_spin.setRange(2, 50)
        self.dbscan_min_spin.setValue(5)
        dbscan_layout.addRow("Min Samples:", self.dbscan_min_spin)

        self.dbscan_iter_spin = QSpinBox()
        self.dbscan_iter_spin.setRange(1, 10)
        self.dbscan_iter_spin.setValue(1)
        dbscan_layout.addRow("Outlier Iters:", self.dbscan_iter_spin)

        # Optimization ranges
        eps_range_row = QHBoxLayout()
        self.dbscan_eps_min = QDoubleSpinBox()
        self.dbscan_eps_min.setRange(0.01, 10.0)
        self.dbscan_eps_min.setValue(0.1)
        self.dbscan_eps_max = QDoubleSpinBox()
        self.dbscan_eps_max.setRange(0.01, 10.0)
        self.dbscan_eps_max.setValue(2.0)
        eps_range_row.addWidget(self.dbscan_eps_min)
        eps_range_row.addWidget(QLabel("-"))
        eps_range_row.addWidget(self.dbscan_eps_max)
        dbscan_layout.addRow("Eps Range:", eps_range_row)

        self.params_stack.addWidget(dbscan_widget)

        # Decision Tree parameters
        dt_widget = QWidget()
        dt_layout = QFormLayout(dt_widget)
        dt_layout.setContentsMargins(0, 5, 0, 5)

        self.dt_k_spin = QSpinBox()
        self.dt_k_spin.setRange(2, 20)
        self.dt_k_spin.setValue(3)
        dt_layout.addRow("Clusters:", self.dt_k_spin)

        self.dt_depth_spin = QSpinBox()
        self.dt_depth_spin.setRange(1, 20)
        self.dt_depth_spin.setValue(5)
        dt_layout.addRow("Max Depth:", self.dt_depth_spin)

        self.dt_min_split_spin = QSpinBox()
        self.dt_min_split_spin.setRange(2, 20)
        self.dt_min_split_spin.setValue(2)
        dt_layout.addRow("Min Split:", self.dt_min_split_spin)

        self.params_stack.addWidget(dt_widget)

    def _on_method_changed(self, method_name: str):
        """Handle method selection change."""
        self.current_clusterer = self.clusterers.get(method_name)

        # Update stacked widget
        index = list(self.clusterers.keys()).index(method_name)
        self.params_stack.setCurrentIndex(index)

        # Show/hide outlier button based on method
        self.btn_select_outliers.setVisible(method_name == 'DBSCAN')
        self.btn_select_outliers.setEnabled(False)

    def set_data(self, data_matrix: np.ndarray):
        """Set the data matrix for clustering."""
        self._data_matrix = data_matrix

    def set_embedding_data(self, matrix: np.ndarray):
        """Set the embedding matrix for clustering on dim-reduced data."""
        self._embeddings_matrix = matrix

    def _get_active_data(self) -> np.ndarray:
        """Return the data matrix based on the selected data source."""
        if self.data_source_combo.currentText() == 'Dim Reduced Embeddings':
            if self._embeddings_matrix is not None and len(self._embeddings_matrix) > 0:
                return self._embeddings_matrix
            print("No embedding data available, falling back to spectra")
        return self._data_matrix

    def _get_params_from_ui(self) -> Dict[str, Any]:
        """Get current parameters from UI for the selected method."""
        method = self.method_combo.currentText()

        if method == 'K-Means':
            return {
                'n_clusters': self.kmeans_k_spin.value(),
                'init': self.kmeans_init_combo.currentText(),
                'max_iter': self.kmeans_iter_spin.value(),
            }
        elif method == 'DBSCAN':
            return {
                'eps': self.dbscan_eps_spin.value(),
                'min_samples': self.dbscan_min_spin.value(),
                'n_iterations': self.dbscan_iter_spin.value(),
            }
        elif method == 'Decision Tree':
            return {
                'n_clusters': self.dt_k_spin.value(),
                'max_depth': self.dt_depth_spin.value(),
                'min_samples_split': self.dt_min_split_spin.value(),
            }

        return {}

    def _update_ui_from_params(self, params: Dict[str, Any]):
        """Update UI from parameters."""
        method = self.method_combo.currentText()

        if method == 'K-Means':
            if 'n_clusters' in params:
                self.kmeans_k_spin.setValue(params['n_clusters'])
        elif method == 'DBSCAN':
            if 'eps' in params:
                self.dbscan_eps_spin.setValue(params['eps'])
            if 'min_samples' in params:
                self.dbscan_min_spin.setValue(params['min_samples'])
        elif method == 'Decision Tree':
            if 'n_clusters' in params:
                self.dt_k_spin.setValue(params['n_clusters'])
            if 'max_depth' in params:
                self.dt_depth_spin.setValue(params['max_depth'])

    def _run_clustering(self):
        """Run clustering with current parameters."""
        data = self._get_active_data()
        if data is None or len(data) == 0:
            print("No data for clustering")
            return

        params = self._get_params_from_ui()
        self.current_clusterer.set_params(**params)

        method = self.method_combo.currentText()

        if method == 'DBSCAN' and self.dbscan_iter_spin.value() > 1:
            # Use iterative outlier detection
            labels = self.current_clusterer.fit_iterative(data)
        else:
            labels = self.current_clusterer.fit(data)

        self.current_labels = labels
        self._update_results_display()

        # Emit signal
        self.clustering_complete.emit(labels)

        # Enable outlier selection for DBSCAN
        if method == 'DBSCAN':
            self.btn_select_outliers.setEnabled(len(self.current_clusterer.outlier_indices_) > 0)

    def _optimize_params(self):
        """Optimize clustering parameters."""
        data = self._get_active_data()
        if data is None or len(data) == 0:
            print("No data for optimization")
            return

        method = self.method_combo.currentText()

        if method == 'DBSCAN':
            param_ranges = {
                'eps': (self.dbscan_eps_min.value(), self.dbscan_eps_max.value()),
                'min_samples': (2, 20),
            }
        elif method == 'K-Means':
            param_ranges = {
                'n_clusters': (2, 10),
            }
        elif method == 'Decision Tree':
            param_ranges = {
                'n_clusters': (2, 10),
                'max_depth': (3, 10),
            }
        else:
            param_ranges = None

        best_params = self.current_clusterer.optimize(data, param_ranges)
        self._update_ui_from_params(best_params)

        # Run clustering with optimized params
        self._run_clustering()

    def _update_results_display(self):
        """Update results label."""
        if self.current_labels is None:
            self.results_label.setText("Clusters: -  Silhouette: -")
            return

        n_clusters = self.current_clusterer.n_clusters_
        silhouette = self.current_clusterer.silhouette_score_

        text = f"Clusters: {n_clusters}"
        if silhouette is not None:
            text += f"  Silhouette: {silhouette:.3f}"

        self.results_label.setText(text)

    def _select_outliers(self):
        """Emit outlier indices for selection."""
        if isinstance(self.current_clusterer, DBSCANClusterer):
            self.outliers_found.emit(self.current_clusterer.outlier_indices_)

    def get_labels(self) -> np.ndarray:
        """Get current cluster labels."""
        return self.current_labels

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance if available (Decision Tree)."""
        return self.current_clusterer.get_feature_importance()
