"""Main application entry point for the Spectrum Viewer."""

import sys
import copy
import json
import numpy as np
from typing import List, Optional, Dict, Any

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QGroupBox, QTabWidget, QPushButton, QSizePolicy,
    QTableWidget, QTableWidgetItem, QAbstractItemView, QHeaderView,
    QFileDialog
)
from PyQt5.QtCore import Qt

from .data_manager import ViewerDataManager
from .widgets import (
    PreprocessingTab, EmbeddingTab, ClusteringTab,
    ColorMappingTab, HyperspectralTab, ExperimentLoaderWidget,
    ClassificationTab, FilterTab
)
from .plots import ScatterPlotWidget, SpectrumPlotWidget, LoadingsPlotWidget
from .utils.dark_theme import apply_dark_theme, DEFAULT_BLUE
from .utils.custom_widgets import SortableTableWidgetItem
from .utils.embedding_cache import EmbeddingCache

from utils.raman_robot_dataset import SpectrumPreprocessorWrapper


class SpectraViewerApp(QMainWindow):
    """Main application window for the Spectra Viewer."""

    def __init__(self, data_folder: str = None, preprocessor: SpectrumPreprocessorWrapper = None):
        super().__init__()

        self.setWindowTitle("Spectra Viewer 3.0")

        # Data management
        self.preprocessor = preprocessor or self._create_default_preprocessor()
        self.data_manager = ViewerDataManager()
        self.embedding_cache = EmbeddingCache()

        # State
        self.selected_indices: List[int] = []
        self.visible_indices: List[int] = []
        self.highlighted_indices: List[int] = []
        self.hovered_index: Optional[int] = None
        self.embedding_method = "PCA"
        self.embedding_dim = 2
        self.n_pcs_for_loading = 5
        self.has_lasso_selection = False
        self.hidden_indices: set = set()  # Union of all hidden spectrum indices

        # Initialize UI
        self._init_ui()

        # Load data if provided
        if data_folder:
            self._add_experiment(data_folder)

    def _create_default_preprocessor(self) -> SpectrumPreprocessorWrapper:
        """Create default preprocessor with standard settings."""
        default_config = {
            'preprocessing': {
                'cropping': {
                    'start_raman_shift_cm': 662.697,
                    'end_raman_shift_cm': 1784.104
                },
                'baseline_correction': {
                    'lam': 100,
                    'diff_order': 1,
                    'max_iter': 15,
                    'tol': 0.005
                },
                'cosmic_rays_removal': {
                    'threshold': 10
                },
                'smoothing': {
                    'window_length': 5,
                    'polyorder': 3
                },
                'normalization_type': 'by_max'
            },
            'enabled': {
                'cropping': True,
                'baseline_correction': True,
                'remove_cosmic_rays': False,
                'normalization': True,
                'smoothing': True
            }
        }
        return SpectrumPreprocessorWrapper.from_dict(default_config)

    def _init_ui(self):
        """Initialize the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        root_layout = QHBoxLayout(central_widget)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(10)

        # Main splitter
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setChildrenCollapsible(False)
        root_layout.addWidget(main_splitter)

        # === LEFT AREA ===
        left_area = QWidget()
        left_layout = QVBoxLayout(left_area)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        # Tabs
        self.left_tabs = QTabWidget()
        self.left_tabs.setDocumentMode(True)

        # Experiment loader tab (first tab)
        self.experiment_loader = ExperimentLoaderWidget()
        self.experiment_loader.experiment_added.connect(self._add_experiment)
        self.experiment_loader.experiment_removed.connect(self._remove_experiment)
        self.experiment_loader.experiments_cleared.connect(self._clear_experiments)
        self.left_tabs.addTab(self.experiment_loader, "Experiments")

        # Preprocessing tab
        self.preprocessing_tab = PreprocessingTab(self.preprocessor)
        self.preprocessing_tab.preprocessing_changed.connect(self._on_preprocessing_changed)
        self.left_tabs.addTab(self.preprocessing_tab, "Preprocessing")

        # Filter tab (combines metadata filters and hidden spectra)
        self.filter_tab = FilterTab()
        self.filter_tab.filter_changed.connect(self._on_filter_changed)
        self.filter_tab.restore_requested.connect(self._restore_hidden_set)
        self.filter_tab.restore_all_requested.connect(self._restore_all_hidden)
        self.left_tabs.addTab(self.filter_tab, "Filter")

        # Embedding tab
        self.embedding_tab = EmbeddingTab()
        self.embedding_tab.method_changed.connect(self._on_embedding_method_changed)
        self.embedding_tab.dim_changed.connect(self._on_embedding_dim_changed)
        self.embedding_tab.pca_loadings_changed.connect(self._on_pca_loadings_changed)
        self.left_tabs.addTab(self.embedding_tab, "Dim Reduction")

        # Color mapping tab
        self.color_tab = ColorMappingTab()
        self.color_tab.color_option_changed.connect(self._on_color_changed)
        self.left_tabs.addTab(self.color_tab, "Color Mapping")

        # Clustering tab
        self.clustering_tab = ClusteringTab()
        self.clustering_tab.clustering_complete.connect(self._on_clustering_complete)
        self.clustering_tab.outliers_found.connect(self._on_outliers_found)
        self.left_tabs.addTab(self.clustering_tab, "Clustering")

        left_layout.addWidget(self.left_tabs, stretch=0)

        # Scatter plot
        scatter_group = QGroupBox("Dim Reduced Scatter")
        scatter_layout = QVBoxLayout(scatter_group)
        scatter_layout.setContentsMargins(8, 8, 8, 8)

        self.scatter_plot = ScatterPlotWidget()
        self.scatter_plot.selection_changed.connect(self._on_scatter_selection)
        self.scatter_plot.point_clicked.connect(self._on_scatter_point_clicked)
        scatter_layout.addWidget(self.scatter_plot)

        left_layout.addWidget(scatter_group, stretch=1)

        # Selection buttons
        sel_btn_row = QHBoxLayout()
        self.btn_lasso = QPushButton("Enable Lasso Selection")
        self.btn_lasso.setCheckable(True)
        self.btn_lasso.setChecked(True)
        self.btn_lasso.toggled.connect(self._toggle_lasso)
        sel_btn_row.addWidget(self.btn_lasso)

        btn_clear = QPushButton("Clear Selection")
        btn_clear.clicked.connect(self._clear_selection)
        sel_btn_row.addWidget(btn_clear)

        self.btn_hide_selected = QPushButton("Hide Selected")
        self.btn_hide_selected.clicked.connect(self._hide_selected)
        self.btn_hide_selected.setEnabled(False)
        sel_btn_row.addWidget(self.btn_hide_selected)

        left_layout.addLayout(sel_btn_row)

        # === RIGHT AREA ===
        right_area = QWidget()
        right_layout = QVBoxLayout(right_area)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        # Vertical splitter for spectrum plot and tabs below
        right_splitter = QSplitter(Qt.Vertical)
        right_splitter.setChildrenCollapsible(False)

        # Spectrum plot
        spectrum_group = QGroupBox("Spectra")
        spectrum_layout = QVBoxLayout(spectrum_group)
        spectrum_layout.setContentsMargins(8, 8, 8, 8)

        # Plot mode toggle
        plot_mode_row = QHBoxLayout()
        plot_mode_row.addStretch(1)
        self.btn_plot_mode = QPushButton("Combined View")
        self.btn_plot_mode.setCheckable(True)
        self.btn_plot_mode.setChecked(True)
        self.btn_plot_mode.toggled.connect(self._toggle_plot_mode)
        self.btn_plot_mode.setMaximumWidth(160)
        plot_mode_row.addWidget(self.btn_plot_mode)
        spectrum_layout.addLayout(plot_mode_row)

        self.spectrum_plot = SpectrumPlotWidget()
        self.spectrum_plot.spectrum_clicked.connect(self._on_spectrum_clicked)
        self.spectrum_plot.spectrum_hovered.connect(self._on_spectrum_hovered)
        spectrum_layout.addWidget(self.spectrum_plot)

        right_splitter.addWidget(spectrum_group)

        # === RIGHT TABS (below spectrum plot) ===
        self.right_tabs = QTabWidget()
        self.right_tabs.setDocumentMode(True)

        # Tab 1: Hyperspectral Map
        self.hyperspectral_tab = HyperspectralTab()
        self.hyperspectral_tab.point_selected.connect(self._on_hyperspectral_point_selected)
        self.hyperspectral_tab.wavenumber_changed.connect(self._on_wavenumber_changed)
        self.right_tabs.addTab(self.hyperspectral_tab, "Hyperspectral Map")

        # Tab 2: PCA Loadings
        load_widget = QWidget()
        load_inner_layout = QVBoxLayout(load_widget)
        load_inner_layout.setContentsMargins(8, 8, 8, 8)

        self.loadings_plot = LoadingsPlotWidget()
        load_inner_layout.addWidget(self.loadings_plot)

        self.right_tabs.addTab(load_widget, "PCA Loadings")

        # Tab 3: Metadata
        metadata_widget = QWidget()
        metadata_inner_layout = QVBoxLayout(metadata_widget)
        metadata_inner_layout.setContentsMargins(8, 8, 8, 8)
        metadata_inner_layout.setSpacing(6)

        self.metadata_table = QTableWidget()
        self.metadata_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.metadata_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.metadata_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.metadata_table.verticalHeader().setVisible(False)
        self.metadata_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.metadata_table.itemSelectionChanged.connect(self._on_table_selection_changed)
        self.metadata_table.horizontalHeader().sectionClicked.connect(self._on_table_header_clicked)
        metadata_inner_layout.addWidget(self.metadata_table)

        btn_save = QPushButton("Save Metadata")
        btn_save.clicked.connect(self._save_metadata)
        metadata_inner_layout.addWidget(btn_save)

        self.right_tabs.addTab(metadata_widget, "Metadata")

        # Tab 4: Classification
        self.classification_tab = ClassificationTab()
        self.right_tabs.addTab(self.classification_tab, "Classification")

        right_splitter.addWidget(self.right_tabs)

        # Set initial splitter sizes (spectrum gets more space)
        right_splitter.setStretchFactor(0, 1)
        right_splitter.setStretchFactor(1, 1)

        right_layout.addWidget(right_splitter)

        # Add to splitter
        main_splitter.addWidget(left_area)
        main_splitter.addWidget(right_area)
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setSizes([520, 1200])

        # Initialize lasso
        self._toggle_lasso(True)

    def _add_experiment(self, path: str):
        """Add experiment data from folder (appends to existing data)."""
        print(f"Adding experiment from: {path}")

        # Add the experiment
        experiment_id = self.data_manager.add_experiment(path, self.preprocessor)

        if experiment_id < 0:
            print(f"Failed to add experiment from {path}")
            return

        # Get experiment info
        experiments = self.data_manager.get_experiments()
        exp_info = next((e for e in experiments if e['id'] == experiment_id), None)

        if exp_info:
            # Add to experiment loader widget
            self.experiment_loader.add_experiment(
                experiment_id,
                exp_info['folder_name'],
                exp_info['spectrum_count'],
                exp_info['path']
            )

        # Update totals in experiment loader
        self.experiment_loader.set_info(
            self.data_manager.get_sample_count(),
            self.data_manager.get_spectrum_count()
        )

        # Update visible indices (respects hidden indices and filters)
        self._recompute_visible_indices()
        self.selected_indices = self.visible_indices.copy()

        # Update metadata keys (experiment_folder should now be available)
        metadata_keys = self.data_manager.get_metadata_keys()
        self.filter_tab.set_metadata_keys(metadata_keys)
        self.preprocessing_tab.set_metadata_keys(metadata_keys)  # For averaging combo
        self.color_tab.populate_color_options(metadata_keys)

        # Add filter row if none exist
        if not self.filter_tab.filters:
            self.filter_tab.add_filter_row()

        # Update hyperspectral tab - set_samples MUST come before set_wavenumber_range
        # because set_wavenumber_range can trigger updates on old map widgets with stale indices
        self.hyperspectral_tab.set_samples(
            self._get_visible_samples(),
            self.data_manager
        )
        wn_min, wn_max = self.data_manager.get_wavenumber_range()
        self.hyperspectral_tab.set_wavenumber_range(wn_min, wn_max)

        # Clear embedding cache and recompute for combined data
        self.embedding_cache.clear()
        self.highlighted_indices = []

        # Compute embedding on combined data and update all plots
        self._compute_embedding(force=True)
        self._update_scatter_plot()
        self._update_spectrum_plot()
        self._update_loadings_plot()

    def _remove_experiment(self, experiment_id: int):
        """Remove an experiment from the dataset."""
        print(f"Removing experiment ID: {experiment_id}")

        # Remove from data manager
        success = self.data_manager.remove_experiment(experiment_id)

        if not success:
            return

        # Remove from experiment loader widget
        self.experiment_loader.remove_experiment(experiment_id)

        # Update totals
        self.experiment_loader.set_info(
            self.data_manager.get_sample_count(),
            self.data_manager.get_spectrum_count()
        )

        if len(self.data_manager) == 0:
            # No data left - reset state
            self._reset_viewer_state()
            return

        # Clear hidden indices - they become invalid when spectra are removed/reindexed
        self.hidden_indices.clear()
        self.filter_tab.reset_hidden()

        # Update visible indices (respects filters, hidden indices now empty)
        self._recompute_visible_indices()
        self.selected_indices = self.visible_indices.copy()

        # Update metadata keys
        metadata_keys = self.data_manager.get_metadata_keys()
        self.filter_tab.set_metadata_keys(metadata_keys)
        self.preprocessing_tab.set_metadata_keys(metadata_keys)  # For averaging combo
        self.color_tab.populate_color_options(metadata_keys)

        # Update hyperspectral tab - set_samples MUST come before set_wavenumber_range
        # because set_wavenumber_range can trigger updates on old map widgets with stale indices
        self.hyperspectral_tab.set_samples(
            self._get_visible_samples(),
            self.data_manager
        )
        wn_min, wn_max = self.data_manager.get_wavenumber_range()
        self.hyperspectral_tab.set_wavenumber_range(wn_min, wn_max)

        # Clear embedding cache and recompute for remaining data
        self.embedding_cache.clear()
        self.highlighted_indices = []

        # Recompute embedding on remaining data
        self._compute_embedding(force=True)
        self._update_scatter_plot()
        self._update_spectrum_plot()
        self._update_loadings_plot()

    def _clear_experiments(self):
        """Clear all loaded experiments."""
        print("Clearing all experiments")

        # Clear data manager
        self.data_manager.clear_all()

        # Clear experiment loader widget
        self.experiment_loader.clear_experiments()

        # Reset viewer state
        self._reset_viewer_state()

    def _reset_viewer_state(self):
        """Reset viewer to empty state."""
        self.visible_indices = []
        self.selected_indices = []
        self.highlighted_indices = []
        self.hovered_index = None
        self.has_lasso_selection = False

        # Clear hidden spectra
        self.hidden_indices.clear()
        self.filter_tab.reset_hidden()
        self.btn_hide_selected.setEnabled(False)

        # Clear embedding cache
        self.embedding_cache.clear()

        # Clear plots
        self.scatter_plot.clear()
        self.spectrum_plot.clear()
        self.loadings_plot.clear()

        # Clear metadata table
        self.metadata_table.clearContents()
        self.metadata_table.setRowCount(0)
        self.metadata_table.setColumnCount(0)

        # Clear hyperspectral tab
        self.hyperspectral_tab.clear()

        # Reset info labels
        self.experiment_loader.set_info(0, 0)

    def _on_preprocessing_changed(self):
        """Handle preprocessing parameter changes."""
        self.preprocessing_tab.update_preprocessor()
        self.data_manager.reprocess(self.preprocessor)

        # Clear embedding cache and highlighted state
        self.embedding_cache.clear()
        self.highlighted_indices = []

        # Recompute and redraw
        self._compute_embedding()
        self._update_scatter_plot()
        self._update_spectrum_plot()
        self._update_loadings_plot()

        print("Data re-processed with new parameters")

    def _on_filter_changed(self):
        """Handle filter changes."""
        self._recompute_visible_indices()

        # Update filter value combos
        for f in self.filter_tab.filters:
            attr = f.get('attr')
            if attr and attr != "None":
                values = self.data_manager.get_metadata_values(attr)
                self.filter_tab.populate_filter_values(f, values)

        # Keep selected indices that are still visible
        self.selected_indices = [i for i in self.selected_indices if i in self.visible_indices]
        if not self.selected_indices:
            self.selected_indices = self.visible_indices.copy()

        # Clear highlights
        self.highlighted_indices = []

        # Recompute and redraw
        self._compute_embedding(force=True)
        self._update_scatter_plot()

    def _recompute_visible_indices(self):
        """Recompute visible indices based on filters and hidden spectra."""
        # Get active filters
        active_filters = self.filter_tab.get_active_filters()

        if not active_filters:
            base_visible = list(range(len(self.data_manager)))
        else:
            base_visible = []
            for i, obj in enumerate(self.data_manager.data_objects):
                ok = True
                for attr, val in active_filters:
                    v = obj.metadata.get(attr)
                    if str(v) != val:
                        ok = False
                        break
                if ok:
                    base_visible.append(i)

        # Exclude hidden indices
        self.visible_indices = [i for i in base_visible if i not in self.hidden_indices]
        self._update_spectrum_plot()

    def _on_embedding_method_changed(self, method: str):
        """Handle embedding method change."""
        self.embedding_method = method
        self.highlighted_indices = []
        self._compute_embedding(force=True)
        self._update_scatter_plot()
        self._update_spectrum_plot()
        self._update_loadings_plot()

    def _on_embedding_dim_changed(self, dim: int):
        """Handle embedding dimension change."""
        self.embedding_dim = dim
        self.highlighted_indices = []
        self._compute_embedding(force=True)
        self._update_scatter_plot()
        self._update_spectrum_plot()
        self._update_loadings_plot()

    def _on_pca_loadings_changed(self, n_pcs: int):
        """Handle PCA loadings count change."""
        self.n_pcs_for_loading = n_pcs
        if self.embedding_method == "PCA":
            self._compute_embedding(force=True)
            self._update_scatter_plot()
            self._update_loadings_plot()

    def _on_color_changed(self):
        """Handle color option change."""
        self._update_scatter_plot()
        self._update_spectrum_plot()

    def _on_clustering_complete(self, labels: np.ndarray):
        """Handle clustering completion."""
        # Store cluster labels in metadata - map to visible indices since clustering
        # only runs on the visible subset
        for i, label in enumerate(labels):
            if i < len(self.visible_indices):
                idx = self.visible_indices[i]
                self.data_manager.data_objects[idx].metadata['cluster'] = int(label)

        # Add 'cluster' to color options
        self.color_tab.add_color_option('cluster')
        self.color_tab.set_color_option('cluster')

        # Refresh metadata keys so 'cluster' appears in filter combos
        metadata_keys = self.data_manager.get_metadata_keys()
        self.filter_tab.set_metadata_keys(metadata_keys)
        self.preprocessing_tab.set_metadata_keys(metadata_keys)

        self._update_scatter_plot()
        self._update_spectrum_plot()

    def _on_outliers_found(self, outlier_indices: set):
        """Handle outlier detection."""
        # Store outlier labels in metadata
        for i in range(len(self.data_manager.data_objects)):
            self.data_manager.data_objects[i].metadata['outlier'] = 1 if i in outlier_indices else 0

        # Add 'outlier' to color options
        self.color_tab.add_color_option('outlier')
        self.color_tab.set_color_option('outlier')

        # Refresh metadata keys so 'outlier' appears in filter combos
        metadata_keys = self.data_manager.get_metadata_keys()
        self.filter_tab.set_metadata_keys(metadata_keys)
        self.preprocessing_tab.set_metadata_keys(metadata_keys)

        self._update_scatter_plot()

    def _on_hyperspectral_point_selected(self, idx: int):
        """Handle point selection from hyperspectral map — highlight, don't replace selection."""
        if idx >= 0:
            self._highlight_spectrum(idx)

    def _on_wavenumber_changed(self, wavenumber: float):
        """Handle wavenumber slider change - draw vertical line on spectrum plot."""
        self.spectrum_plot.set_wavenumber_line(wavenumber)

    def _on_scatter_selection(self, indices: List[int]):
        """Handle lasso selection in scatter plot."""
        self.selected_indices = indices
        self.highlighted_indices = []
        self.has_lasso_selection = len(indices) > 0 and set(indices) != set(self.visible_indices)

        # Enable hide button only when there's a valid lasso selection
        self.btn_hide_selected.setEnabled(self.has_lasso_selection and len(indices) > 0)

        self._update_spectrum_plot()
        self._update_hyperspectral_selection()

    def _on_scatter_point_clicked(self, idx: int):
        """Handle point click in scatter plot."""
        if idx >= 0:
            self._highlight_spectrum(idx)

    def _on_spectrum_clicked(self, idx: int):
        """Handle spectrum line click."""
        if idx >= 0:
            self._highlight_spectrum(idx)
        else:
            self._clear_highlights()

    def _on_spectrum_hovered(self, idx: int):
        """Handle spectrum line hover."""
        self.hovered_index = idx if idx >= 0 else None
        self._update_highlights()

    def _on_table_selection_changed(self):
        """Handle table selection change."""
        selected_rows = {item.row() for item in self.metadata_table.selectedItems()}
        highlighted_indices = []

        for row in selected_rows:
            item = self.metadata_table.item(row, 0)
            if item:
                idx = item.data(Qt.UserRole)
                if idx is not None:
                    highlighted_indices.append(idx)

        self._update_highlights(highlighted_indices)

    def _on_table_header_clicked(self, column: int):
        """Handle table header click for sorting."""
        self.metadata_table.sortItems(column, Qt.AscendingOrder)
        self._update_highlights()

    def _compute_embedding(self, force: bool = False):
        """Compute embedding for visible data."""
        if len(self.data_manager) == 0:
            return

        self.embedding_cache.compute_embedding(
            self.data_manager.data_objects,
            method=self.embedding_method,
            dim=self.embedding_dim,
            indices=self.visible_indices,
            n_pcs_for_loading=self.n_pcs_for_loading,
            force=force
        )

        # Update clustering tab data
        if self.visible_indices:
            data_matrix = np.array([
                self.data_manager.data_objects[i].intensity
                for i in self.visible_indices
            ])
            self.clustering_tab.set_data(data_matrix)

            # Extract visible-only embedding rows for clustering on embeddings
            if self.embedding_cache.embedding_result is not None:
                visible_emb = self.embedding_cache.embedding_result[
                    np.asarray(self.visible_indices, dtype=int)
                ]
                # Filter out rows with NaN (shouldn't happen for visible, but be safe)
                valid_mask = ~np.any(np.isnan(visible_emb), axis=1)
                if np.all(valid_mask):
                    self.clustering_tab.set_embedding_data(visible_emb)
                else:
                    self.clustering_tab.set_embedding_data(visible_emb[valid_mask])

        # Update classification tab
        self._update_classification_tab()

    def _update_classification_tab(self):
        """Update the classification tab with current data."""
        if not self.visible_indices or len(self.data_manager) == 0:
            return

        self.classification_tab.set_data(
            data_objects=self.data_manager.data_objects,
            visible_indices=self.visible_indices,
            embedding_matrix=self.embedding_cache.embedding_result
        )

    def _update_scatter_plot(self):
        """Update the scatter plot."""
        if self.embedding_cache.embedding_result is None:
            return

        self.scatter_plot.set_embedding(
            self.embedding_cache.embedding_result,
            self.embedding_dim,
            self.embedding_method
        )
        self.scatter_plot.set_visible_indices(self.visible_indices)
        self.scatter_plot.enable_lasso(self.btn_lasso.isChecked() and self.embedding_dim == 2)

        self.scatter_plot.plot(
            self.data_manager.data_objects,
            self.color_tab.get_color_attr(),
            self.color_tab.get_cmap()
        )

        # Only show star overlay for highlighted (single clicked) spectrum,
        # not for all lasso-selected spectra
        self.scatter_plot.update_highlight(self.highlighted_indices)
        self.scatter_plot.update_markers(self.highlighted_indices)

        # Update hyperspectral maps with current scatter colors
        self._update_hyperspectral_colors()

    def _update_spectrum_plot(self):
        """Update the spectrum plot."""
        n_objects = len(self.data_manager)

        # Filter out invalid indices (can happen after experiment removal)
        if self.selected_indices:
            self.selected_indices = [i for i in self.selected_indices if i < n_objects]

        if not self.selected_indices:
            self.selected_indices = self.visible_indices.copy() if self.visible_indices else []

        scatter_colors = self.scatter_plot.get_scatter_colors()
        index_to_pos = self.scatter_plot.index_to_scatter_pos

        self.spectrum_plot.plot(
            self.data_manager.data_objects,
            self.selected_indices,
            scatter_colors,
            index_to_pos,
            group_by='patient'
        )

        self._update_metadata_table()

    def _update_loadings_plot(self):
        """Update the PCA loadings plot."""
        if self.embedding_method != "PCA":
            self.loadings_plot.clear()
            return

        if self.embedding_cache.pca_loadings is None:
            self.loadings_plot.clear()
            return

        self.loadings_plot.set_pca_data(
            self.embedding_cache.pca_loadings,
            self.embedding_cache.pca_explained_var,
            self.data_manager.get_common_wavenumber_grid(),
            self.n_pcs_for_loading
        )

        xlim = self.spectrum_plot.get_xlim()
        self.loadings_plot.set_xlim(xlim)

        left, right = self.spectrum_plot.get_margins()
        self.loadings_plot.set_margins(left, right)

        self.loadings_plot.plot()

    def _update_metadata_table(self):
        """Update the metadata table."""
        self.metadata_table.clearContents()
        self.metadata_table.setRowCount(0)
        self.metadata_table.setColumnCount(0)

        if not self.selected_indices:
            return

        n_objects = len(self.data_manager)
        valid_indices = [i for i in self.selected_indices if i < n_objects]
        if not valid_indices:
            return

        # Get metadata keys
        all_keys = set()
        for idx in valid_indices:
            all_keys.update(self.data_manager.data_objects[idx].metadata.keys())
        metadata_keys = sorted(all_keys)

        self.metadata_table.setColumnCount(len(metadata_keys))
        self.metadata_table.setHorizontalHeaderLabels(
            [k.replace("_", " ").title() for k in metadata_keys]
        )
        self.metadata_table.setRowCount(len(valid_indices))

        for row, idx in enumerate(valid_indices):
            obj = self.data_manager.data_objects[idx]
            for col, key in enumerate(metadata_keys):
                val = obj.metadata.get(key, "")
                item = SortableTableWidgetItem(str(val))
                if col == 0:
                    item.setData(Qt.UserRole, idx)
                self.metadata_table.setItem(row, col, item)

    def _update_highlights(self, highlighted_indices: List[int] = None):
        """Update highlights across all views."""
        if highlighted_indices is None:
            highlighted_indices = []

        self.highlighted_indices = highlighted_indices

        scatter_colors = self.scatter_plot.get_scatter_colors()
        index_to_pos = self.scatter_plot.index_to_scatter_pos

        self.spectrum_plot.hovered_index = self.hovered_index
        self.spectrum_plot.update_highlights(
            highlighted_indices,
            scatter_colors,
            index_to_pos
        )

        self.scatter_plot.update_highlight(highlighted_indices)
        self.scatter_plot.update_markers(highlighted_indices)

        # Update hyperspectral map highlight
        if highlighted_indices:
            self.hyperspectral_tab.highlight_point(highlighted_indices[0])
        else:
            self.hyperspectral_tab.clear_highlight()

    def _highlight_spectrum(self, idx: int):
        """Highlight a specific spectrum."""
        self.highlighted_indices = [idx]

        # Find row in table
        for row in range(self.metadata_table.rowCount()):
            item = self.metadata_table.item(row, 0)
            if item and item.data(Qt.UserRole) == idx:
                self.metadata_table.blockSignals(True)
                self.metadata_table.clearSelection()
                self.metadata_table.selectRow(row)
                self.metadata_table.blockSignals(False)
                break

        self._update_highlights([idx])

    def _clear_highlights(self):
        """Clear all highlights."""
        self.metadata_table.clearSelection()
        self.hovered_index = None
        self.highlighted_indices = []
        self._update_highlights([])

    def _update_hyperspectral_colors(self):
        """Update hyperspectral maps with current scatter colors."""
        scatter_colors = self.scatter_plot.get_scatter_colors()
        index_to_pos = self.scatter_plot.index_to_scatter_pos
        if scatter_colors is not None:
            self.hyperspectral_tab.update_point_colors(scatter_colors, index_to_pos)

    def _update_hyperspectral_selection(self):
        """Update hyperspectral maps with current lasso selection."""
        if self.has_lasso_selection:
            scatter_colors = self.scatter_plot.get_scatter_colors()
            index_to_pos = self.scatter_plot.index_to_scatter_pos
            self.hyperspectral_tab.update_selection(
                self.selected_indices, scatter_colors, index_to_pos
            )
        else:
            self.hyperspectral_tab.clear_selection()

    def _toggle_lasso(self, enabled: bool):
        """Toggle lasso selection."""
        if enabled:
            self.btn_lasso.setText("Disable Lasso Selection")
        else:
            self.btn_lasso.setText("Enable Lasso Selection")

        self.scatter_plot.enable_lasso(enabled and self.embedding_dim == 2)

    def _toggle_plot_mode(self, checked: bool):
        """Toggle between combined and grouped plot mode."""
        if checked:
            self.spectrum_plot.set_plot_mode('combined')
            self.btn_plot_mode.setText("Grouped View")
        else:
            self.spectrum_plot.set_plot_mode('grouped')
            self.btn_plot_mode.setText("Combined View")

        self._update_spectrum_plot()

    def _clear_selection(self):
        """Clear current selection."""
        self.selected_indices = []
        self.has_lasso_selection = False
        self.btn_hide_selected.setEnabled(False)
        self.spectrum_plot.clear()
        self.metadata_table.clearContents()
        self.metadata_table.setRowCount(0)
        self.metadata_table.setColumnCount(0)
        self._clear_highlights()
        self._update_hyperspectral_selection()

    def _hide_selected(self):
        """Hide the currently selected spectra."""
        if not self.selected_indices or not self.has_lasso_selection:
            return

        # Get the set of indices to hide
        indices_to_hide = set(self.selected_indices)

        if not indices_to_hide:
            return

        print(f"Hiding {len(indices_to_hide)} spectra")

        # Add to hidden sets widget
        self.filter_tab.add_hidden_set(indices_to_hide)

        # Update our tracking set
        self.hidden_indices.update(indices_to_hide)

        # Recompute visible indices
        self._recompute_visible_indices()

        # Clear selection
        self.selected_indices = self.visible_indices.copy()
        self.has_lasso_selection = False
        self.btn_hide_selected.setEnabled(False)
        self.highlighted_indices = []

        # Recompute embedding and update all views
        self.embedding_cache.clear()
        self._compute_embedding(force=True)
        self._update_scatter_plot()
        self._update_spectrum_plot()
        self._update_loadings_plot()

        # Update hyperspectral with new samples (excluding hidden)
        self.hyperspectral_tab.set_samples(
            self._get_visible_samples(),
            self.data_manager
        )

    def _restore_hidden_set(self, set_index: int):
        """Restore a specific set of hidden spectra."""
        restored_indices = self.filter_tab.remove_hidden_set(set_index)

        if not restored_indices:
            return

        print(f"Restoring {len(restored_indices)} spectra from set {set_index}")

        # Remove from our tracking set
        self.hidden_indices -= restored_indices

        # Recompute and update
        self._after_restore()

    def _restore_all_hidden(self):
        """Restore all hidden spectra."""
        restored_indices = self.filter_tab.clear_hidden_sets()

        if not restored_indices:
            return

        print(f"Restoring all {len(restored_indices)} hidden spectra")

        # Clear our tracking set
        self.hidden_indices.clear()

        # Recompute and update
        self._after_restore()

    def _after_restore(self):
        """Common logic after restoring hidden spectra."""
        # Recompute visible indices
        self._recompute_visible_indices()

        # Update selection to include all visible
        self.selected_indices = self.visible_indices.copy()
        self.has_lasso_selection = False
        self.btn_hide_selected.setEnabled(False)
        self.highlighted_indices = []

        # Recompute embedding and update all views
        self.embedding_cache.clear()
        self._compute_embedding(force=True)
        self._update_scatter_plot()
        self._update_spectrum_plot()
        self._update_loadings_plot()

        # Update hyperspectral with new samples
        self.hyperspectral_tab.set_samples(
            self._get_visible_samples(),
            self.data_manager
        )

    def _get_visible_samples(self) -> dict:
        """Get samples dict filtered to only include visible indices."""
        all_samples = self.data_manager.get_samples()
        visible_set = set(self.visible_indices)

        # Filter each sample's indices to only visible ones
        visible_samples = {}
        for sample_id, indices in all_samples.items():
            visible_in_sample = [i for i in indices if i in visible_set]
            if visible_in_sample:
                visible_samples[sample_id] = visible_in_sample

        return visible_samples

    def _save_metadata(self):
        """Save metadata for selected spectra."""
        if not self.selected_indices:
            return

        metadata_list = [
            self.data_manager.data_objects[idx].metadata
            for idx in self.selected_indices
        ]

        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Metadata", "", "JSON Files (*.json)"
        )
        if filename:
            with open(filename, "w") as f:
                json.dump(metadata_list, f, indent=4, default=str)
            print(f"Metadata saved to {filename}")


def run_viewer(data_folder: str = None, preprocessor: SpectrumPreprocessorWrapper = None):
    """
    Run the Spectra Viewer application.

    Args:
        data_folder: Optional path to data folder to load on startup
        preprocessor: Optional preprocessor configuration
    """
    app = QApplication(sys.argv)
    apply_dark_theme(app)

    window = SpectraViewerApp(data_folder, preprocessor)
    window.resize(1400, 800)
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Spectra Viewer")
    parser.add_argument("--data", "-d", type=str, default=None,
                        help="Path to data folder")
    args = parser.parse_args()

    run_viewer(args.data)
