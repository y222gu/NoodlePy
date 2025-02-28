import sys
import json
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QListWidget, QListWidgetItem, 
                             QFileDialog, QComboBox, QLabel, QGroupBox, QCheckBox, QDoubleSpinBox, QSpinBox, QRadioButton, QButtonGroup)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
from sklearn.manifold import TSNE
from combat.pycombat import pycombat
import scanpy as sc
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from typing import List
import seaborn as sns

# Use the "fast" style.
plt.style.use('fast')

########################################
# Dummy preprocessor class definition
########################################

class SpectrumPreprocessor:
    def __init__(self, cropping=True, 
                 baseline_correction=False,
                 remove_cosmic_rays=True,
                 normalization=False,
                 smoothing=False):
        self.cropping = cropping
        self.baseline_correction = baseline_correction
        self.remove_cosmic_rays = remove_cosmic_rays
        self.normalization = normalization
        self.smoothing = smoothing

    def preprocess(self, obj):
        new_intensity = obj.intensity.copy()
        new_raman = obj.raman_shift_cm.copy()
        if self.cropping:
            n = int(len(new_intensity) * 0.7)
            new_intensity = new_intensity[:n]
            new_raman = new_raman[:n]
        if self.remove_cosmic_rays:
            new_intensity = new_intensity - np.random.normal(0, 0.01, size=new_intensity.shape)
        return SpectraData(new_intensity, new_raman, obj.metadata)

########################################
# Data class for spectra
########################################

class SpectraData:
    def __init__(self, intensity, raman_shift_cm, metadata):
        self.intensity = intensity
        self.raman_shift_cm = raman_shift_cm
        self.metadata = metadata

########################################
# Viewer class with Outlier Detection & DBSCAN Threshold Entry
########################################

class SpectraViewer(QMainWindow):
    def __init__(self, data_objects, preprocessor):
        super().__init__()
        self.setWindowTitle("Spectra Viewer")
        self.original_data_objects = data_objects  
        self.preprocessor = preprocessor
        self.data_objects = [self.preprocessor.preprocess(obj) for obj in self.original_data_objects]
        
        self.selected_indices = []
        self.selected_index_to_line = {}
        self.hovered_index = None

        self.embedding_method = "T-SNE"
        self.embedding_dim = 2
        self.embedding_result = None
        self.embedding_cache = {}
        self.outlier_indices = set()

        # Parameters for outlier detection.
        self.n_iterations = 3
        self.dbscan_eps = 0.5   # default eps
        self.dbscan_min = 5     # default min_samples


        self.color_by = "None"

        self.lasso = None
        
        self._pan_active = False
        self._pan_press_event = None

        self.initUI()
        self.populate_color_combo()
        self.compute_embedding()
        self.plot_embedding()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        left_layout = QVBoxLayout()
        
        # Create a horizontal layout for preprocessing and DBSCAN groups.
        preproc_dbscan_layout = QHBoxLayout()
        
        # Preprocessing Options Group Box.
        preproc_group = QGroupBox("Preprocessing Options")
        preproc_layout = QVBoxLayout()
        self.cb_cropping = QCheckBox("Cropping")
        self.cb_cropping.setChecked(self.preprocessor.cropping)
        self.cb_baseline = QCheckBox("Baseline Correction")
        self.cb_baseline.setChecked(self.preprocessor.baseline_correction)
        self.cb_cosmic = QCheckBox("Remove Cosmic Rays")
        self.cb_cosmic.setChecked(self.preprocessor.remove_cosmic_rays)
        self.cb_norm = QCheckBox("Normalization")
        self.cb_norm.setChecked(self.preprocessor.normalization)
        self.cb_smooth = QCheckBox("Smoothing")
        self.cb_smooth.setChecked(self.preprocessor.smoothing)
        for cb in [self.cb_cropping, self.cb_baseline, self.cb_cosmic, self.cb_norm, self.cb_smooth]:
            preproc_layout.addWidget(cb)
        # New: Batch Correction check box.
        self.cb_batch = QCheckBox("Apply Batch Correction")
        self.cb_batch.setChecked(False)
        preproc_layout.addWidget(self.cb_batch)

        # New: Radio buttons for selecting the batch correction method.
        self.rbg_batch = QButtonGroup(self)
        self.rb_combat = QRadioButton("ComBat")
        self.rb_combat.setChecked(True)  # default selection
        self.rbg_batch.addButton(self.rb_combat)
        
        # Create a new layout for the radio buttons and add indentation.
        batch_radio_layout = QVBoxLayout()
        batch_radio_layout.addWidget(self.rb_combat)
        batch_radio_column_layout = QVBoxLayout()
        batch_radio_column_layout.addLayout(batch_radio_layout)
        batch_radio_column_layout.setContentsMargins(20, 0, 0, 0)  # Add indentation
        
        preproc_layout.addLayout(batch_radio_column_layout)

        self.btn_recalculate = QPushButton("Re-process")
        self.btn_recalculate.clicked.connect(self.update_preprocessing)
        preproc_layout.addWidget(self.btn_recalculate)

        preproc_group.setLayout(preproc_layout)
        preproc_dbscan_layout.addWidget(preproc_group)

        # DBSCAN parameter entries group box.
        dbscan_group = QGroupBox("DBSCAN Parameters")
        dbscan_layout = QVBoxLayout()

        # Optimization range for eps.
        dbscan_opt_layout = QHBoxLayout()
        opt_eps_label = QLabel("Optimize eps range:")
        self.opt_eps_min_spin = QDoubleSpinBox()
        self.opt_eps_min_spin.setRange(0.0, 10.0)
        self.opt_eps_min_spin.setSingleStep(0.1)
        self.opt_eps_min_spin.setValue(0.1)
        self.opt_eps_max_spin = QDoubleSpinBox()
        self.opt_eps_max_spin.setRange(0.0, 10.0)
        self.opt_eps_max_spin.setSingleStep(0.1)
        self.opt_eps_max_spin.setValue(2.0)
        dbscan_opt_layout.addWidget(opt_eps_label)
        dbscan_opt_layout.addWidget(self.opt_eps_min_spin)
        dbscan_opt_layout.addWidget(self.opt_eps_max_spin)
        dbscan_layout.addLayout(dbscan_opt_layout)

        # Optimization range for min_samples.
        dbscan_min_opt_layout = QHBoxLayout()
        opt_min_label = QLabel("Optimize min_samples range:")
        self.opt_min_min_spin = QSpinBox()
        self.opt_min_min_spin.setRange(2, 50)
        self.opt_min_min_spin.setSingleStep(1)
        self.opt_min_min_spin.setValue(2)
        self.opt_min_max_spin = QSpinBox()
        self.opt_min_max_spin.setRange(2, 50)
        self.opt_min_max_spin.setSingleStep(1)
        self.opt_min_max_spin.setValue(20)
        dbscan_min_opt_layout.addWidget(opt_min_label)
        dbscan_min_opt_layout.addWidget(self.opt_min_min_spin)
        dbscan_min_opt_layout.addWidget(self.opt_min_max_spin)
        dbscan_layout.addLayout(dbscan_min_opt_layout)

        # Optimize DBSCAN button.
        self.btn_optimize_dbscan = QPushButton("Optimize DBSCAN")
        self.btn_optimize_dbscan.clicked.connect(self.optimize_dbscan_params)
        dbscan_layout.addWidget(self.btn_optimize_dbscan)

        # DBSCAN parameter entries.
        dbscan_params_row_layout = QHBoxLayout()
        eps_label = QLabel("DBSCAN eps:")
        self.dbscan_spin = QDoubleSpinBox()
        self.dbscan_spin.setRange(0.0, 10.0)
        self.dbscan_spin.setSingleStep(0.1)
        self.dbscan_spin.setValue(self.dbscan_eps)
        self.dbscan_spin.valueChanged.connect(self.on_dbscan_threshold_change)
        dbscan_params_row_layout.addWidget(eps_label)
        dbscan_params_row_layout.addWidget(self.dbscan_spin)
        
        min_label = QLabel("DBSCAN min samples:")
        self.dbscan_min_spin = QSpinBox()
        self.dbscan_min_spin.setRange(2, 50)
        self.dbscan_min_spin.setSingleStep(1)
        self.dbscan_min_spin.setValue(self.dbscan_min)
        self.dbscan_min_spin.valueChanged.connect(self.on_dbscan_min_change)
        dbscan_params_row_layout.addWidget(min_label)
        dbscan_params_row_layout.addWidget(self.dbscan_min_spin)

        self.btn_find_outliers = QPushButton("Find Outliers")
        self.btn_find_outliers.clicked.connect(self.find_outliers)
        dbscan_params_row_layout.addWidget(self.btn_find_outliers)

        dbscan_layout.addLayout(dbscan_params_row_layout)
        dbscan_group.setLayout(dbscan_layout)

        # Add the DBSCAN group box to the horizontal layout.
        preproc_dbscan_layout.addWidget(dbscan_group)

        # Add the horizontal layout to the left layout.
        left_layout.addLayout(preproc_dbscan_layout)

        # Embedding and Color Controls.
        control_layout = QHBoxLayout()
        method_label = QLabel("Embedding:")
        self.embedding_combo = QComboBox()
        self.embedding_combo.addItems(["T-SNE", "PCA"])
        self.embedding_combo.currentIndexChanged.connect(self.on_embedding_change)
        control_layout.addWidget(method_label)
        control_layout.addWidget(self.embedding_combo)
        
        dim_label = QLabel("Dimension:")
        self.dim_combo = QComboBox()
        self.dim_combo.addItems(["2D", "3D"])
        self.dim_combo.currentIndexChanged.connect(self.on_dimension_change)
        control_layout.addWidget(dim_label)
        control_layout.addWidget(self.dim_combo)
        
        color_label = QLabel("Color by:")
        self.color_combo = QComboBox()
        self.color_combo.currentIndexChanged.connect(self.on_color_change)
        control_layout.addWidget(color_label)
        control_layout.addWidget(self.color_combo)
        left_layout.addLayout(control_layout)

        # Embedding Scatter Plot.
        self.fig_scatter = Figure(figsize=(7, 6))
        self.canvas_scatter = FigureCanvas(self.fig_scatter)
        left_layout.addWidget(self.canvas_scatter)
        
        self.canvas_scatter.mpl_connect('scroll_event', self.on_scroll)
        self.canvas_scatter.mpl_connect('button_press_event', self.on_pan_press)
        self.canvas_scatter.mpl_connect('motion_notify_event', self.on_pan_motion)
        self.canvas_scatter.mpl_connect('button_release_event', self.on_pan_release)
        
        # Home and Select Buttons.
        button_layout = QHBoxLayout()
        
        self.btn_home = QPushButton("Home")
        self.btn_home.clicked.connect(self.reset_view)
        button_layout.addWidget(self.btn_home)
        
        self.btn_select = QPushButton("Select")
        self.btn_select.setCheckable(True)
        self.btn_select.toggled.connect(self.toggle_lasso)
        button_layout.addWidget(self.btn_select)
               
        self.btn_clear = QPushButton("Clear Selection")
        self.btn_clear.clicked.connect(self.clear_selection)
        button_layout.addWidget(self.btn_clear)

        # New: Select Outliers Button (disabled by default)
        self.btn_select_outliers = QPushButton("Select Outliers")
        self.btn_select_outliers.setEnabled(False)
        self.btn_select_outliers.clicked.connect(self.select_outliers)
        button_layout.addWidget(self.btn_select_outliers)

        self.btn_draw_heatmap = QPushButton("Draw Heatmap")
        self.btn_draw_heatmap.clicked.connect(self.compute_pca_metadata_correlation_heatmap)
        left_layout.addWidget(self.btn_draw_heatmap)
        
        left_layout.addLayout(button_layout)

        # Right panel for line plot and metadata.
        right_layout = QVBoxLayout()
        self.fig_line = Figure(figsize=(8, 6))
        self.canvas_line = FigureCanvas(self.fig_line)
        self.ax_line = self.fig_line.add_subplot(111)
        self.ax_line.set_title("Select points to view spectra")
        right_layout.addWidget(self.canvas_line)
        self.canvas_line.mpl_connect('motion_notify_event', self.on_line_hover)
        self.canvas_line.mpl_connect('pick_event', self.on_line_pick)
        
        self.metadata_list = QListWidget()
        self.metadata_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.metadata_list.itemSelectionChanged.connect(self.update_line_highlights)
        self.btn_save_metadata = QPushButton("Save Metadata")
        self.btn_save_metadata.clicked.connect(self.save_metadata)
        meta_layout = QVBoxLayout()
        meta_layout.addWidget(self.metadata_list)
        meta_layout.addWidget(self.btn_save_metadata)
        meta_widget = QWidget()
        meta_widget.setLayout(meta_layout)
        right_layout.addWidget(meta_widget)
        
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)
        
        self.canvas_scatter.mpl_connect('pick_event', self.on_scatter_pick)

    def on_dbscan_threshold_change(self, value):
        self.dbscan_eps = value
        print(f"DBSCAN eps updated to {self.dbscan_eps}")

    def on_dbscan_min_change(self, value):
        self.dbscan_min = value
        print(f"DBSCAN min_samples updated to {self.dbscan_min}")

    def optimize_dbscan_params(self):
        # Use the PCA projection (first 2 components) for evaluation.
        data_matrix = np.array([obj.intensity for obj in self.data_objects])
        pca_data = PCA(n_components=2).fit_transform(data_matrix)
        best_score = -1
        best_eps = self.dbscan_eps
        best_min = self.dbscan_min
        # Define grid ranges (adjust as needed).
        for eps in np.linspace(self.opt_eps_min_spin.value(), self.opt_eps_max_spin.value(), 10):
            for min_samples in range(self.opt_min_min_spin.value(), self.opt_min_max_spin.value() + 1):
                db = DBSCAN(eps=eps, min_samples=min_samples)
                labels = db.fit_predict(pca_data)
                # Exclude noise and require at least 2 clusters.
                clusters = [l for l in set(labels) if l != -1]
                if len(clusters) < 2:
                    continue
                try:
                    score = silhouette_score(pca_data, labels)
                    if score > best_score:
                        best_score = score
                        best_eps = eps
                        best_min = min_samples
                except Exception as e:
                    pass
        # Update parameters.
        self.dbscan_eps = best_eps
        self.dbscan_min = best_min
        self.dbscan_spin.setValue(best_eps)
        self.dbscan_min_spin.setValue(best_min)
        print(f"Optimized DBSCAN: eps={best_eps:.3f}, min_samples={best_min}, silhouette={best_score:.3f}")

    def populate_color_combo(self):
        self.color_combo.clear()
        self.color_combo.addItem("None")
        if self.data_objects:
            keys = set()
            for obj in self.data_objects:
                keys.update(obj.metadata.keys())
            for key in sorted(keys):
                self.color_combo.addItem(key)

    def update_preprocessing(self):
        self.preprocessor.cropping = self.cb_cropping.isChecked()
        self.preprocessor.baseline_correction = self.cb_baseline.isChecked()
        self.preprocessor.remove_cosmic_rays = self.cb_cosmic.isChecked()
        self.preprocessor.normalization = self.cb_norm.isChecked()
        self.preprocessor.smoothing = self.cb_smooth.isChecked()
        self.embedding_cache = {}

        # Save the current selection as a list.
        selected_indices = self.selected_indices.copy()
        
        # Reprocess all data objects from the original dataset.
        self.data_objects = [self.preprocessor.preprocess(obj) for obj in self.original_data_objects]
        
        # If the batch correction check box is checked, apply the selected method.
        if self.cb_batch.isChecked():
            if self.rb_combat.isChecked():
                self.data_objects = self.combat_batch_correction()
        
        self.populate_color_combo()
        self.compute_embedding()
        self.plot_embedding()
        
        self.selected_indices = selected_indices
        if len(self.selected_indices) > 0:
            self.update_line_plot()

    def combat_batch_correction(self):
        """
        Applies ComBat batch correction to the intensity values of all spectra in the dataset.
        This function modifies the dataset's intensities while keeping Raman shift and metadata intact.
        """
        if not self.data_objects:
            print("Dataset is empty. No batch correction applied.")
            return

        # Extract intensity values and batch labels (using 'date' as batch label).
        intensity_matrix = np.array([spectrum.intensity for spectrum in self.data_objects])
        batch_labels = pd.Series([spectrum.metadata['date'] for spectrum in self.data_objects])  # Convert to Pandas Series

        # Convert batch labels to categorical numeric labels.
        batch_categories = pd.factorize(batch_labels)[0]

        # Transpose data so that features are rows.
        data_transposed = pd.DataFrame(intensity_matrix.T)  # Convert NumPy array to DataFrame

        print("Data transposed shape:", data_transposed.shape)
        print("Batch categories shape:", len(batch_categories))

        # Apply ComBat for batch effect correction.
        corrected_data_transposed = pycombat(data_transposed, batch_categories)  # Now using a DataFrame

        # Convert back to NumPy array and transpose to original shape.
        corrected_intensity_matrix = corrected_data_transposed.to_numpy().T  # Convert DataFrame to NumPy and transpose back

        # Replace original intensity values while keeping metadata and Raman shift intact.
        for i, spectrum in enumerate(self.data_objects):
            self.data_objects[i].intensity = corrected_intensity_matrix[i]  # Update intensity

        print("Batch effect correction using ComBat has been applied successfully.")
        return self.data_objects  # Return corrected dataset

    def compute_pca_metadata_correlation_heatmap(self):
        """
        Perform PCA on intensity data and compute adjusted R² values between PCs and metadata variables.

        Parameters:
        - spectra_list (List[SpectraData]): List of SpectraData objects.
        - n_components (int): Number of principal components to retain.

        Returns:
        - None (Displays a heatmap of adjusted R² values)
        """
        n_components = 5

        # Extract intensity data and metadata
        intensity_data = np.array([spectrum.intensity for spectrum in self.data_objects])  # Shape: (samples, features)
        metadata_list = [spectrum.metadata for spectrum in self.data_objects]

        # Convert metadata to DataFrame
        metadata_df = pd.DataFrame(metadata_list)

        # Keep date and patient_id as numerical values
        numeric_cols = ['date', 'patient_id', 'ring', 'line', 'staging', 'spectrum_id']
        metadata_encoded = metadata_df[numeric_cols].copy()
        metadata_encoded[numeric_cols] = metadata_df[numeric_cols].astype(float)

        # Perform PCA
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(intensity_data)
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(data_scaled)
        pc_df = pd.DataFrame(principal_components, columns=[f'PC{i+1}' for i in range(n_components)])

        # Function to calculate adjusted R²
        def adjusted_r2_score(y_true, y_pred, n, p):
            r2 = r2_score(y_true, y_pred)
            return 1 - (1 - r2) * (n - 1) / (n - p - 1)

        # Compute adjusted R² values
        adj_r2_results = np.zeros((metadata_encoded.shape[1], n_components))

        for i, meta_col in enumerate(metadata_encoded.columns):
            for j in range(n_components):
                model = LinearRegression()
                X = metadata_encoded[[meta_col]]
                y = pc_df.iloc[:, j]
                model.fit(X, y)
                y_pred = model.predict(X)
                adj_r2_results[i, j] = adjusted_r2_score(y, y_pred, len(y), X.shape[1])

        # Convert to DataFrame
        adj_r2_df = pd.DataFrame(adj_r2_results, index=metadata_encoded.columns, columns=pc_df.columns)

        # Plot heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(adj_r2_df, annot=True, cmap='RdYlGn', center=0)
        plt.title("Adjusted R² of Association between PCs and Metadata")
        plt.xlabel("Principal Components")
        plt.ylabel("Metadata Variables")
        plt.show()


    def compute_embedding(self):
        key = (self.embedding_method, self.embedding_dim)
        if key in self.embedding_cache:
            self.embedding_result = self.embedding_cache[key]
            return
        X = np.array([obj.intensity for obj in self.data_objects])
        n_components = self.embedding_dim
        if self.embedding_method == "T-SNE":
            embedding = TSNE(n_components=n_components, random_state=42).fit_transform(X)
        else:
            embedding = PCA(n_components=n_components).fit_transform(X)
        self.embedding_cache[key] = embedding
        self.embedding_result = embedding

    def plot_embedding(self):
        self.fig_scatter.clear()
        if self.embedding_dim == 2:
            self.ax_scatter = self.fig_scatter.add_subplot(111)
        else:
            self.ax_scatter = self.fig_scatter.add_subplot(111, projection='3d')
        if self.embedding_result is None:
            return
        if self.color_by == "None":
            point_colors = None
        elif self.color_by == "outlier":
            outlier_labels = [1 if idx in self.outlier_indices else 0 for idx in range(len(self.data_objects))]
            cmap = ListedColormap(['blue', 'red'])
            norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
            point_colors = outlier_labels
        else:
            values = [obj.metadata.get(self.color_by, None) for obj in self.data_objects]
            unique_vals = sorted(set(values))
            cmap = plt.cm.get_cmap('cool', len(unique_vals))
            point_colors = [unique_vals.index(v) for v in values]
            cat_unique = unique_vals
        if self.embedding_dim == 2:
            if self.color_by == "None":
                self.ax_scatter.scatter(
                    self.embedding_result[:, 0],
                    self.embedding_result[:, 1],
                    color='C0',
                    alpha=0.6,
                    picker=True
                )
            elif self.color_by == "outlier":
                sc = self.ax_scatter.scatter(
                    self.embedding_result[:, 0],
                    self.embedding_result[:, 1],
                    c=point_colors,
                    cmap=cmap,
                    norm=norm,
                    alpha=0.6,
                    picker=True
                )
                cbar = self.fig_scatter.colorbar(sc, ax=self.ax_scatter, ticks=[0, 1])
                cbar.ax.set_yticklabels(['normal', 'outlier'])
            else:
                sc = self.ax_scatter.scatter(
                    self.embedding_result[:, 0],
                    self.embedding_result[:, 1],
                    c=point_colors,
                    cmap=cmap,
                    alpha=0.6,
                    picker=True
                )
                cbar = self.fig_scatter.colorbar(sc, ax=self.ax_scatter, ticks=range(len(cat_unique)))
                cbar.set_ticklabels([str(v) for v in cat_unique])
                cbar.ax.tick_params(labelsize=10)
                cbar.set_label(self.color_by)
            self.ax_scatter.set_title(f"2D {self.embedding_method} of Spectra Intensities")
            self.canvas_scatter.draw()
            if self.btn_select.isChecked():
                self.lasso = LassoSelector(self.ax_scatter, onselect=self.onselect)
                if hasattr(self.lasso, 'line'):
                    self.lasso.line.set_color('gold')
        else:
            if self.color_by == "None":
                self.ax_scatter.scatter(
                    self.embedding_result[:, 0],
                    self.embedding_result[:, 1],
                    self.embedding_result[:, 2],
                    color='C0',
                    alpha=0.6,
                    picker=True
                )
            elif self.color_by == "outlier":
                sc = self.ax_scatter.scatter(
                    self.embedding_result[:, 0],
                    self.embedding_result[:, 1],
                    self.embedding_result[:, 2],
                    c=point_colors,
                    cmap=cmap,
                    norm=norm,
                    alpha=0.6,
                    picker=True
                )
                cbar = self.fig_scatter.colorbar(sc, ax=self.ax_scatter, ticks=[0, 1])
                cbar.ax.set_yticklabels(['normal', 'outlier'])
            else:
                sc = self.ax_scatter.scatter(
                    self.embedding_result[:, 0],
                    self.embedding_result[:, 1],
                    self.embedding_result[:, 2],
                    c=point_colors,
                    cmap=cmap,
                    alpha=0.6,
                    picker=True
                )
                cbar = self.fig_scatter.colorbar(sc, ax=self.ax_scatter, ticks=range(len(cat_unique)))
                cbar.set_ticklabels([str(v) for v in cat_unique])
                cbar.ax.tick_params(labelsize=10)
                cbar.set_label(self.color_by)
            self.ax_scatter.set_title(f"3D {self.embedding_method} of Spectra Intensities")
            self.canvas_scatter.draw()

    def reset_view(self):
        self.plot_embedding()

    def toggle_lasso(self, checked):
        if self.embedding_dim != 2:
            if self.lasso is not None:
                self.lasso.disconnect_events()
                self.lasso = None
            return
        if checked:
            if self.lasso is None:
                self.lasso = LassoSelector(self.ax_scatter, onselect=self.onselect)
                if hasattr(self.lasso, 'line'):
                    self.lasso.line.set_color('gold')
        else:
            if self.lasso is not None:
                self.lasso.disconnect_events()
                self.lasso = None

    def on_color_change(self, index):
        self.color_by = self.color_combo.currentText()
        self.plot_embedding()

    def on_embedding_change(self, index):
        self.embedding_method = self.embedding_combo.currentText()
        self.clear_selection()
        self.compute_embedding()
        self.plot_embedding()

    def on_dimension_change(self, index):
        text = self.dim_combo.currentText()
        self.embedding_dim = 2 if text == "2D" else 3
        if self.embedding_dim != 2 and self.lasso is not None:
            self.lasso.disconnect_events()
            self.lasso = None
        self.clear_selection()
        self.compute_embedding()
        self.plot_embedding()

    def onselect(self, verts):
        if self.embedding_dim != 2:
            return
        path = Path(verts)
        ind = np.nonzero(path.contains_points(self.embedding_result))[0]
        self.selected_indices = ind.tolist()  # Convert to list.
        self.update_line_plot()


    def on_scatter_pick(self, event):
        if self.embedding_dim != 3:
            return
        if not self.btn_select.isChecked():
            return
        if hasattr(event, 'ind') and len(event.ind) > 0:
            picked_index = event.ind[0]
            if picked_index not in self.selected_indices:
                self.selected_indices.append(picked_index)
            self.update_line_plot()

    def on_scroll(self, event):
        ax = event.inaxes
        if ax is None:
            return
        if self.embedding_dim == 2:
            cur_xlim = ax.get_xlim()
            cur_ylim = ax.get_ylim()
            center_x = (cur_xlim[0] + cur_xlim[1]) / 2
            center_y = (cur_ylim[0] + cur_ylim[1]) / 2
            scale_factor = 0.9 if event.button == 'up' else 1.1
            new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
            new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
            ax.set_xlim([center_x - new_width/2, center_x + new_width/2])
            ax.set_ylim([center_y - new_height/2, center_y + new_height/2])
        else:
            if hasattr(ax, 'dist'):
                if event.button == 'up':
                    ax.dist *= 0.9
                elif event.button == 'down':
                    ax.dist *= 1.1
        self.canvas_scatter.draw_idle()

    def on_pan_press(self, event):
        if self.embedding_dim != 2:
            return
        if self.btn_select.isChecked():
            return
        if event.inaxes != self.ax_scatter:
            return
        if event.button != 1:
            return
        self._pan_active = True
        self._pan_press_event = event

    def on_pan_motion(self, event):
        if self.embedding_dim != 2:
            return
        if not self._pan_active:
            return
        if event.inaxes != self.ax_scatter:
            return
        dx = event.xdata - self._pan_press_event.xdata
        dy = event.ydata - self._pan_press_event.ydata
        cur_xlim = self.ax_scatter.get_xlim()
        cur_ylim = self.ax_scatter.get_ylim()
        self.ax_scatter.set_xlim(cur_xlim[0] - dx, cur_xlim[1] - dx)
        self.ax_scatter.set_ylim(cur_ylim[0] - dy, cur_ylim[1] - dy)
        self._pan_press_event = event
        self.canvas_scatter.draw_idle()

    def on_pan_release(self, event):
        if self.embedding_dim != 2:
            return
        self._pan_active = False
        self._pan_press_event = None

    def update_line_plot(self):
        self.ax_line.clear()
        self.selected_index_to_line = {}
        self.metadata_list.clear()
        if len(self.selected_indices) == 0:
            self.ax_line.set_title("Select points to view spectra")
        else:
            for idx in self.selected_indices:
                obj = self.data_objects[idx]
                line, = self.ax_line.plot(obj.raman_shift_cm, obj.intensity, color='C0', picker=5)
                self.selected_index_to_line[idx] = line
                item = QListWidgetItem(str(obj.metadata))
                item.setData(Qt.UserRole, idx)
                self.metadata_list.addItem(item)
            self.ax_line.set_title("Spectra for Selected Points")
            self.ax_line.set_xlabel("Raman Shift")
            self.ax_line.set_ylabel("Intensity")
        self.canvas_line.draw_idle()

    def clear_selection(self):
        self.selected_indices = []
        self.ax_line.clear()
        self.ax_line.set_title("Select points to view spectra")
        self.metadata_list.clear()
        self.selected_index_to_line = {}
        self.canvas_line.draw_idle()

    def on_line_hover(self, event):
        if event.inaxes != self.ax_line:
            self.hovered_index = None
            self.update_line_highlights()
            return
        hovered_index = None
        for idx, line in self.selected_index_to_line.items():
            contains, _ = line.contains(event)
            if contains:
                hovered_index = idx
                break
        self.hovered_index = hovered_index
        self.update_line_highlights()

    def on_line_pick(self, event):
        artist = event.artist
        picked_index = None
        for idx, line in self.selected_index_to_line.items():
            if artist == line:
                picked_index = idx
                break
        if picked_index is not None:
            self.metadata_list.clearSelection()
            for i in range(self.metadata_list.count()):
                item = self.metadata_list.item(i)
                if item.data(Qt.UserRole) == picked_index:
                    item.setSelected(True)
                    self.metadata_list.scrollToItem(item)
                    break
        self.update_line_highlights()

    def update_line_highlights(self):
        selected_items = self.metadata_list.selectedItems()
        selected_indices_from_list = [item.data(Qt.UserRole) for item in selected_items]
        for idx, line in self.selected_index_to_line.items():
            if idx == self.hovered_index or idx in selected_indices_from_list:
                line.set_linewidth(3)
                line.set_color('orange')
            else:
                line.set_linewidth(1)
                line.set_color('C0')
        self.canvas_line.draw_idle()

    def save_metadata(self):
        metadata_list = [self.data_objects[idx].metadata for idx in self.selected_indices]
        filename, _ = QFileDialog.getSaveFileName(self, "Save Metadata", "", "JSON Files (*.json)")
        if filename:
            with open(filename, "w") as f:
                json.dump(metadata_list, f, indent=4)

    def find_outliers(self):
        data_matrix = np.array([obj.intensity for obj in self.data_objects])
        outlier_indices = set()
        for iteration in range(self.n_iterations):
            print(f"Running PCA + DBSCAN Outlier Detection - Iteration {iteration+1}/{self.n_iterations}")
            pca_data = PCA().fit_transform(data_matrix)
            if iteration == 0:
                pca_data = pca_data[:, :2]
            else:
                pcstart, pcend = 3, 5
                pca_data = pca_data[:, pcstart:pcend]
            dbscan = DBSCAN(eps=self.dbscan_eps, min_samples=5)
            cluster_labels = dbscan.fit_predict(pca_data)
            iter_outliers = {idx for idx, label in enumerate(cluster_labels) if label == -1}
            outlier_indices.update(iter_outliers)
            print(f"Iteration {iteration+1} found {len(iter_outliers)} outliers.")
        self.outlier_indices = outlier_indices
        print(f"Total outliers found: {len(outlier_indices)}")
        
        if self.color_combo.findText("outlier") == -1:
            self.color_combo.addItem("outlier")
        self.color_combo.setCurrentText("outlier")
        self.plot_embedding()
        
        # Enable the Select Outliers button since we now have outlier information.
        self.btn_select_outliers.setEnabled(True)

    def select_outliers(self):
        # Set the selected indices to all indices flagged as outliers.
        if self.outlier_indices:
            self.selected_indices = list(self.outlier_indices)
            self.update_line_plot()


if __name__ == "__main__":
    num_samples = 50
    data_objects = []
    for i in range(num_samples):
        intensity = np.random.rand(100)
        raman_shift = np.linspace(100, 3000, 100)
        metadata = {
            "patient_id": np.random.randint(100, 600),
            "sample_type": "plasma",
            "date": "20230622",
            "position": str(np.random.randint(1, 20)),
            "staging": np.random.choice([0, 1, 2]),
            "gender": "Male",
            "race": "White",
            "spectrum_id": i
        }
        data_objects.append(SpectraData(intensity, raman_shift, metadata))
        
    preprocessor = SpectrumPreprocessor(cropping=True, 
                                        baseline_correction=False,
                                        remove_cosmic_rays=True,
                                        normalization=False,
                                        smoothing=False)
    app = QApplication(sys.argv)
    window = SpectraViewer(data_objects, preprocessor)
    window.resize(1300, 650)
    window.show()
    sys.exit(app.exec_())
