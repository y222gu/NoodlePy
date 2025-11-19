import sys
import json
import yaml
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QListWidget, QListWidgetItem, 
                             QFileDialog, QComboBox, QLabel, QGroupBox, QCheckBox, 
                             QDoubleSpinBox, QSpinBox, QRadioButton, QButtonGroup,
                             QScrollArea, QFormLayout, QLineEdit)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
from sklearn.manifold import TSNE
from combat.pycombat import pycombat
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
import copy
import os
# Import the actual Spectrum and SpectrumPreprocessor classes
from noodlepy.utils.spectrum import Spectrum
from noodlepy.utils.spectrumpreprocessor import SpectrumPreprocessor
from noodlepy.utils.izabelladataset import OC_Dataset
# Use the "fast" style.
plt.style.use('fast')

########################################
# Enhanced Preprocessor class wrapper
########################################

class SpectrumPreprocessorWrapper:
    """Wrapper to maintain compatibility with the UI while using actual SpectrumPreprocessor"""
    def __init__(self, 
                 cropping=True,
                 baseline_correction=False,
                 remove_cosmic_rays=True,
                 normalization=False,
                 smoothing=False,
                 # Cropping parameters
                 start_raman_shift_cm=662.697,
                 end_raman_shift_cm=1784.104,
                 # Baseline correction parameters (airPLS)
                 baseline_lam=1000,
                 baseline_diff_order=1,
                 baseline_max_iter=15,
                 baseline_tol=0.005,
                 # Cosmic ray removal parameters (modified z-scores)
                 cosmic_threshold=10,
                 # Smoothing parameters (Savitzky-Golay)
                 smooth_window_length=9,
                 smooth_polyorder=2,
                 # Normalization type
                 normalization_type='by_max',
                 # Config path for actual preprocessor
                 config_path=None):
        
        # Store UI parameters
        self.cropping = cropping
        self.baseline_correction = baseline_correction
        self.remove_cosmic_rays = remove_cosmic_rays
        self.normalization = normalization
        self.smoothing = smoothing
        
        self.start_raman_shift_cm = start_raman_shift_cm
        self.end_raman_shift_cm = end_raman_shift_cm
        self.baseline_lam = baseline_lam
        self.baseline_diff_order = baseline_diff_order
        self.baseline_max_iter = baseline_max_iter
        self.baseline_tol = baseline_tol
        self.cosmic_threshold = cosmic_threshold
        self.smooth_window_length = smooth_window_length
        self.smooth_polyorder = smooth_polyorder
        self.normalization_type = normalization_type
        
        # Create config file for the actual preprocessor
        self._create_temp_config()
        
        # Initialize the actual SpectrumPreprocessor
        self.actual_preprocessor = SpectrumPreprocessor(
            cropping=cropping,
            baseline_correction=baseline_correction,
            remove_cosmic_rays=remove_cosmic_rays,
            normalization=normalization,
            smoothing=smoothing,
            config_path=self.temp_config_path
        )

    def _create_temp_config(self):
        """Create a temporary config file for the actual preprocessor"""
        config = {
            'preprocessing': {
                'cropping': {
                    'start_raman_shift_cm': self.start_raman_shift_cm,
                    'end_raman_shift_cm': self.end_raman_shift_cm
                },
                'baseline_correction': {
                    'lam': self.baseline_lam,
                    'diff_order': self.baseline_diff_order,
                    'max_iter': self.baseline_max_iter,
                    'tol': self.baseline_tol
                },
                'cosmic_rays_removal': {
                    'threshold': self.cosmic_threshold
                },
                'smoothing': {
                    'window_length': self.smooth_window_length,
                    'polyorder': self.smooth_polyorder
                },
                'normalization_type': self.normalization_type
            }
        }
        
        # Save to temp file
        import tempfile
        self.temp_config_path = tempfile.mktemp(suffix='.yml')
        with open(self.temp_config_path, 'w') as f:
            yaml.dump(config, f)

    def preprocess(self, spectrum: Spectrum) -> Spectrum:
        """Preprocess using the actual SpectrumPreprocessor with robust cropping"""
        try:
            # Make a deep copy to avoid modifying the original
            preprocessed = copy.deepcopy(spectrum)
            
            # Manual cropping with approximate matching
            if self.cropping:
                mask = (preprocessed.raman_shift_cm >= self.start_raman_shift_cm) & \
                       (preprocessed.raman_shift_cm <= self.end_raman_shift_cm)
                
                if not np.any(mask):
                    print(f"Warning: Cropping range [{self.start_raman_shift_cm}, {self.end_raman_shift_cm}] "
                          f"outside spectrum range [{preprocessed.raman_shift_cm.min()}, {preprocessed.raman_shift_cm.max()}]")
                    # Skip cropping if range is invalid
                else:
                    preprocessed.raman_shift_cm = preprocessed.raman_shift_cm[mask]
                    preprocessed.intensity = preprocessed.intensity[mask]
            
            # Apply other preprocessing steps
            if self.remove_cosmic_rays:
                preprocessed.remove_cosmic_rays(threshold=self.cosmic_threshold)
            
            if self.baseline_correction:
                preprocessed.airPLS(
                    lam=self.baseline_lam,
                    diff_order=self.baseline_diff_order,
                    max_iter=self.baseline_max_iter,
                    tol=self.baseline_tol
                )
            
            if self.normalization:
                preprocessed.normalize_spectrum(self.normalization_type)
            
            if self.smoothing:
                preprocessed.savgol_filter(
                    window_length=self.smooth_window_length,
                    polyorder=self.smooth_polyorder
                )
            
            return preprocessed
            
        except Exception as e:
            print(f"Error preprocessing spectrum: {e}")
            import traceback
            traceback.print_exc()
            # Return a copy of original on error
            return copy.deepcopy(spectrum)
    
    def to_dict(self):
        """Export preprocessor parameters to dictionary matching YAML structure"""
        return {
            'preprocessing': {
                'cropping': {
                    'start_raman_shift_cm': self.start_raman_shift_cm,
                    'end_raman_shift_cm': self.end_raman_shift_cm
                },
                'baseline_correction': {
                    'lam': self.baseline_lam,
                    'diff_order': self.baseline_diff_order,
                    'max_iter': self.baseline_max_iter,
                    'tol': self.baseline_tol
                },
                'cosmic_rays_removal': {
                    'threshold': self.cosmic_threshold
                },
                'smoothing': {
                    'window_length': self.smooth_window_length,
                    'polyorder': self.smooth_polyorder
                },
                'normalization_type': self.normalization_type
            },
            'enabled': {
                'cropping': self.cropping,
                'baseline_correction': self.baseline_correction,
                'remove_cosmic_rays': self.remove_cosmic_rays,
                'normalization': self.normalization,
                'smoothing': self.smoothing
            }
        }
    
    @classmethod
    def from_dict(cls, config):
        """Create preprocessor from dictionary (YAML structure)"""
        preproc = config.get('preprocessing', {})
        enabled = config.get('enabled', {})
        
        cropping_params = preproc.get('cropping', {})
        baseline_params = preproc.get('baseline_correction', {})
        cosmic_params = preproc.get('cosmic_rays_removal', {})
        smoothing_params = preproc.get('smoothing', {})
        norm_type = preproc.get('normalization_type', 'by_max')
        
        return cls(
            cropping=enabled.get('cropping', True),
            baseline_correction=enabled.get('baseline_correction', False),
            remove_cosmic_rays=enabled.get('remove_cosmic_rays', True),
            normalization=enabled.get('normalization', False),
            smoothing=enabled.get('smoothing', False),
            start_raman_shift_cm=cropping_params.get('start_raman_shift_cm', 662.697),
            end_raman_shift_cm=cropping_params.get('end_raman_shift_cm', 1784.104),
            baseline_lam=baseline_params.get('lam', 1000),
            baseline_diff_order=baseline_params.get('diff_order', 1),
            baseline_max_iter=baseline_params.get('max_iter', 15),
            baseline_tol=baseline_params.get('tol', 0.005),
            cosmic_threshold=cosmic_params.get('threshold', 10),
            smooth_window_length=smoothing_params.get('window_length', 9),
            smooth_polyorder=smoothing_params.get('polyorder', 2),
            normalization_type=norm_type
        )



########################################
# Main SpectraViewer Window
########################################



class SpectraViewer(QMainWindow):
    def __init__(self, data_objects, preprocessor, config_file=None):
        super().__init__()
        self.setWindowTitle("Spectra Viewer - Enhanced")
        self.original_data_objects = data_objects  
        self.preprocessor = preprocessor
        self.config_file = config_file
        
        # Preprocess data with error handling
        print(f"Preprocessing {len(data_objects)} spectra...")
        self.data_objects = []
        for i, obj in enumerate(self.original_data_objects):
            try:
                processed = self.preprocessor.preprocess(obj)
                self.data_objects.append(processed)
            except Exception as e:
                print(f"Error preprocessing spectrum {i}: {e}")
                # Keep original if preprocessing fails
                self.data_objects.append(copy.deepcopy(obj))
        
        print(f"Successfully preprocessed {len(self.data_objects)} spectra")
        
        # Align all spectra to common grid
        self.align_spectra_to_common_grid()
        
        # Validate that we have data
        if len(self.data_objects) == 0:
            print("ERROR: No data objects available after preprocessing!")
        else:
            # Check a sample of the data
            sample_obj = self.data_objects[0]
            print(f"Sample spectrum shape: intensity={len(sample_obj.intensity)}, raman={len(sample_obj.raman_shift_cm)}")
            if len(sample_obj.intensity) > 0:
                print(f"Raman shift range: [{sample_obj.raman_shift_cm.min():.2f}, {sample_obj.raman_shift_cm.max():.2f}]")
        
        self.selected_indices = []
        self.selected_index_to_line = {}
        self.hovered_index = None
        self.scatter_highlight = None

        self.embedding_method = "T-SNE"
        self.embedding_dim = 2
        self.embedding_result = None
        self.embedding_cache = {}
        self.outlier_indices = set()

        # Parameters for outlier detection
        self.n_iterations = 3
        self.dbscan_eps = 0.5
        self.dbscan_min = 5

        self.color_by = "None"
        self.lasso = None
        
        self._pan_active = False
        self._pan_press_event = None
        
        # Plotting mode: 'grouped' (by patient) or 'combined' (all in one plot)
        self.plot_mode = 'grouped'

        self.initUI()
        self.populate_color_combo()
        self.compute_embedding()
        self.plot_embedding()

    def align_spectra_to_common_grid(self):
        """Align all spectra to a common wavenumber grid using interpolation"""
        if not self.data_objects:
            return
        
        # Find the common wavenumber range across all spectra
        min_raman = max(obj.raman_shift_cm.min() for obj in self.data_objects)
        max_raman = min(obj.raman_shift_cm.max() for obj in self.data_objects)
        
        # Find the spectrum with the most points to use as reference
        reference_spectrum = max(self.data_objects, key=lambda obj: len(obj.raman_shift_cm))
        
        # Create a common grid within the overlapping range
        reference_mask = (reference_spectrum.raman_shift_cm >= min_raman) & \
                        (reference_spectrum.raman_shift_cm <= max_raman)
        common_grid = reference_spectrum.raman_shift_cm[reference_mask]
        
        print(f"Aligning spectra to common grid: {len(common_grid)} points, "
              f"range [{min_raman:.2f}, {max_raman:.2f}] cm⁻¹")
        
        # Interpolate all spectra to the common grid
        for obj in self.data_objects:
            if len(obj.raman_shift_cm) != len(common_grid) or \
               not np.allclose(obj.raman_shift_cm, common_grid):
                # Need to interpolate
                interpolated_intensity = np.interp(common_grid, obj.raman_shift_cm, obj.intensity)
                obj.raman_shift_cm = common_grid.copy()
                obj.intensity = interpolated_intensity

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # ==== COLUMN 1: Preprocessing and DBSCAN ====
        left_layout = QVBoxLayout()
        
        # ===== Enhanced Preprocessing Options Group Box =====
        preproc_group = QGroupBox("Preprocessing Options")
        preproc_main_layout = QVBoxLayout()
        
        # Create scrollable area for preprocessing parameters
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumWidth(350)
        scroll_widget = QWidget()
        preproc_layout = QVBoxLayout(scroll_widget)
        
        # ==== CROPPING ====
        self.cb_cropping = QCheckBox("Cropping")
        self.cb_cropping.setChecked(self.preprocessor.cropping)
        self.cb_cropping.stateChanged.connect(self.toggle_cropping_params)
        preproc_layout.addWidget(self.cb_cropping)
        
        self.crop_params_widget = QWidget()
        crop_params_layout = QFormLayout(self.crop_params_widget)
        crop_params_layout.setContentsMargins(20, 0, 0, 0)
        
        self.start_raman_spin = QDoubleSpinBox()
        self.start_raman_spin.setRange(0.0, 5000.0)
        self.start_raman_spin.setSingleStep(0.1)
        self.start_raman_spin.setDecimals(3)
        self.start_raman_spin.setValue(self.preprocessor.start_raman_shift_cm)
        crop_params_layout.addRow("Start Raman shift (cm⁻¹):", self.start_raman_spin)
        
        self.end_raman_spin = QDoubleSpinBox()
        self.end_raman_spin.setRange(0.0, 5000.0)
        self.end_raman_spin.setSingleStep(0.1)
        self.end_raman_spin.setDecimals(3)
        self.end_raman_spin.setValue(self.preprocessor.end_raman_shift_cm)
        crop_params_layout.addRow("End Raman shift (cm⁻¹):", self.end_raman_spin)
        
        preproc_layout.addWidget(self.crop_params_widget)
        
        # ==== BASELINE CORRECTION (airPLS) ====
        self.cb_baseline = QCheckBox("Baseline Correction (airPLS)")
        self.cb_baseline.setChecked(self.preprocessor.baseline_correction)
        self.cb_baseline.stateChanged.connect(self.toggle_baseline_params)
        preproc_layout.addWidget(self.cb_baseline)
        
        self.baseline_params_widget = QWidget()
        baseline_params_layout = QFormLayout(self.baseline_params_widget)
        baseline_params_layout.setContentsMargins(20, 0, 0, 0)
        
        self.baseline_lam_spin = QDoubleSpinBox()
        self.baseline_lam_spin.setRange(1.0, 1000000.0)
        self.baseline_lam_spin.setSingleStep(100)
        self.baseline_lam_spin.setValue(self.preprocessor.baseline_lam)
        baseline_params_layout.addRow("Lambda (λ):", self.baseline_lam_spin)
        
        self.baseline_diff_order_spin = QSpinBox()
        self.baseline_diff_order_spin.setRange(1, 3)
        self.baseline_diff_order_spin.setValue(self.preprocessor.baseline_diff_order)
        baseline_params_layout.addRow("Diff Order:", self.baseline_diff_order_spin)
        
        self.baseline_max_iter_spin = QSpinBox()
        self.baseline_max_iter_spin.setRange(1, 100)
        self.baseline_max_iter_spin.setValue(self.preprocessor.baseline_max_iter)
        baseline_params_layout.addRow("Max Iterations:", self.baseline_max_iter_spin)
        
        self.baseline_tol_spin = QDoubleSpinBox()
        self.baseline_tol_spin.setRange(0.0001, 0.1)
        self.baseline_tol_spin.setSingleStep(0.001)
        self.baseline_tol_spin.setDecimals(4)
        self.baseline_tol_spin.setValue(self.preprocessor.baseline_tol)
        baseline_params_layout.addRow("Tolerance:", self.baseline_tol_spin)
        
        preproc_layout.addWidget(self.baseline_params_widget)
        
        # ==== COSMIC RAY REMOVAL (Modified Z-scores) ====
        self.cb_cosmic = QCheckBox("Remove Cosmic Rays (Modified Z-scores)")
        self.cb_cosmic.setChecked(self.preprocessor.remove_cosmic_rays)
        self.cb_cosmic.stateChanged.connect(self.toggle_cosmic_params)
        preproc_layout.addWidget(self.cb_cosmic)
        
        self.cosmic_params_widget = QWidget()
        cosmic_params_layout = QFormLayout(self.cosmic_params_widget)
        cosmic_params_layout.setContentsMargins(20, 0, 0, 0)
        
        self.cosmic_threshold_spin = QDoubleSpinBox()
        self.cosmic_threshold_spin.setRange(1.0, 50.0)
        self.cosmic_threshold_spin.setSingleStep(0.5)
        self.cosmic_threshold_spin.setValue(self.preprocessor.cosmic_threshold)
        cosmic_params_layout.addRow("Threshold:", self.cosmic_threshold_spin)
        
        preproc_layout.addWidget(self.cosmic_params_widget)
        
        # ==== SMOOTHING (Savitzky-Golay) ====
        self.cb_smooth = QCheckBox("Smoothing (Savitzky-Golay)")
        self.cb_smooth.setChecked(self.preprocessor.smoothing)
        self.cb_smooth.stateChanged.connect(self.toggle_smooth_params)
        preproc_layout.addWidget(self.cb_smooth)
        
        self.smooth_params_widget = QWidget()
        smooth_params_layout = QFormLayout(self.smooth_params_widget)
        smooth_params_layout.setContentsMargins(20, 0, 0, 0)
        
        self.smooth_window_spin = QSpinBox()
        self.smooth_window_spin.setRange(3, 51)
        self.smooth_window_spin.setSingleStep(2)
        self.smooth_window_spin.setValue(self.preprocessor.smooth_window_length)
        smooth_params_layout.addRow("Window Length:", self.smooth_window_spin)
        
        self.smooth_polyorder_spin = QSpinBox()
        self.smooth_polyorder_spin.setRange(1, 5)
        self.smooth_polyorder_spin.setValue(self.preprocessor.smooth_polyorder)
        smooth_params_layout.addRow("Poly Order:", self.smooth_polyorder_spin)
        
        preproc_layout.addWidget(self.smooth_params_widget)
        
        # ==== NORMALIZATION ====
        self.cb_norm = QCheckBox("Normalization")
        self.cb_norm.setChecked(self.preprocessor.normalization)
        self.cb_norm.stateChanged.connect(self.toggle_norm_params)
        preproc_layout.addWidget(self.cb_norm)
        
        self.norm_params_widget = QWidget()
        norm_params_layout = QFormLayout(self.norm_params_widget)
        norm_params_layout.setContentsMargins(20, 0, 0, 0)
        
        self.norm_type_combo = QComboBox()
        self.norm_type_combo.addItems(['by_max', 'by_area'])
        self.norm_type_combo.setCurrentText(self.preprocessor.normalization_type)
        norm_params_layout.addRow("Type:", self.norm_type_combo)
        
        preproc_layout.addWidget(self.norm_params_widget)
        
        # ==== BATCH CORRECTION ====
        self.cb_batch = QCheckBox("Apply Batch Correction")
        self.cb_batch.setChecked(False)
        preproc_layout.addWidget(self.cb_batch)
        
        self.rbg_batch = QButtonGroup(self)
        self.rb_combat = QRadioButton("ComBat")
        self.rb_combat.setChecked(True)
        self.rbg_batch.addButton(self.rb_combat)
        
        batch_radio_layout = QVBoxLayout()
        batch_radio_layout.addWidget(self.rb_combat)
        batch_radio_column_layout = QVBoxLayout()
        batch_radio_column_layout.addLayout(batch_radio_layout)
        batch_radio_column_layout.setContentsMargins(20, 0, 0, 0)
        
        preproc_layout.addLayout(batch_radio_column_layout)
        
        scroll_area.setWidget(scroll_widget)
        preproc_main_layout.addWidget(scroll_area)
        
        # Config management buttons
        config_buttons_layout = QHBoxLayout()
        
        self.btn_load_config = QPushButton("Load Config")
        self.btn_load_config.clicked.connect(self.load_config)
        config_buttons_layout.addWidget(self.btn_load_config)
        
        self.btn_save_config = QPushButton("Save Config")
        self.btn_save_config.clicked.connect(self.save_config)
        config_buttons_layout.addWidget(self.btn_save_config)
        
        preproc_main_layout.addLayout(config_buttons_layout)
        
        self.btn_recalculate = QPushButton("Re-process")
        self.btn_recalculate.clicked.connect(self.update_preprocessing)
        preproc_main_layout.addWidget(self.btn_recalculate)

        preproc_group.setLayout(preproc_main_layout)
        
        # Initialize parameter widget visibility
        self.toggle_cropping_params()
        self.toggle_baseline_params()
        self.toggle_cosmic_params()
        self.toggle_norm_params()
        self.toggle_smooth_params()
        
        left_layout.addWidget(preproc_group)

        # DBSCAN parameter entries group box
        dbscan_group = QGroupBox("DBSCAN Parameters")
        dbscan_layout = QVBoxLayout()

        # Optimization range for eps
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

        # Optimization range for min_samples
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

        # Optimize DBSCAN button
        self.btn_optimize_dbscan = QPushButton("Optimize DBSCAN")
        self.btn_optimize_dbscan.clicked.connect(self.optimize_dbscan_params)
        dbscan_layout.addWidget(self.btn_optimize_dbscan)

        # Display selected eps and min_samples
        self.dbscan_params_label = QLabel(f"eps: {self.dbscan_eps}, min_samples: {self.dbscan_min}")
        dbscan_layout.addWidget(self.dbscan_params_label)

        # Number of iterations spin box
        iter_layout = QHBoxLayout()
        iter_label = QLabel("Outlier Detection Iterations:")
        self.n_iter_spin = QSpinBox()
        self.n_iter_spin.setRange(1, 10)
        self.n_iter_spin.setValue(self.n_iterations)
        iter_layout.addWidget(iter_label)
        iter_layout.addWidget(self.n_iter_spin)
        dbscan_layout.addLayout(iter_layout)

        # Outlier Detection button
        self.btn_find_outliers = QPushButton("Find Outliers")
        self.btn_find_outliers.clicked.connect(self.find_outliers)
        dbscan_layout.addWidget(self.btn_find_outliers)

        # Select Outliers button
        self.btn_select_outliers = QPushButton("Select All Outliers")
        self.btn_select_outliers.setEnabled(False)
        self.btn_select_outliers.clicked.connect(self.select_outliers)
        dbscan_layout.addWidget(self.btn_select_outliers)

        dbscan_group.setLayout(dbscan_layout)
        left_layout.addWidget(dbscan_group)
        
        # Add left column to main layout
        main_layout.addLayout(left_layout, stretch=1)
        
        # ==== COLUMN 2: Embedding, Color, Metadata, and Scatter Plot ====
        middle_layout = QVBoxLayout()

        # Embedding controls
        embedding_group = QGroupBox("Embedding Options")
        embedding_layout = QVBoxLayout()
        self.combo_dim = QComboBox()
        self.combo_dim.addItems(["2D", "3D"])
        self.combo_dim.currentTextChanged.connect(self.update_dim)
        embedding_layout.addWidget(QLabel("Dimensionality:"))
        embedding_layout.addWidget(self.combo_dim)
        
        self.combo_method = QComboBox()
        self.combo_method.addItems(["T-SNE", "PCA"])
        self.combo_method.currentTextChanged.connect(self.update_method)
        embedding_layout.addWidget(QLabel("Method:"))
        embedding_layout.addWidget(self.combo_method)
        embedding_group.setLayout(embedding_layout)
        middle_layout.addWidget(embedding_group)

        # Color options
        color_group = QGroupBox("Color Options")
        color_layout = QVBoxLayout()
        self.color_combo = QComboBox()
        self.color_combo.currentTextChanged.connect(self.plot_embedding)
        color_layout.addWidget(QLabel("Color By:"))
        color_layout.addWidget(self.color_combo)
        color_group.setLayout(color_layout)
        middle_layout.addWidget(color_group)

        # Metadata list
        metadata_group = QGroupBox("Selected Spectra Metadata")
        metadata_layout = QVBoxLayout()
        self.metadata_list = QListWidget()
        self.metadata_list.itemSelectionChanged.connect(self.update_line_highlights)
        metadata_layout.addWidget(self.metadata_list)
        btn_save = QPushButton("Save Metadata")
        btn_save.clicked.connect(self.save_metadata)
        metadata_layout.addWidget(btn_save)
        metadata_group.setLayout(metadata_layout)
        middle_layout.addWidget(metadata_group)

        # Clear selection button
        btn_clear = QPushButton("Clear Selection")
        btn_clear.clicked.connect(self.clear_selection)
        middle_layout.addWidget(btn_clear)
        
        # Toggle button for lasso selection
        self.btn_select = QPushButton("Enable Lasso Selection")
        self.btn_select.setCheckable(True)
        self.btn_select.toggled.connect(self.toggle_lasso)
        middle_layout.addWidget(self.btn_select)

        # Embedding scatter plot
        self.fig_scatter = Figure(figsize=(5,4))
        self.ax_scatter = self.fig_scatter.add_subplot(111)
        self.canvas_scatter = FigureCanvas(self.fig_scatter)
        self.canvas_scatter.mpl_connect('scroll_event', self.on_scroll)
        self.canvas_scatter.mpl_connect('button_press_event', self.on_pan_press)
        self.canvas_scatter.mpl_connect('motion_notify_event', self.on_pan_motion)
        self.canvas_scatter.mpl_connect('button_release_event', self.on_pan_release)
        middle_layout.addWidget(self.canvas_scatter, stretch=1)
        
        # Add middle column to main layout
        main_layout.addLayout(middle_layout, stretch=1)

        # ==== COLUMN 3: Spectra Plot (needs vertical space) ====
        right_layout = QVBoxLayout()
        
        # Add toggle button for plot mode at the top
        plot_mode_layout = QHBoxLayout()
        plot_mode_layout.addStretch()
        self.btn_toggle_plot_mode = QPushButton("Combined View")
        self.btn_toggle_plot_mode.setCheckable(True)
        self.btn_toggle_plot_mode.setChecked(False)
        self.btn_toggle_plot_mode.toggled.connect(self.toggle_plot_mode)
        self.btn_toggle_plot_mode.setMaximumWidth(150)
        plot_mode_layout.addWidget(self.btn_toggle_plot_mode)
        right_layout.addLayout(plot_mode_layout)

        # Line plot for spectra with patient-wise subplots
        self.fig_line = Figure(figsize=(8,10))
        self.canvas_line = FigureCanvas(self.fig_line)
        self.canvas_line.mpl_connect('motion_notify_event', self.on_line_hover)
        self.canvas_line.mpl_connect('pick_event', self.on_line_pick)
        self.canvas_line.mpl_connect('button_press_event', self.on_line_click)
        right_layout.addWidget(self.canvas_line, stretch=1)


        # Add right column to main layout
        main_layout.addLayout(right_layout, stretch=2)

    def toggle_cropping_params(self):
        self.crop_params_widget.setVisible(self.cb_cropping.isChecked())
    
    def toggle_baseline_params(self):
        self.baseline_params_widget.setVisible(self.cb_baseline.isChecked())
    
    def toggle_cosmic_params(self):
        self.cosmic_params_widget.setVisible(self.cb_cosmic.isChecked())
    
    def toggle_norm_params(self):
        self.norm_params_widget.setVisible(self.cb_norm.isChecked())
    
    def toggle_smooth_params(self):
        self.smooth_params_widget.setVisible(self.cb_smooth.isChecked())

    def toggle_plot_mode(self, checked):
        """Toggle between grouped (by patient) and combined plot mode"""
        if checked:
            self.plot_mode = 'combined'
            self.btn_toggle_plot_mode.setText("Grouped View")
        else:
            self.plot_mode = 'grouped'
            self.btn_toggle_plot_mode.setText("Combined View")
        
        # Update the plot if there are selected spectra
        if self.selected_indices:
            self.update_line_plot()

    def load_config(self):
        """Load preprocessing configuration from YAML or JSON file"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Config", "", "Config Files (*.yaml *.yml *.json)"
        )
        if filename:
            with open(filename, 'r') as f:
                if filename.endswith(('.yaml', '.yml')):
                    config = yaml.safe_load(f)
                else:
                    config = json.load(f)
            
            # Update preprocessor
            self.preprocessor = SpectrumPreprocessor.from_dict(config)
            
            # Update UI elements
            self.cb_cropping.setChecked(self.preprocessor.cropping)
            self.cb_baseline.setChecked(self.preprocessor.baseline_correction)
            self.cb_cosmic.setChecked(self.preprocessor.remove_cosmic_rays)
            self.cb_norm.setChecked(self.preprocessor.normalization)
            self.cb_smooth.setChecked(self.preprocessor.smoothing)
            
            self.start_raman_spin.setValue(self.preprocessor.start_raman_shift_cm)
            self.end_raman_spin.setValue(self.preprocessor.end_raman_shift_cm)
            self.baseline_lam_spin.setValue(self.preprocessor.baseline_lam)
            self.baseline_diff_order_spin.setValue(self.preprocessor.baseline_diff_order)
            self.baseline_max_iter_spin.setValue(self.preprocessor.baseline_max_iter)
            self.baseline_tol_spin.setValue(self.preprocessor.baseline_tol)
            self.cosmic_threshold_spin.setValue(self.preprocessor.cosmic_threshold)
            self.smooth_window_spin.setValue(self.preprocessor.smooth_window_length)
            self.smooth_polyorder_spin.setValue(self.preprocessor.smooth_polyorder)
            self.norm_type_combo.setCurrentText(self.preprocessor.normalization_type)
            
            print(f"Configuration loaded from {filename}")

    def save_config(self):
        """Save current preprocessing configuration to YAML or JSON file"""
        self.update_preprocessor_from_ui()
        
        config = self.preprocessor.to_dict()
        
        filename, selected_filter = QFileDialog.getSaveFileName(
            self, "Save Config", "", "YAML Files (*.yaml);;JSON Files (*.json)"
        )
        if filename:
            with open(filename, 'w') as f:
                if filename.endswith('.json'):
                    json.dump(config, f, indent=4)
                else:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            print(f"Configuration saved to {filename}")

    def update_preprocessor_from_ui(self):
        """Update preprocessor parameters from UI elements"""
        self.preprocessor.cropping = self.cb_cropping.isChecked()
        self.preprocessor.baseline_correction = self.cb_baseline.isChecked()
        self.preprocessor.remove_cosmic_rays = self.cb_cosmic.isChecked()
        self.preprocessor.normalization = self.cb_norm.isChecked()
        self.preprocessor.smoothing = self.cb_smooth.isChecked()
        
        self.preprocessor.start_raman_shift_cm = self.start_raman_spin.value()
        self.preprocessor.end_raman_shift_cm = self.end_raman_spin.value()
        self.preprocessor.baseline_lam = self.baseline_lam_spin.value()
        self.preprocessor.baseline_diff_order = self.baseline_diff_order_spin.value()
        self.preprocessor.baseline_max_iter = self.baseline_max_iter_spin.value()
        self.preprocessor.baseline_tol = self.baseline_tol_spin.value()
        self.preprocessor.cosmic_threshold = self.cosmic_threshold_spin.value()
        self.preprocessor.smooth_window_length = self.smooth_window_spin.value()
        self.preprocessor.smooth_polyorder = self.smooth_polyorder_spin.value()
        self.preprocessor.normalization_type = self.norm_type_combo.currentText()

    def populate_color_combo(self):
        self.color_combo.addItem("None")
        if len(self.data_objects) > 0:
            metadata_keys = set()
            for obj in self.data_objects:
                metadata_keys.update(obj.metadata.keys())
            for key in sorted(metadata_keys):
                self.color_combo.addItem(key)

    def update_dim(self, text):
        self.embedding_dim = 2 if text == "2D" else 3
        self.compute_embedding()
        self.plot_embedding()

    def update_method(self, text):
        self.embedding_method = text
        self.compute_embedding()
        self.plot_embedding()

    def compute_embedding(self):
        cache_key = (self.embedding_method, self.embedding_dim)
        if cache_key in self.embedding_cache:
            self.embedding_result = self.embedding_cache[cache_key]
            return
        
        # Check if we have data to process
        if len(self.data_objects) == 0:
            print("Warning: No data objects to compute embedding")
            self.embedding_result = np.array([])
            return
            
        data_matrix = np.array([obj.intensity for obj in self.data_objects])
        
        # Check if data matrix is valid
        if data_matrix.size == 0:
            print("Warning: Empty data matrix")
            self.embedding_result = np.array([])
            return
        
        # Ensure we don't request more components than available
        n_samples, n_features = data_matrix.shape
        n_components = min(self.embedding_dim, n_samples, n_features)
        
        try:
            if self.embedding_method == "T-SNE":
                # T-SNE requires at least n_components + 1 samples
                if n_samples < n_components + 1:
                    print(f"Warning: Not enough samples ({n_samples}) for T-SNE with {n_components} components")
                    # Fall back to PCA
                    reducer = PCA(n_components=n_components)
                    self.embedding_result = reducer.fit_transform(data_matrix)
                else:
                    reducer = TSNE(n_components=n_components, random_state=42)
                    self.embedding_result = reducer.fit_transform(data_matrix)
            elif self.embedding_method == "PCA":
                reducer = PCA(n_components=n_components)
                self.embedding_result = reducer.fit_transform(data_matrix)
            else:
                print(f"Unknown embedding method: {self.embedding_method}")
                self.embedding_result = np.array([])
                return
                
            self.embedding_cache[cache_key] = self.embedding_result
            print(f"Computed {self.embedding_method} embedding: {self.embedding_result.shape}")
            
        except Exception as e:
            print(f"Error computing embedding: {e}")
            import traceback
            traceback.print_exc()
            self.embedding_result = np.array([])

    def plot_embedding(self):
        self.ax_scatter.clear()
        
        # Check if we have a valid embedding result
        if self.embedding_result is None or len(self.embedding_result) == 0:
            self.ax_scatter.text(0.5, 0.5, 'No embedding data available.\nPlease check data preprocessing.', 
                               ha='center', va='center', transform=self.ax_scatter.transAxes)
            self.ax_scatter.set_title("Embedding Unavailable")
            self.canvas_scatter.draw_idle()
            return
        
        color_attr = self.color_combo.currentText()
        
        if color_attr == "outlier":
            colors = ['red' if i in self.outlier_indices else 'blue' for i in range(len(self.data_objects))]
            cmap_to_use = None
        elif color_attr != "None":
            values = [obj.metadata.get(color_attr, None) for obj in self.data_objects]
            unique_vals = sorted(set(v for v in values if v is not None))
            n_unique = len(unique_vals)
            
            # Use a colormap that can handle any number of categories
            # Generate distinct colors by spreading them across the colormap
            color_map = {val: idx / max(n_unique - 1, 1) for idx, val in enumerate(unique_vals)}
            colors = [color_map.get(v, -1) for v in values]
            
            # Use 'tab10' for up to 10 categories, then switch to continuous colormaps
            if n_unique <= 12:
                cmap_to_use = 'Paired'
            else:
                cmap_to_use = 'hsv'  # HSV colormap cycles through all hues
        else:
            colors = 'blue'
            cmap_to_use = None
        
        try:
            if self.embedding_dim == 2:
                self.scatter = self.ax_scatter.scatter(self.embedding_result[:, 0], self.embedding_result[:, 1],
                                                  c=colors, cmap=cmap_to_use,
                                                  picker=5)
                self.ax_scatter.set_xlabel(f"{self.embedding_method} 1")
                self.ax_scatter.set_ylabel(f"{self.embedding_method} 2")
            else:
                self.ax_scatter = self.fig_scatter.add_subplot(111, projection='3d')
                self.scatter = self.ax_scatter.scatter(self.embedding_result[:, 0], 
                                                  self.embedding_result[:, 1], 
                                                  self.embedding_result[:, 2],
                                                  c=colors, cmap=cmap_to_use,
                                                  picker=5)
                self.ax_scatter.set_xlabel(f"{self.embedding_method} 1")
                self.ax_scatter.set_ylabel(f"{self.embedding_method} 2")
                self.ax_scatter.set_zlabel(f"{self.embedding_method} 3")
            
            self.ax_scatter.set_title(f"{self.embedding_method} Embedding")
        except Exception as e:
            print(f"Error plotting embedding: {e}")
            import traceback
            traceback.print_exc()
            self.ax_scatter.text(0.5, 0.5, f'Error plotting embedding:\n{str(e)}', 
                               ha='center', va='center', transform=self.ax_scatter.transAxes)
        
        self.canvas_scatter.draw_idle()

    def toggle_lasso(self, checked):
        if checked:
            self.btn_select.setText("Disable Lasso Selection")
            self.lasso = LassoSelector(self.ax_scatter, self.on_lasso_select, useblit=True)
        else:
            self.btn_select.setText("Enable Lasso Selection")
            if self.lasso is not None:
                self.lasso.disconnect_events()
                self.lasso = None

    def on_lasso_select(self, verts):
        if self.embedding_dim != 2:
            return
        path = Path(verts)
        points = self.embedding_result[:, :2]
        selected = path.contains_points(points)
        self.selected_indices = [i for i, s in enumerate(selected) if s]
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
        """Update line plot - supports both grouped (by patient) and combined modes"""
        self.fig_line.clear()
        self.selected_index_to_line = {}
        self.metadata_list.clear()
        
        if len(self.selected_indices) == 0:
            ax = self.fig_line.add_subplot(111)
            ax.set_title("Select points to view spectra")
            ax.axis('off')
            self.canvas_line.draw_idle()
            return
        
        # Get colors from scatter plot
        scatter_colors = None
        if hasattr(self, 'scatter') and self.scatter is not None:
            scatter_colors = self.scatter.get_facecolors()
        
        if self.plot_mode == 'combined':
            # Combined mode: all spectra in one plot
            ax = self.fig_line.add_subplot(111)
            
            for idx in self.selected_indices:
                obj = self.data_objects[idx]
                
                # Get the color for this specific spectrum from the scatter plot
                if scatter_colors is not None and len(scatter_colors) > idx:
                    spectrum_color = scatter_colors[idx]
                else:
                    spectrum_color = None  # Will use default color
                
                line, = ax.plot(obj.raman_shift_cm, obj.intensity, 
                              color=spectrum_color, alpha=0.7, picker=5)
                self.selected_index_to_line[idx] = line
                
                # Add to metadata list
                metadata = obj.metadata
                patient_id = metadata.get('patient_id', 'Unknown')
                
                # Format metadata as a readable string with all fields
                metadata_parts = [f"Patient {patient_id}"]
                
                # Add all metadata fields except patient_id
                for key, value in sorted(metadata.items()):
                    if key != 'patient_id':
                        formatted_key = key.replace('_', ' ').title()
                        metadata_parts.append(f"{formatted_key}: {value}")
                
                metadata_str = " | ".join(metadata_parts)
                
                item = QListWidgetItem(metadata_str)
                item.setData(Qt.UserRole, idx)
                self.metadata_list.addItem(item)
            
            ax.set_title(f"All Selected Spectra ({len(self.selected_indices)} total)")
            ax.set_xlabel("Raman Shift (cm⁻¹)")
            ax.set_ylabel("Intensity")
            
        else:
            # Grouped mode: separate subplots by patient
            # Group selected spectra by patient_id
            patient_groups = {}
            for idx in self.selected_indices:
                obj = self.data_objects[idx]
                patient_id = obj.metadata.get('patient_id', 'Unknown')
                if patient_id not in patient_groups:
                    patient_groups[patient_id] = []
                patient_groups[patient_id].append(idx)
            
            # Create subplots - one per patient, stacked vertically
            n_patients = len(patient_groups)
            
            for plot_idx, (patient_id, indices) in enumerate(sorted(patient_groups.items())):
                ax = self.fig_line.add_subplot(n_patients, 1, plot_idx + 1)
                
                # Plot all spectra for this patient
                for idx in indices:
                    obj = self.data_objects[idx]
                    
                    # Get the color for this specific spectrum from the scatter plot
                    if scatter_colors is not None and len(scatter_colors) > idx:
                        spectrum_color = scatter_colors[idx]
                    else:
                        spectrum_color = None  # Will use default color
                    
                    line, = ax.plot(obj.raman_shift_cm, obj.intensity, 
                                  color=spectrum_color, alpha=0.7, picker=5)
                    self.selected_index_to_line[idx] = line
                    
                    # Add to metadata list - show ALL metadata fields
                    metadata = obj.metadata
                    
                    # Format metadata as a readable string with all fields
                    metadata_parts = [f"Patient {patient_id}"]
                    
                    # Add all metadata fields except patient_id (already shown)
                    for key, value in sorted(metadata.items()):
                        if key != 'patient_id':
                            # Format the key nicely (e.g., 'sample_type' -> 'Sample Type')
                            formatted_key = key.replace('_', ' ').title()
                            metadata_parts.append(f"{formatted_key}: {value}")
                    
                    metadata_str = " | ".join(metadata_parts)
                    
                    item = QListWidgetItem(metadata_str)
                    item.setData(Qt.UserRole, idx)
                    self.metadata_list.addItem(item)
                
                # Set subplot title and labels
                ax.set_title(f"Patient {patient_id} ({len(indices)} spectra)", fontsize=9, pad=3)
                ax.set_xlabel("Raman Shift (cm⁻¹)", fontsize=8)
                ax.set_ylabel("Intensity", fontsize=8)
                ax.tick_params(labelsize=7)
                
                # Share x-axis for all subplots except the last
                if plot_idx < n_patients - 1:
                    ax.set_xlabel('')
        
        # Tight layout to minimize white space
        self.fig_line.tight_layout(pad=0.5, h_pad=0.5)
        self.canvas_line.draw_idle()

    def clear_selection(self):
        self.selected_indices = []
        self.fig_line.clear()
        ax = self.fig_line.add_subplot(111)
        ax.set_title("Select points to view spectra")
        ax.axis('off')
        self.metadata_list.clear()
        self.selected_index_to_line = {}
        self.clear_scatter_highlight()
        self.canvas_line.draw_idle()

    def on_line_hover(self, event):
        for ax in self.fig_line.get_axes():
            if event.inaxes == ax:
                for idx, line in self.selected_index_to_line.items():
                    if line.axes == ax:
                        contains, _ = line.contains(event)
                        if contains:
                            if self.hovered_index != idx:
                                self.hovered_index = idx
                                self.update_line_highlights()
                            return
        if self.hovered_index is not None:
            self.hovered_index = None
            self.update_line_highlights()

    def on_line_pick(self, event):
        if event.artist in self.selected_index_to_line.values():
            idx = [k for k, v in self.selected_index_to_line.items() if v == event.artist][0]
            
            for i in range(self.metadata_list.count()):
                item = self.metadata_list.item(i)
                if item.data(Qt.UserRole) == idx:
                    was_selected = item.isSelected()
                    item.setSelected(not was_selected)
                    break

    def on_line_click(self, event):
        """Clear selection when clicking on blank space"""
        if event.inaxes is None:
            return
        
        clicked_on_line = False
        for idx, line in self.selected_index_to_line.items():
            if line.axes == event.inaxes:
                contains, _ = line.contains(event)
                if contains:
                    clicked_on_line = True
                    break
        
        if not clicked_on_line:
            self.metadata_list.clearSelection()
            self.clear_scatter_highlight()
            self.update_line_highlights()

    def update_line_highlights(self):
        scatter_colors = None
        if hasattr(self, 'scatter') and self.scatter is not None:
            scatter_colors = self.scatter.get_facecolors()
        
        selected_items = self.metadata_list.selectedItems()
        selected_indices_from_list = [item.data(Qt.UserRole) for item in selected_items]
        for idx, line in self.selected_index_to_line.items():
            if idx == self.hovered_index or idx in selected_indices_from_list:
                line.set_linewidth(2.5)
                line.set_color('gold')
                line.set_alpha(1.0)
            else:
                line.set_linewidth(1)
                if scatter_colors is not None and len(scatter_colors) > idx:
                    line.set_color(scatter_colors[idx])
                else:
                    line.set_color('C0')
                line.set_alpha(0.7)
        self.canvas_line.draw_idle()
        
        self.update_scatter_highlight(selected_indices_from_list)

    def update_scatter_highlight(self, selected_indices):
        """Highlight selected spectra in the scatter plot"""
        if not hasattr(self, 'scatter') or self.scatter is None:
            return
        
        sizes = np.full(len(self.data_objects), 20)
        for idx in selected_indices:
            sizes[idx] = 100
        
        self.scatter.set_sizes(sizes)
        self.canvas_scatter.draw_idle()

    def clear_scatter_highlight(self):
        """Clear scatter plot highlights"""
        if hasattr(self, 'scatter') and self.scatter is not None:
            sizes = np.full(len(self.data_objects), 20)
            self.scatter.set_sizes(sizes)
            self.canvas_scatter.draw_idle()

    def save_metadata(self):
        metadata_list = [self.data_objects[idx].metadata for idx in self.selected_indices]
        filename, _ = QFileDialog.getSaveFileName(self, "Save Metadata", "", "JSON Files (*.json)")
        if filename:
            with open(filename, "w") as f:
                json.dump(metadata_list, f, indent=4)

    def update_preprocessing(self):
        """Re-process data with current preprocessing parameters"""
        self.update_preprocessor_from_ui()
        
        # Re-preprocess all data
        self.data_objects = [self.preprocessor.preprocess(copy.deepcopy(obj)) for obj in self.original_data_objects]
        
        # Align spectra to common grid
        self.align_spectra_to_common_grid()
        
        # Clear cache and recompute embedding
        self.embedding_cache.clear()
        self.compute_embedding()
        self.plot_embedding()
        
        # Update line plot if there are selected spectra
        if self.selected_indices:
            self.update_line_plot()
        
        print("Data re-processed with new parameters")

    def optimize_dbscan_params(self):
        """Optimize DBSCAN parameters using silhouette score"""
        data_matrix = np.array([obj.intensity for obj in self.data_objects])
        pca_data = PCA(n_components=min(10, data_matrix.shape[1])).fit_transform(data_matrix)
        
        eps_min = self.opt_eps_min_spin.value()
        eps_max = self.opt_eps_max_spin.value()
        min_samples_min = self.opt_min_min_spin.value()
        min_samples_max = self.opt_min_max_spin.value()
        
        eps_range = np.linspace(eps_min, eps_max, 10)
        min_samples_range = range(min_samples_min, min_samples_max + 1, 2)
        
        best_score = -1
        best_eps = self.dbscan_eps
        best_min_samples = self.dbscan_min
        
        for eps in eps_range:
            for min_samples in min_samples_range:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(pca_data)
                
                if len(set(labels)) > 1 and -1 not in labels:
                    score = silhouette_score(pca_data, labels)
                    if score > best_score:
                        best_score = score
                        best_eps = eps
                        best_min_samples = min_samples
        
        self.dbscan_eps = best_eps
        self.dbscan_min = best_min_samples
        self.dbscan_params_label.setText(f"eps: {self.dbscan_eps:.2f}, min_samples: {self.dbscan_min}")
        print(f"Optimized DBSCAN: eps={best_eps:.2f}, min_samples={best_min_samples}, silhouette={best_score:.3f}")

    def find_outliers(self):
        """Find outliers using iterative PCA + DBSCAN"""
        self.n_iterations = self.n_iter_spin.value()
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
            dbscan = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min)
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
        
        self.btn_select_outliers.setEnabled(True)

    def select_outliers(self):
        """Select all outliers"""
        if self.outlier_indices:
            self.selected_indices = list(self.outlier_indices)
            self.update_line_plot()

if __name__ == "__main__":
    # Example usage - update these paths to your actual data
    data_folder = r"C:\Users\Yifei\Box\Carney Lab Shared\Data\Raman_Robot\2025_11_18"
    
    # Create default config
    default_config = {
        'preprocessing': {
            'cropping': {
                'start_raman_shift_cm': 662.697,
                'end_raman_shift_cm': 1784.104
            },
            'baseline_correction': {
                'lam': 1000,
                'diff_order': 1,
                'max_iter': 15,
                'tol': 0.005
            },
            'cosmic_rays_removal': {
                'threshold': 10
            },
            'smoothing': {
                'window_length': 9,
                'polyorder': 2
            },
            'normalization_type': 'by_max'
        },
        'enabled': {
            'cropping': True,
            'baseline_correction': False,
            'remove_cosmic_rays': True,
            'normalization': False,
            'smoothing': False
        }
    }

    preprocessor = SpectrumPreprocessorWrapper.from_dict(default_config)
    dataset = OC_Dataset(data_folder, preprocessor, augmentor=None)

    app = QApplication(sys.argv)
    window = SpectraViewer(dataset.db, preprocessor)
    window.resize(1400, 800)
    window.show()
    sys.exit(app.exec_())