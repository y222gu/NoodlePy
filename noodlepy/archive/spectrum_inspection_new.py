import sys
import json
import yaml
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QHBoxLayout, QPushButton, QFileDialog, QComboBox, 
    QLabel, QGroupBox, QCheckBox, QDoubleSpinBox, QSpinBox, 
    QRadioButton, QButtonGroup, QScrollArea, QFormLayout, 
    QTableWidgetItem, QGridLayout, QSizePolicy, QStackedLayout,
    QTableWidget, QAbstractItemView, QHeaderView  # <-- add these
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.path import Path
from sklearn.manifold import TSNE
from combat.pycombat import pycombat
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import copy
from matplotlib.widgets import LassoSelector
# Import the actual Spectrum and SpectrumPreprocessor classes
from noodlepy.utils.spectrum import Spectrum
from noodlepy.utils.spectrumpreprocessor import SpectrumPreprocessor
from noodlepy.utils.raman_robot_dataset import RamanRobotDataset
from matplotlib import colors, cm  # add this
import umap
from PyQt5.QtGui import QPalette, QColor

DEFAULT_BLUE = "#496fa7"

def apply_dark_theme(app: QApplication):
    app.setStyle("Fusion")

    dark = QPalette()
    dark.setColor(QPalette.Window, QColor(30, 30, 30))
    dark.setColor(QPalette.WindowText, QColor(220, 220, 220))
    dark.setColor(QPalette.Base, QColor(25, 25, 25))
    dark.setColor(QPalette.AlternateBase, QColor(35, 35, 35))
    dark.setColor(QPalette.ToolTipBase, QColor(220, 220, 220))
    dark.setColor(QPalette.ToolTipText, QColor(220, 220, 220))
    dark.setColor(QPalette.Text, QColor(220, 220, 220))
    dark.setColor(QPalette.Button, QColor(45, 45, 45))
    dark.setColor(QPalette.ButtonText, QColor(220, 220, 220))
    dark.setColor(QPalette.BrightText, QColor(255, 0, 0))
    dark.setColor(QPalette.Highlight, QColor(80, 120, 200))
    dark.setColor(QPalette.HighlightedText, QColor(0, 0, 0))

    app.setPalette(dark)

# # Use the "fast" style.
# plt.style.use('fast')

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

class SpectraViewer(QMainWindow):
    def __init__(self, data_objects, preprocessor, config_file=None):
        super().__init__()
        self.setWindowTitle("Spectra Viewer 2.1")
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
            sample_obj = self.data_objects[0]
            print(f"Sample spectrum shape: intensity={len(sample_obj.intensity)}, raman={len(sample_obj.raman_shift_cm)}")
            if len(sample_obj.intensity) > 0:
                print(f"Raman shift range: [{sample_obj.raman_shift_cm.min():.2f}, {sample_obj.raman_shift_cm.max():.2f}]")

        self.selected_indices = []
        self.selected_index_to_line = {}
        self.hovered_index = None
        self.scatter_highlight = None
        self.top_index = None   # spectrum index to be drawn on top
        self.scatter_selected = None   # overlay scatter for selected points (on top)
        self.scatter_cbar = None       # colorbar for scatter

        self.embedding_method = "PCA"
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
        self.plot_mode = 'combined'

        # ---- Filtering state (multi-filter support) ----
        self.filters = []  # each item: {'widget', 'attr_combo', 'value_combo', 'attr', 'value'}
        self.visible_indices = list(range(len(self.data_objects))) if self.data_objects else []
        self.index_to_scatter_pos = {}

        # --- PCA state for loadings plot ---
        self.pca_model = None
        self.pca_loadings = None          # shape: (n_components, n_features)
        self.pca_explained_var = None     # explained_variance_ratio_
        self.n_pcs_for_loading = 5

        self.initUI()
        self.populate_color_combo()
        self.add_filter_row()  # create the first filter row
        self.compute_embedding()
        self.plot_embedding()
        self.btn_select.setChecked(True)
        self.update_line_plot()
        self.update_pca_loadings_plot()  # optional; safe


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
        from PyQt5.QtWidgets import (
            QWidget, QHBoxLayout, QVBoxLayout, QGroupBox, QScrollArea, QTabWidget,
            QSplitter, QSizePolicy, QFrame
        )

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        root_layout = QHBoxLayout(central_widget)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(10)

        # ============================================================
        # MAIN HORIZONTAL SPLITTER: [LEFT] | [RIGHT]
        # ============================================================
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setChildrenCollapsible(False)
        root_layout.addWidget(main_splitter)

        # =========================
        # LEFT AREA: Tabs (top) + Scatter (mid) + Selection (bottom)
        # =========================
        left_area = QWidget()
        left_layout = QVBoxLayout(left_area)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        # --- Tabs container ---
        left_tabs = QTabWidget()
        left_tabs.setDocumentMode(True)

        # ---------------------------------------------------------------------
        # TAB 1: PREPROCESSING (includes Filters)
        # ---------------------------------------------------------------------
        preproc_tab = QWidget()
        preproc_tab_layout = QVBoxLayout(preproc_tab)
        preproc_tab_layout.setContentsMargins(0, 0, 0, 0)
        preproc_tab_layout.setSpacing(10)

        preproc_group = QGroupBox("Preprocessing Options")
        preproc_group_layout = QVBoxLayout(preproc_group)
        preproc_group_layout.setSpacing(8)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)

        scroll_widget = QWidget()
        preproc_layout = QVBoxLayout(scroll_widget)
        preproc_layout.setSpacing(10)

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
        crop_params_layout.addRow("Start (cm⁻¹):", self.start_raman_spin)

        self.end_raman_spin = QDoubleSpinBox()
        self.end_raman_spin.setRange(0.0, 5000.0)
        self.end_raman_spin.setSingleStep(0.1)
        self.end_raman_spin.setDecimals(3)
        self.end_raman_spin.setValue(self.preprocessor.end_raman_shift_cm)
        crop_params_layout.addRow("End (cm⁻¹):", self.end_raman_spin)

        preproc_layout.addWidget(self.crop_params_widget)

        # ==== BASELINE ====
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
        baseline_params_layout.addRow("Max Iter:", self.baseline_max_iter_spin)

        self.baseline_tol_spin = QDoubleSpinBox()
        self.baseline_tol_spin.setRange(0.0001, 0.1)
        self.baseline_tol_spin.setSingleStep(0.001)
        self.baseline_tol_spin.setDecimals(4)
        self.baseline_tol_spin.setValue(self.preprocessor.baseline_tol)
        baseline_params_layout.addRow("Tolerance:", self.baseline_tol_spin)

        preproc_layout.addWidget(self.baseline_params_widget)

        # ==== COSMIC ====
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

        # ==== SMOOTHING ====
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
        smooth_params_layout.addRow("Window:", self.smooth_window_spin)

        self.smooth_polyorder_spin = QSpinBox()
        self.smooth_polyorder_spin.setRange(1, 5)
        self.smooth_polyorder_spin.setValue(self.preprocessor.smooth_polyorder)
        smooth_params_layout.addRow("Poly:", self.smooth_polyorder_spin)

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
        self.norm_type_combo.addItems(["by_max", "by_area"])
        self.norm_type_combo.setCurrentText(self.preprocessor.normalization_type)
        norm_params_layout.addRow("Type:", self.norm_type_combo)

        preproc_layout.addWidget(self.norm_params_widget)

        # ==== BATCH ====
        self.cb_batch = QCheckBox("Apply Batch Correction")
        self.cb_batch.setChecked(False)
        preproc_layout.addWidget(self.cb_batch)

        self.rbg_batch = QButtonGroup(self)
        self.rb_combat = QRadioButton("ComBat")
        self.rb_combat.setChecked(True)
        self.rbg_batch.addButton(self.rb_combat)

        batch_radio_column_layout = QVBoxLayout()
        batch_radio_column_layout.setContentsMargins(20, 0, 0, 0)
        batch_radio_column_layout.addWidget(self.rb_combat)
        preproc_layout.addLayout(batch_radio_column_layout)

        scroll_area.setWidget(scroll_widget)
        preproc_group_layout.addWidget(scroll_area)

        config_buttons_layout = QHBoxLayout()
        self.btn_load_config = QPushButton("Load Config")
        self.btn_load_config.clicked.connect(self.load_config)
        config_buttons_layout.addWidget(self.btn_load_config)

        self.btn_save_config = QPushButton("Save Config")
        self.btn_save_config.clicked.connect(self.save_config)
        config_buttons_layout.addWidget(self.btn_save_config)

        preproc_group_layout.addLayout(config_buttons_layout)
        preproc_tab_layout.addWidget(preproc_group)

        # ---- Filters in Preprocessing tab ----
        filter_group = QGroupBox("Filters")
        filter_group_layout = QVBoxLayout(filter_group)
        filter_group_layout.setSpacing(6)

        header = QHBoxLayout()
        header.addWidget(QLabel("Filter by type:"))
        self.btn_add_filter = QPushButton("➕")
        self.btn_add_filter.setFixedSize(24, 24)
        self.btn_add_filter.setStyleSheet("""
            QPushButton { font-size: 16px; background-color: #e8e8e8; border: none; outline: none; }
            QPushButton:hover { background-color: #d0d0d0; border: none; outline: none; }
        """)
        self.btn_add_filter.clicked.connect(self.add_filter_row)
        header.addWidget(self.btn_add_filter)
        header.addStretch(1)
        filter_group_layout.addLayout(header)

        self.filters_container_layout = QVBoxLayout()
        self.filters_container_layout.setSpacing(6)
        filter_group_layout.addLayout(self.filters_container_layout)

        self.btn_recalculate = QPushButton("Re-process")
        self.btn_recalculate.clicked.connect(self.update_preprocessing)
        preproc_group_layout.addWidget(self.btn_recalculate)

        preproc_tab_layout.addWidget(filter_group)
        left_tabs.addTab(preproc_tab, "Preprocessing")

        # ---------------------------------------------------------------------
        # TAB 2: EMBEDDING
        # ---------------------------------------------------------------------
        embed_tab = QWidget()
        embed_tab_layout = QVBoxLayout(embed_tab)
        embed_tab_layout.setContentsMargins(0, 0, 0, 0)
        embed_tab_layout.setSpacing(10)

        embedding_group = QGroupBox("Dim Reduction Options")
        embedding_layout = QVBoxLayout(embedding_group)
        embedding_layout.setSpacing(8)

        self.combo_dim = QComboBox()
        self.combo_dim.addItems(["2D", "3D"])
        self.combo_dim.currentTextChanged.connect(self.update_dim)
        embedding_layout.addWidget(QLabel("Dimensionality:"))
        embedding_layout.addWidget(self.combo_dim)

        self.combo_method = QComboBox()
        self.combo_method.addItems(["PCA", "T-SNE", "UMAP"])
        self.combo_method.currentTextChanged.connect(self.update_method)
        embedding_layout.addWidget(QLabel("Method:"))
        embedding_layout.addWidget(self.combo_method)

        self.spin_pca_loadings = QSpinBox()
        self.spin_pca_loadings.setRange(1, 20)
        self.spin_pca_loadings.setValue(self.n_pcs_for_loading)
        self.spin_pca_loadings.setKeyboardTracking(False)
        self.spin_pca_loadings.lineEdit().returnPressed.connect(
            lambda: self.on_pca_loading_count_changed(self.spin_pca_loadings.value())
        )
        embedding_layout.addWidget(QLabel("PCA Loadings PCs:"))
        embedding_layout.addWidget(self.spin_pca_loadings)
        embed_tab_layout.addWidget(embedding_group)

        embed_tab_layout.addStretch(1)
        left_tabs.addTab(embed_tab, "Dim Reduction")

        # ---------------------------------------------------------------------
        # TAB 3: Color Mapping
        # ---------------------------------------------------------------------
        embed_tab = QWidget()
        embed_tab_layout = QVBoxLayout(embed_tab)
        embed_tab_layout.setContentsMargins(0, 0, 0, 0)
        embed_tab_layout.setSpacing(10)

        color_group = QGroupBox("Color Mapping")
        color_layout = QVBoxLayout(color_group)
        color_layout.setSpacing(8)

        self.color_combo = QComboBox()
        self.color_combo.currentTextChanged.connect(self.on_color_option_changed)
        color_layout.addWidget(QLabel("Color By:"))
        color_layout.addWidget(self.color_combo)

        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(["tab10", "turbo", "hsv", "cool"])
        self.cmap_combo.setCurrentText("tab10")
        self.cmap_combo.currentTextChanged.connect(self.on_color_option_changed)
        color_layout.addWidget(QLabel("Color Map:"))
        color_layout.addWidget(self.cmap_combo)

        embed_tab_layout.addWidget(color_group)
        embed_tab_layout.addStretch(1)
        left_tabs.addTab(embed_tab, "Color Mapping")

        # ---------------------------------------------------------------------
        # TAB 4: DBSCAN
        # ---------------------------------------------------------------------
        dbscan_tab = QWidget()
        dbscan_tab_layout = QVBoxLayout(dbscan_tab)
        dbscan_tab_layout.setContentsMargins(0, 0, 0, 0)
        dbscan_tab_layout.setSpacing(10)

        dbscan_group = QGroupBox("DBSCAN Parameters")
        dbscan_layout = QVBoxLayout(dbscan_group)
        dbscan_layout.setSpacing(8)

        eps_row = QHBoxLayout()
        eps_row.addWidget(QLabel("Optimize eps range:"))
        self.opt_eps_min_spin = QDoubleSpinBox()
        self.opt_eps_min_spin.setRange(0.0, 10.0)
        self.opt_eps_min_spin.setSingleStep(0.1)
        self.opt_eps_min_spin.setValue(0.1)
        self.opt_eps_max_spin = QDoubleSpinBox()
        self.opt_eps_max_spin.setRange(0.0, 10.0)
        self.opt_eps_max_spin.setSingleStep(0.1)
        self.opt_eps_max_spin.setValue(2.0)
        eps_row.addWidget(self.opt_eps_min_spin)
        eps_row.addWidget(self.opt_eps_max_spin)
        dbscan_layout.addLayout(eps_row)

        min_row = QHBoxLayout()
        min_row.addWidget(QLabel("Optimize min_samples:"))
        self.opt_min_min_spin = QSpinBox()
        self.opt_min_min_spin.setRange(2, 50)
        self.opt_min_min_spin.setValue(2)
        self.opt_min_max_spin = QSpinBox()
        self.opt_min_max_spin.setRange(2, 50)
        self.opt_min_max_spin.setValue(20)
        min_row.addWidget(self.opt_min_min_spin)
        min_row.addWidget(self.opt_min_max_spin)
        dbscan_layout.addLayout(min_row)

        self.btn_optimize_dbscan = QPushButton("Optimize DBSCAN")
        self.btn_optimize_dbscan.clicked.connect(self.optimize_dbscan_params)
        dbscan_layout.addWidget(self.btn_optimize_dbscan)

        self.dbscan_params_label = QLabel(f"eps: {self.dbscan_eps}, min_samples: {self.dbscan_min}")
        dbscan_layout.addWidget(self.dbscan_params_label)

        it_row = QHBoxLayout()
        it_row.addWidget(QLabel("Outlier Detection Iterations:"))
        self.n_iter_spin = QSpinBox()
        self.n_iter_spin.setRange(1, 10)
        self.n_iter_spin.setValue(self.n_iterations)
        it_row.addWidget(self.n_iter_spin)
        dbscan_layout.addLayout(it_row)

        self.btn_find_outliers = QPushButton("Find Outliers")
        self.btn_find_outliers.clicked.connect(self.find_outliers)
        dbscan_layout.addWidget(self.btn_find_outliers)

        self.btn_select_outliers = QPushButton("Select All Outliers")
        self.btn_select_outliers.setEnabled(False)
        self.btn_select_outliers.clicked.connect(self.select_outliers)
        dbscan_layout.addWidget(self.btn_select_outliers)

        dbscan_tab_layout.addWidget(dbscan_group)
        dbscan_tab_layout.addStretch(1)
        left_tabs.addTab(dbscan_tab, "DBSCAN")

        # Add tabs to left area
        left_layout.addWidget(left_tabs, stretch=0)

        # Init preprocessing param visibility
        self.toggle_cropping_params()
        self.toggle_baseline_params()
        self.toggle_cosmic_params()
        self.toggle_norm_params()
        self.toggle_smooth_params()

        # --- Scatter plot (left column) ---
        scatter_group = QGroupBox("Dim Reduced Scatter")
        scatter_layout = QVBoxLayout(scatter_group)
        scatter_layout.setContentsMargins(8, 8, 8, 8)

        self.fig_scatter = Figure(figsize=(5, 5))
        self.ax_scatter = self.fig_scatter.add_subplot(111)
        self.canvas_scatter = FigureCanvas(self.fig_scatter)
        self._make_mpl_transparent(self.fig_scatter, self.canvas_scatter)

        self.canvas_scatter.mpl_connect('scroll_event', self.on_scroll)
        self.canvas_scatter.mpl_connect('button_press_event', self.on_pan_press)
        self.canvas_scatter.mpl_connect('motion_notify_event', self.on_pan_motion)
        self.canvas_scatter.mpl_connect('button_release_event', self.on_pan_release)
        self._pending_line_color_update = False
        self.canvas_scatter.mpl_connect('draw_event', self._on_scatter_draw)

        self.canvas_scatter.setMinimumSize(520, 520)
        self.canvas_scatter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        scatter_layout.addWidget(self.canvas_scatter)
        left_layout.addWidget(scatter_group, stretch=1)

        # Clear + Lasso buttons on one row (saves vertical space)
        sel_btn_row = QHBoxLayout()

        # Toggle button for lasso selection
        self.btn_select = QPushButton("Enable Lasso Selection")
        self.btn_select.setCheckable(True)
        self.btn_select.toggled.connect(self.toggle_lasso)
        self.toggle_lasso(True)  # Initialize lasso selection as enabled

        btn_clear = QPushButton("Clear Selection")
        btn_clear.clicked.connect(self.clear_selection)
        sel_btn_row.addWidget(self.btn_select)
        sel_btn_row.addWidget(btn_clear)
        left_layout.addLayout(sel_btn_row)

        # =========================
        # RIGHT AREA: Spectra (top) + Loadings (mid) + Metadata (bottom)
        # =========================
        right_area = QWidget()
        right_layout = QVBoxLayout(right_area)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        # ---- Spectrum plot (top) ----
        spectrum_group = QGroupBox("Spectra")
        spectrum_layout = QVBoxLayout(spectrum_group)
        spectrum_layout.setContentsMargins(8, 8, 8, 8)

        self.fig_line = Figure(figsize=(12, 4))
        self.canvas_line = FigureCanvas(self.fig_line)
        self._make_mpl_transparent(self.fig_line, self.canvas_line)
        self.canvas_line.mpl_connect('motion_notify_event', self.on_line_hover)
        self.canvas_line.mpl_connect('pick_event', self.on_line_pick)
        self.canvas_line.mpl_connect('button_press_event', self.on_line_click)

        self.canvas_line.setMinimumHeight(420)
        self.canvas_line.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # --- Plot mode toggle above spectra plot ---
        plot_mode_row = QHBoxLayout()
        plot_mode_row.addStretch(1)

        self.btn_toggle_plot_mode = QPushButton("Combined View")
        self.btn_toggle_plot_mode.setCheckable(True)
        self.btn_toggle_plot_mode.setChecked(False)
        self.btn_toggle_plot_mode.toggled.connect(self.toggle_plot_mode)
        self.btn_toggle_plot_mode.setMaximumWidth(160)

        plot_mode_row.addWidget(self.btn_toggle_plot_mode)
        spectrum_layout.addLayout(plot_mode_row)
        spectrum_layout.addWidget(self.canvas_line)
        right_layout.addWidget(spectrum_group, stretch=2)

        # ---- Loadings plot (middle) ----
        load_group = QGroupBox("PCA Loadings")
        load_layout = QVBoxLayout(load_group)
        load_layout.setContentsMargins(8, 8, 8, 8)

        self.fig_load = Figure(figsize=(12, 2))
        self.ax_load = self.fig_load.add_subplot(111)
        self.canvas_load = FigureCanvas(self.fig_load)
        self._make_mpl_transparent(self.fig_load, self.canvas_load)

        # Make sure initial load plot styling is white on dark background
        try:
            self.fig_load.patch.set_alpha(0.0)
        except Exception:
            pass
        self.ax_load.set_facecolor("none")
        self.ax_load.set_title("PCA loadings", color="white")
        self.ax_load.set_xlabel("Raman Shift (cm⁻¹)", color="white")
        self.ax_load.set_ylabel("Loading", color="white")
        self.ax_load.tick_params(colors="white")
        for spine in self.ax_load.spines.values():
            spine.set_color("white")
        self.ax_load.grid(color="white", alpha=0.15)
        self.canvas_load.draw_idle()

        self.canvas_load.setMinimumHeight(200)
        self.canvas_load.setMaximumHeight(260)
        self.canvas_load.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        load_layout.addWidget(self.canvas_load)
        right_layout.addWidget(load_group, stretch=0)

        # ---- Metadata table (bottom) ----
        metadata_group = QGroupBox("Selected Spectra Metadata")
        metadata_layout = QVBoxLayout(metadata_group)
        metadata_layout.setSpacing(6)

        self.metadata_table = QTableWidget()
        self.metadata_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.metadata_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.metadata_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.metadata_table.verticalHeader().setVisible(False)
        self.metadata_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.metadata_table.itemSelectionChanged.connect(self.update_line_highlights)
        self.metadata_table.horizontalHeader().sectionClicked.connect(self.on_metadata_header_clicked)

        metadata_layout.addWidget(self.metadata_table)

        btn_save = QPushButton("Save Metadata")
        btn_save.clicked.connect(self.save_metadata)
        metadata_layout.addWidget(btn_save)

        right_layout.addWidget(metadata_group, stretch=1)

        # =========================
        # Add to splitter
        # =========================
        main_splitter.addWidget(left_area)
        main_splitter.addWidget(right_area)
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setSizes([520, 1200])


    def get_metadata_keys(self):
        """Return sorted list of metadata keys across all spectra."""
        metadata_keys = set()
        for obj in self.data_objects:
            metadata_keys.update(obj.metadata.keys())
        return sorted(metadata_keys)
    
    def add_filter_row(self):
        """Add one filter row: [attr_combo] [value_combo] [remove button]."""
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)

        attr_combo = QComboBox()
        value_combo = ScrollableComboBox()  # <-- changed here
        value_combo.addItem("All")
        value_combo.setEnabled(False)

        btn_remove = QPushButton("➖")
        btn_remove.setFixedSize(24, 24)
        btn_remove.setStyleSheet("""
            QPushButton {
            font-size: 16px;
            background-color: #e8e8e8;
            border: none;
            outline: none;
            }
            QPushButton:hover {
            background-color: #d0d0d0;
            border: none;
            outline: none;
            }
        """)

        row_layout.addWidget(attr_combo)
        row_layout.addWidget(value_combo)
        row_layout.addWidget(btn_remove)

        self.filters_container_layout.addWidget(row_widget)

        filter_obj = {
            'widget': row_widget,
            'attr_combo': attr_combo,
            'value_combo': value_combo,
            'attr': None,
            'value': "All",
        }
        self.filters.append(filter_obj)

        # Populate attribute combo
        self.populate_filter_attr_combo(attr_combo)

        # Connect signals (capture filter_obj via default arg)
        attr_combo.currentTextChanged.connect(
            lambda text, f=filter_obj: self.on_filter_attr_changed(f, text)
        )
        value_combo.currentTextChanged.connect(
            lambda text, f=filter_obj: self.on_filter_value_changed(f, text)
        )
        btn_remove.clicked.connect(
            lambda _, f=filter_obj: self.remove_filter_row(f)
        )

        # On creation, apply filters (no-op until user changes attr)
        self.update_visible_indices()

    def remove_filter_row(self, filter_obj):
        """Remove a filter row and reapply filters."""
        if filter_obj in self.filters:
            self.filters.remove(filter_obj)
        widget = filter_obj.get('widget')
        if widget is not None:
            widget.setParent(None)
        self.update_visible_indices()

    def on_color_option_changed(self, *_):
        """
        Called when color-by metadata or colormap changes.
        We replot the embedding and then wait for the canvas 'draw_event'
        to refresh line colors so they exactly match the scatter.
        """
        # Mark that, after the scatter is drawn, we want to refresh line colors
        self._pending_line_color_update = True

        # This will recreate the scatter with the new colors and call
        # canvas_scatter.draw_idle(), which eventually fires 'draw_event'
        self.plot_embedding()

    def toggle_lasso(self, checked):
        """Enable/disable Lasso selection on the embedding scatter plot."""
        # Only meaningful in 2D
        if self.embedding_dim != 2:
            if self.lasso is not None:
                try:
                    self.lasso.disconnect_events()
                except Exception:
                    pass
                self.lasso = None
            self.btn_select.setChecked(False)
            self.btn_select.setText("Enable Lasso Selection")
            return

        if checked:
            self.btn_select.setText("Disable Lasso Selection")
            # Remove old lasso if it exists
            if self.lasso is not None:
                try:
                    self.lasso.disconnect_events()
                except Exception:
                    pass
                self.lasso = None

            self.lasso = LassoSelector(
                self.ax_scatter,
                self.on_lasso_select,
                useblit=True,
                props=dict(color="white", linewidth=1.5, alpha=0.95)
            )
        else:
            self.btn_select.setText("Enable Lasso Selection")
            if self.lasso is not None:
                try:
                    self.lasso.disconnect_events()
                except Exception:
                    pass
                self.lasso = None

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
        self.color_combo.blockSignals(True)
        self.color_combo.clear()
        self.color_combo.addItem("None")
        if len(self.data_objects) > 0:
            metadata_keys = set()
            for obj in self.data_objects:
                metadata_keys.update(obj.metadata.keys())
            for key in sorted(metadata_keys):
                self.color_combo.addItem(key)
        self.color_combo.blockSignals(False)

    def populate_filter_attr_combo(self, combo):
        """Fill a given attribute combo with metadata keys."""
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("None")
        for key in self.get_metadata_keys():
            combo.addItem(key)
        combo.blockSignals(False)

    def get_visible_indices(self):
        """Return indices that should currently be shown in the scatter plot."""
        if self.visible_indices is None:
            return list(range(len(self.data_objects)))
        return self.visible_indices

    def update_dim(self, text):
        self.embedding_dim = 2 if text == "2D" else 3
        self.compute_embedding(indices=self.visible_indices, force=True)
        self.plot_embedding()
        self.update_line_plot()
        self.update_pca_loadings_plot()

    def update_method(self, text):
        self.embedding_method = text
        self.compute_embedding(indices=self.visible_indices, force=True)
        self.plot_embedding()
        self.update_line_plot()
        self.update_line_highlights()
        self.update_pca_loadings_plot()

    def compute_embedding(self, indices=None, force=False):
        """
        Compute embedding for a subset of points (indices). If indices is None, use all points.
        Stores result in self.embedding_result as a full-size array (N_total x emb_dim) with NaNs
        for points not included in the fit, so the rest of the GUI can still index by global idx.
        """
        import hashlib

        # ----- choose fit indices -----
        N_total = len(self.data_objects)
        if N_total == 0:
            self.embedding_result = np.array([])
            return

        if indices is None:
            fit_indices = list(range(N_total))
        else:
            fit_indices = list(indices)

        if len(fit_indices) == 0:
            # nothing to fit; keep empty embedding
            self.embedding_result = np.full((N_total, self.embedding_dim), np.nan, dtype=float)
            # also clear PCA state if not PCA
            if self.embedding_method != "PCA":
                self.pca_model = None
                self.pca_loadings = None
                self.pca_explained_var = None
            return

        # ----- build cache key that depends on the subset -----
        idx_bytes = np.asarray(fit_indices, dtype=np.int32).tobytes()
        idx_hash = hashlib.md5(idx_bytes).hexdigest()  # stable + short
        cache_key = (
            self.embedding_method,
            self.embedding_dim,
            int(getattr(self, "n_pcs_for_loading", 0)) if self.embedding_method == "PCA" else None,
            idx_hash,
        )

        if (not force) and cache_key in self.embedding_cache:
            cached = self.embedding_cache[cache_key]
            if self.embedding_method == "PCA":
                self.embedding_result, self.pca_model, self.pca_loadings, self.pca_explained_var = cached
            else:
                self.embedding_result = cached
            return

        # ----- make matrix for ONLY the visible indices -----
        data_matrix = np.array([self.data_objects[i].intensity for i in fit_indices])
        if data_matrix.size == 0:
            self.embedding_result = np.full((N_total, self.embedding_dim), np.nan, dtype=float)
            return

        n_samples, n_features = data_matrix.shape
        emb_components = min(self.embedding_dim, n_samples, n_features)

        # PCA can fit more components than scatter dim for loadings
        pca_fit_components = min(
            max(emb_components, int(getattr(self, "n_pcs_for_loading", 3))),
            n_samples,
            n_features
        )

        # Clear PCA state when leaving PCA
        if self.embedding_method != "PCA":
            self.pca_model = None
            self.pca_loadings = None
            self.pca_explained_var = None

        try:
            if self.embedding_method == "T-SNE":
                # Barnes-Hut TSNE only supports n_components <= 3
                emb_components = min(emb_components, 3)

                if n_samples < emb_components + 1:
                    reducer = PCA(n_components=emb_components)
                    sub_emb = reducer.fit_transform(data_matrix)
                else:
                    perplexity = min(30, (n_samples - 1) / 3)
                    reducer = TSNE(
                        n_components=emb_components,
                        perplexity=perplexity,
                        random_state=42
                    )
                    sub_emb = reducer.fit_transform(data_matrix)

            elif self.embedding_method == "UMAP":
                reducer = umap.UMAP(n_components=emb_components, random_state=42)
                sub_emb = reducer.fit_transform(data_matrix)

            elif self.embedding_method == "PCA":
                reducer = PCA(n_components=pca_fit_components)
                full_scores = reducer.fit_transform(data_matrix)

                # Scatter uses first emb_components
                sub_emb = full_scores[:, :emb_components]

                self.pca_model = reducer
                self.pca_loadings = reducer.components_
                self.pca_explained_var = reducer.explained_variance_ratio_

            else:
                print(f"Unknown embedding method: {self.embedding_method}")
                self.embedding_result = np.full((N_total, self.embedding_dim), np.nan, dtype=float)
                return

            # ----- write into full-size embedding array -----
            full = np.full((N_total, emb_components), np.nan, dtype=float)
            full[np.asarray(fit_indices, dtype=int), :] = sub_emb
            self.embedding_result = full

            # cache (store PCA state too)
            if self.embedding_method == "PCA":
                self.embedding_cache[cache_key] = (
                    self.embedding_result, self.pca_model, self.pca_loadings, self.pca_explained_var
                )
            else:
                self.embedding_cache[cache_key] = self.embedding_result

            print(f"Computed {self.embedding_method} embedding on {len(fit_indices)}/{N_total} points: {sub_emb.shape}")

        except Exception as e:
            print(f"Error computing embedding: {e}")
            import traceback
            traceback.print_exc()
            self.embedding_result = np.full((N_total, self.embedding_dim), np.nan, dtype=float)


    def plot_embedding(self):
        """Plot embedding, keeping axis limits fixed and managing colorbar/selection/lasso."""
        # Clear the entire figure so old axes/colorbars are removed
        self.fig_scatter.clear()

        # Recreate axes depending on 2D/3D
        if self.embedding_dim == 2:
            self.ax_scatter = self.fig_scatter.add_subplot(111)
        else:
            self.ax_scatter = self.fig_scatter.add_subplot(111, projection='3d')
            # after ax is created as projection='3d' and after labeling:
            self.style_dark_3d_axes(self.ax_scatter)


        # Keep the scatter plotting box square
        try:
            self.ax_scatter.set_box_aspect(1)  # best option (matplotlib >=3.3)
        except Exception:
            self.ax_scatter.set_aspect('equal', adjustable='box')

        self.ax_scatter.set_facecolor("none")
        self.style_dark_axes(self.ax_scatter)
        self.fig_scatter.patch.set_alpha(0.0)

        self.scatter_cbar = None  # reset colorbar handle

        # Check if we have a valid embedding result
        if self.embedding_result is None or len(self.embedding_result) == 0:
            self.ax_text_center("No embedding data available.\nPlease check data preprocessing.", title="Embedding Unavailable")
            self.canvas_scatter.draw_idle()
            return

        # ---- Global bounds over ALL points (independent of filters) ----
        emb = self.embedding_result
        n_samples, n_dims = emb.shape

        mask = np.isfinite(emb[:, 0]) & np.isfinite(emb[:, 1])
        emb_valid = emb[mask]
        if emb_valid.size == 0:
            # show "no embedding"
            ...
        x_all = emb_valid[:, 0]
        y_all = emb_valid[:, 1]

        x_min, x_max = float(x_all.min()), float(x_all.max())
        y_min, y_max = float(y_all.min()), float(y_all.max())

        def with_margin(vmin, vmax):
            if vmin == vmax:
                delta = 1.0
                return vmin - delta, vmax + delta
            span = vmax - vmin
            pad = 0.05 * span
            return vmin - pad, vmax + pad

        x_min, x_max = with_margin(x_min, x_max)
        y_min, y_max = with_margin(y_min, y_max)

        z_min = z_max = None
        if self.embedding_dim == 3 and n_dims >= 3:
            z_all = emb[:, 2]

            # IMPORTANT: only finite values
            z_valid = z_all[np.isfinite(z_all)]

            if z_valid.size > 0:
                z_min, z_max = float(z_valid.min()), float(z_valid.max())
                z_min, z_max = with_margin(z_min, z_max)

        # ---- Visible indices based on filters ----
        indices = self.get_visible_indices()
        if len(indices) == 0:
            self.ax_text_center("No points match the current filter.", title=f"{self.embedding_method} Embedding (filtered)")
            self.ax_scatter.set_xlim(x_min, x_max)
            self.ax_scatter.set_ylim(y_min, y_max)
            self.ax_scatter.set_title(f"{self.embedding_method} Embedding (filtered)")
            self.canvas_scatter.draw_idle()
            return

        # mapping from global index -> position in scatter arrays
        self.index_to_scatter_pos = {idx: pos for pos, idx in enumerate(indices)}

        color_attr = self.color_combo.currentText()
        cmap_name = self.cmap_combo.currentText() if hasattr(self, "cmap_combo") else "tab10"

        colors_array = None
        cmap_to_use = None
        unique_vals = []
        n_unique = 0

        if color_attr != "None":
            values = [self.data_objects[i].metadata.get(color_attr, None) for i in indices]
            # Filter out None, NaN, and ensure consistent types
            valid_vals = []
            for v in values:
                if v is None:
                    continue
                # Check for NaN (works for both float and numpy types)
                try:
                    if pd.isna(v):
                        continue
                except (TypeError, ValueError):
                    pass
                valid_vals.append(v)
            
            # Sort with consistent type handling
            try:
                unique_vals = sorted(set(valid_vals))
            except TypeError:
                # Mixed types - convert all to strings for sorting
                unique_vals = sorted(set(str(v) for v in valid_vals))
            
            n_unique = len(unique_vals)

            if n_unique > 0:
                color_map = {val: idx for idx, val in enumerate(unique_vals)}
                colors_array = [color_map.get(v, -1) if v is not None and not (isinstance(v, float) and pd.isna(v)) else -1 for v in values]
                cmap_to_use = cmap_name
            else:
                colors_array = DEFAULT_BLUE
        else:
            colors_array = DEFAULT_BLUE
            cmap_to_use = None

        try:
            if self.embedding_dim == 2:
                coords = self.embedding_result[indices, :2]
                # Base scatter (all points)
                self.scatter = self.ax_scatter.scatter(
                    coords[:, 0], coords[:, 1],
                    c=colors_array, cmap=cmap_to_use,
                    picker=5, zorder=1
                )

                # Overlay scatter for selected points (initially empty, always on top)
                self.scatter_selected = self.ax_scatter.scatter(
                    [], [], marker='*', s=120,
                    edgecolors='white', facecolors='none',
                    linewidths=1.0, zorder=10, picker=False
                )

                self.ax_scatter.set_xlabel(f"{self.embedding_method} 1")
                self.ax_scatter.set_ylabel(f"{self.embedding_method} 2")
                self.ax_scatter.set_xlim(x_min, x_max)
                self.ax_scatter.set_ylim(y_min, y_max)
            else:
                coords = self.embedding_result[indices, :3]
                self.scatter = self.ax_scatter.scatter(
                    coords[:, 0], coords[:, 1], coords[:, 2],
                    c=colors_array, cmap=cmap_to_use,
                    picker=10, zorder=1
                )
                # Overlay scatter for selected points (initially empty, always on top)
                self.scatter_selected = self.ax_scatter.scatter(
                    [], [], marker='*', s=120,
                    edgecolors='white', facecolors='none',
                    linewidths=5.0, zorder=10, picker=False
                )

                self.ax_scatter.set_xlabel(f"{self.embedding_method} 1")
                self.ax_scatter.set_ylabel(f"{self.embedding_method} 2")
                self.ax_scatter.set_zlabel(f"{self.embedding_method} 3")
                # Make z-axis label white for 3D plots
                self.ax_scatter.zaxis.label.set_color("white")
                self.ax_scatter.set_xlim(x_min, x_max)
                self.ax_scatter.set_ylim(y_min, y_max)
                if z_min is not None and z_max is not None:
                    self.ax_scatter.set_zlim(z_min, z_max)

            title_suffix = "" if len(indices) == len(self.data_objects) else f" (filtered: {len(indices)}/{len(self.data_objects)})"
            self.ax_scatter.set_title(f"{self.embedding_method} {title_suffix}")
        except Exception as e:
            print(f"Error plotting embedding: {e}")
            import traceback
            traceback.print_exc()
            self.ax_text_center(f"Error plotting embedding:\n{str(e)}", title=f"{self.embedding_method} Embedding (error)")
            self.canvas_scatter.draw_idle()
            return

        # ---- Colorbar for metadata-based coloring (DO NOT squeeze square scatter) ----
        self.scatter_cbar = None
        if color_attr not in ("None") and cmap_to_use is not None and n_unique > 0:
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes

            cmap_obj = plt.get_cmap(cmap_to_use)
            norm = colors.Normalize(vmin=0, vmax=max(n_unique - 1, 1))
            sm = cm.ScalarMappable(norm=norm, cmap=cmap_obj)

            # Put colorbar BELOW the scatter without resizing the scatter axes.
            # Increase height a bit so the label fits.
            cax = inset_axes(
                self.ax_scatter,
                width="94%",
                height="7%",
                loc="lower center",
                bbox_to_anchor=(0.03, -0.22, 0.94, 1.0),
                bbox_transform=self.ax_scatter.transAxes,
                borderpad=0
            )

            self.scatter_cbar = self.fig_scatter.colorbar(sm, cax=cax, orientation="horizontal")
            self.scatter_cbar.set_ticks(list(range(n_unique)))
            self.scatter_cbar.set_ticklabels([str(v) for v in unique_vals])

            # Make label + ticks white
            self.scatter_cbar.ax.tick_params(colors="white", labelsize=8)
            for t in self.scatter_cbar.ax.get_xticklabels():
                t.set_rotation(45)
                t.set_ha("right")

            # IMPORTANT: label visible + padded
            self.scatter_cbar.set_label(color_attr, color="white", fontsize=9, labelpad=6)

            for spine in self.scatter_cbar.ax.spines.values():
                spine.set_color("white")

            # Prevent clipping of the inset (label/ticks) by giving the figure extra bottom room.
            # This does NOT change the scatter axes size because the colorbar is an inset.
            self.fig_scatter.subplots_adjust(bottom=0.22)
        else:
            # If no colorbar, reclaim the space
            self.fig_scatter.subplots_adjust(bottom=0.08)

        # ---- Recreate lasso selector if it was active ----
        if self.embedding_dim == 2 and hasattr(self, "btn_select"):
            if self.btn_select.isChecked():
                # Remove old lasso (if any) bound to previous axes
                if self.lasso is not None:
                    try:
                        self.lasso.disconnect_events()
                    except Exception:
                        pass
                    self.lasso = None

                from matplotlib.widgets import LassoSelector
                self.lasso = LassoSelector(
                    self.ax_scatter,
                    self.on_lasso_select,
                    useblit=True,
                    props=dict(color="white", linewidth=1.5, alpha=0.95)
                )
            else:
                # ensure no stale lasso
                if self.lasso is not None:
                    try:
                        self.lasso.disconnect_events()
                    except Exception:
                        pass
                    self.lasso = None
        else:
            # In 3D mode or if button doesn't exist, disable lasso
            if self.lasso is not None:
                try:
                    self.lasso.disconnect_events()
                except Exception:
                    pass
                self.lasso = None
            if hasattr(self, "btn_select"):
                self.btn_select.setChecked(False)
                self.btn_select.setText("Enable Lasso Selection")

        self.canvas_scatter.draw_idle()

    def update_pca_loadings_plot(self):
        """Draw PCA loadings in separate canvas, x-range forced to match spectra exactly."""
        show = (
            self.embedding_method == "PCA"
            and self.pca_loadings is not None
            and self.pca_explained_var is not None
            and bool(self.data_objects)
            and hasattr(self, "_spectra_xlim")
        )

        # Do NOT hide the widget (that causes resizing). Switch stacked page instead.
        if hasattr(self, "loadings_stack"):
            self.loadings_stack.setCurrentWidget(self.canvas_load if show else self._loadings_blank)

        if not show:
            return

        self.canvas_load.setVisible(bool(show))
        if not show:
            self.fig_load.clear()
            self.canvas_load.draw_idle()
            return

        self.fig_load.clear()
        try:
            self.fig_load.patch.set_alpha(0.0)
        except Exception:
            pass

        ax = self.fig_load.add_subplot(111)
        ax.set_facecolor("none")

        x = self.data_objects[0].raman_shift_cm  # common grid
        n_show = min(int(self.n_pcs_for_loading), self.pca_loadings.shape[0])

        for k in range(n_show):
            ev = self.pca_explained_var[k] if k < len(self.pca_explained_var) else None
            label = f"PC{k+1}" + (f" ({ev*100:.1f}%)" if ev is not None else "")
            ax.plot(x, self.pca_loadings[k, :], linewidth=1.0, label=label)

        ax.axhline(0, linewidth=0.8)
        ax.set_title("PCA loadings")
        ax.set_xlabel("Raman Shift (cm⁻¹)")
        ax.set_ylabel("Loading")
        ax.legend(fontsize=8, ncol=min(n_show, 3))

        self.style_dark_axes(ax)

        # THE FIX: force exact same x-range and remove margins
        xmin, xmax = self._spectra_xlim
        ax.set_xlim(xmin, xmax)
        ax.margins(x=0)
        ax.set_autoscalex_on(False)

        # Match left/right margins with spectra plot
        if hasattr(self, "_shared_left") and hasattr(self, "_shared_right"):
            self.fig_load.subplots_adjust(left=self._shared_left, right=self._shared_right, top=0.90, bottom=0.25)

        self.canvas_load.draw_idle()


    def _plot_pca_loadings_axis(self, ax):
        """Plot PCA loadings (PC1..PCn) vs Raman shift on an existing axis."""
        if self.pca_loadings is None or self.pca_explained_var is None or not self.data_objects:
            ax.set_title("PCA loadings unavailable")
            ax.axis("off")
            return

        x = self.data_objects[0].raman_shift_cm
        n_show = min(int(self.n_pcs_for_loading), self.pca_loadings.shape[0])

        for k in range(n_show):
            ev = self.pca_explained_var[k] if k < len(self.pca_explained_var) else None
            label = f"PC{k+1}" + (f" ({ev*100:.1f}%)" if ev is not None else "")
            ax.plot(x, self.pca_loadings[k, :], linewidth=1.0, label=label)

        ax.axhline(0, linewidth=0.8)
        ax.set_title("PCA loadings")
        ax.set_xlabel("Raman Shift (cm⁻¹)")
        ax.set_ylabel("Loading")
        ax.legend(fontsize=8, ncol=min(n_show, 3))

        ax.set_facecolor("none")
        try:
            ax.figure.patch.set_alpha(0.0)
        except Exception:
            pass


    def on_pca_loading_count_changed(self, value):
        """Called when user changes 'PCA Loadings PCs'."""
        self.n_pcs_for_loading = int(value)

        if self.embedding_method == "PCA":
            # Force PCA refit with new component count used for loadings
            for k in [k for k in self.embedding_cache.keys() if k[0] == "PCA"]:
                del self.embedding_cache[k]

            self.compute_embedding()
            self.plot_embedding()
            self.update_line_plot()
            self.update_pca_loadings_plot()
        else:
            self.plot_embedding()
            self.update_line_plot()

    def update_line_plot(self):
        """Update spectra plot (combined/grouped). Also locks a shared x-range for loadings."""
        self.fig_line.clear()
        self.selected_index_to_line = {}

        # Transparent background
        try:
            self.fig_line.patch.set_alpha(0.0)
        except Exception:
            pass

        # Clear metadata table
        has_table = hasattr(self, "metadata_table")
        if has_table:
            self.metadata_table.clearContents()
            self.metadata_table.setRowCount(0)
            self.metadata_table.setColumnCount(0)

        # - else if startup default -> plot all visible spectra
        if not self.selected_indices:
            self.selected_indices = self.get_visible_indices()

        # Shared x-axis range (authoritative!)
        # Spectra are aligned to a common grid, so use any selected spectrum.
        x = self.data_objects[self.selected_indices[0]].raman_shift_cm
        xmin, xmax = float(np.min(x)), float(np.max(x))
        self._spectra_xlim = (xmin, xmax)

        # Build metadata keys for table
        all_keys = set()
        for idx in self.selected_indices:
            all_keys.update(self.data_objects[idx].metadata.keys())
        metadata_keys = sorted(all_keys)

        if has_table:
            self.metadata_table.setColumnCount(len(metadata_keys))
            self.metadata_table.setHorizontalHeaderLabels([k.replace("_", " ").title() for k in metadata_keys])
            self.metadata_table.setRowCount(len(self.selected_indices))

        row_for_idx = {idx: r for r, idx in enumerate(self.selected_indices)}

        # Colors from scatter
        scatter_colors = None
        index_to_pos = getattr(self, "index_to_scatter_pos", {}) or {}
        if hasattr(self, "scatter") and self.scatter is not None:
            try:
                scatter_colors = self.scatter.get_facecolors()
            except Exception:
                scatter_colors = None

        def pick_color(idx, fallback):
            if scatter_colors is not None and len(scatter_colors) > 0:
                pos = index_to_pos.get(idx)
                if pos is not None and pos < len(scatter_colors):
                    return scatter_colors[pos]
            return fallback

        # ---- Plot ----
        if self.plot_mode == "combined":
            ax = self.fig_line.add_subplot(111)
            ax.set_facecolor("none")

            for idx in self.selected_indices:
                obj = self.data_objects[idx]
                c = pick_color(idx, DEFAULT_BLUE)

                line, = ax.plot(obj.raman_shift_cm, obj.intensity, color=c, alpha=0.7, picker=5)
                self.selected_index_to_line[idx] = line

                if has_table:
                    row = row_for_idx[idx]
                    for col, key in enumerate(metadata_keys):
                        val = obj.metadata.get(key, "")
                        item = SortableTableWidgetItem(str(val))
                        if col == 0:
                            item.setData(Qt.UserRole, idx)
                        self.metadata_table.setItem(row, col, item)

            ax.set_title(f"All Selected Spectra ({len(self.selected_indices)} total)")
            ax.set_xlabel("Raman Shift (cm⁻¹)")
            ax.set_ylabel("Intensity")
            self.style_dark_axes(ax)

            # FORCE x sync
            ax.set_xlim(xmin, xmax)
            ax.margins(x=0)
            ax.set_autoscalex_on(False)

        else:
            patient_groups = {}
            for idx in self.selected_indices:
                pid = self.data_objects[idx].metadata.get("patient", "Unknown")
                patient_groups.setdefault(pid, []).append(idx)

            n_patients = len(patient_groups)
            for plot_idx, (pid, indices) in enumerate(sorted(patient_groups.items())):
                ax = self.fig_line.add_subplot(n_patients, 1, plot_idx + 1)
                ax.set_facecolor("none")

                for idx in indices:
                    obj = self.data_objects[idx]
                    c = pick_color(idx, "white")

                    line, = ax.plot(obj.raman_shift_cm, obj.intensity, color=c, alpha=0.7, picker=5)
                    self.selected_index_to_line[idx] = line

                    if has_table:
                        row = row_for_idx[idx]
                        for col, key in enumerate(metadata_keys):
                            val = obj.metadata.get(key, "")
                            item = SortableTableWidgetItem(str(val))
                            if col == 0:
                                item.setData(Qt.UserRole, idx)
                            self.metadata_table.setItem(row, col, item)

                ax.set_title(f"Patient {pid} ({len(indices)} spectra)", fontsize=9, pad=3)
                ax.set_xlabel("Raman Shift (cm⁻¹)", fontsize=8)
                ax.set_ylabel("Intensity", fontsize=8)
                ax.tick_params(labelsize=7)
                self.style_dark_axes(ax)

                # FORCE x sync on every subplot
                ax.set_xlim(xmin, xmax)
                ax.margins(x=0)
                ax.set_autoscalex_on(False)

                if plot_idx < n_patients - 1:
                    ax.set_xlabel("")

        self.update_line_zorder()

        # Fix margins so loadings can match left/right exactly
        self.fig_line.subplots_adjust(left=0.10, right=0.985, top=0.95, bottom=0.10, hspace=0.35)
        self.canvas_line.draw()

        sp = self.fig_line.subplotpars
        self._shared_left = sp.left
        self._shared_right = sp.right

        self.update_pca_loadings_plot()


    def on_filter_attr_changed(self, filter_obj, text):
        """When a filter's metadata key (type) changes."""
        filter_obj['attr'] = text
        value_combo = filter_obj['value_combo']

        if text == "None":
            value_combo.blockSignals(True)
            value_combo.clear()
            value_combo.addItem("All")
            value_combo.setEnabled(False)
            value_combo.blockSignals(False)
            filter_obj['value'] = "All"
            self.update_visible_indices()
            return

        # Collect unique values for this metadata field
        values = sorted({
            str(obj.metadata.get(text))
            for obj in self.data_objects
            if text in obj.metadata
        }, key=lambda x: float(x) if x.replace('.', '', 1).isdigit() else float('inf'))

        value_combo.blockSignals(True)
        value_combo.clear()
        value_combo.addItem("All")
        for v in values:
            value_combo.addItem(v)
        value_combo.setEnabled(True)
        value_combo.blockSignals(False)

        # Default to "All" for the new attribute
        filter_obj['value'] = "All"
        self.update_visible_indices()

    def on_filter_value_changed(self, filter_obj, text):
        """When a filter's value changes."""
        if not filter_obj['value_combo'].isEnabled():
            return
        filter_obj['value'] = text
        self.update_visible_indices()

    def update_visible_indices(self):
        """Recompute visible indices based on ALL active filters (AND logic)."""
        # Build list of (attr, value) for active filters
        active_filters = []
        for f in self.filters:
            attr = f.get('attr')
            val = f.get('value')
            if not attr or attr == "None" or not val or val == "All":
                continue
            active_filters.append((attr, val))

        if not active_filters:
            # No active filters → show all
            self.visible_indices = list(range(len(self.data_objects)))
        else:
            visible = []
            for i, obj in enumerate(self.data_objects):
                ok = True
                for attr, val in active_filters:
                    v = obj.metadata.get(attr)
                    if str(v) != val:
                        ok = False
                        break
                if ok:
                    visible.append(i)
            self.visible_indices = visible

        # Keep only selected indices that are still visible
        self.selected_indices = [i for i in self.selected_indices if i in self.visible_indices]

        # Redraw
        self.compute_embedding(indices=self.visible_indices, force=True)
        self.plot_embedding()
        self.update_line_plot()

    def on_lasso_select(self, verts):
        """Handle lasso selection in the embedding plot (2D only)."""
        if self.embedding_dim != 2:
            return

        visible_indices = self.visible_indices or list(range(len(self.data_objects)))
        if not visible_indices:
            return

        path = Path(verts)
        points = self.embedding_result[visible_indices, :2]
        selected_mask = path.contains_points(points)
        self.selected_indices = [visible_indices[i] for i, s in enumerate(selected_mask) if s]
        self.update_line_plot()
    
        if self.embedding_method == "PCA":
            self.canvas_line.draw_idle()

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

    def update_line_zorder(self, primary_idx=None):
        """
        Ensure one spectrum line is drawn on top by setting its z-order higher.
        primary_idx: the spectrum index we want on top (if provided).
        """
        if primary_idx is not None:
            self.top_index = primary_idx

        if not self.selected_index_to_line:
            return

        base_z = 1
        top_z = 10

        # If current top_index is not in the mapping anymore, clear it
        if self.top_index not in self.selected_index_to_line:
            self.top_index = None

        for idx, line in self.selected_index_to_line.items():
            if idx == self.top_index:
                line.set_zorder(top_z)
            else:
                line.set_zorder(base_z)

        self.canvas_line.draw_idle()

    def clear_selection(self):
        self.selected_indices = []
        self.fig_line.clear()
        ax = self.fig_line.add_subplot(111)
        ax.set_title("Select points to view spectra", color='white')
        ax.axis('off')

        if hasattr(self, "metadata_table"):
            self.metadata_table.clearContents()
            self.metadata_table.setRowCount(0)
            self.metadata_table.setColumnCount(0)

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
        """When a line in the spectra plot is clicked, select the matching row in the metadata table and bring it on top."""
        if event.artist not in self.selected_index_to_line.values():
            return

        # Find which global index this line corresponds to
        idx = None
        for k, v in self.selected_index_to_line.items():
            if v is event.artist:
                idx = k
                break

        if idx is None:
            return

        if not hasattr(self, "metadata_table") or self.metadata_table.rowCount() == 0:
            # Even if there's no table (shouldn't happen now), still bring it on top
            self.update_line_zorder(idx)
            return

        # Find the row in the metadata table with this idx stored in the first column's UserRole
        row_to_select = None
        for row in range(self.metadata_table.rowCount()):
            item0 = self.metadata_table.item(row, 0)
            if item0 is None:
                continue
            stored_idx = item0.data(Qt.UserRole)
            if stored_idx == idx:
                row_to_select = row
                break

        # Update table selection
        self.metadata_table.blockSignals(True)
        self.metadata_table.clearSelection()
        if row_to_select is not None:
            self.metadata_table.selectRow(row_to_select)
        self.metadata_table.blockSignals(False)

        # Update highlights and bring clicked line on top
        self.update_line_highlights()
        self.update_line_zorder(idx)

    def on_line_click(self, event):
        """Clear metadata/table selection when clicking on blank space in the line plot."""
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
            # Clear row selection in the metadata table
            if hasattr(self, "metadata_table"):
                self.metadata_table.clearSelection()

            # Clear scatter highlight + hover state + redraw styles
            self.clear_scatter_highlight()
            self.hovered_index = None
            self.update_line_highlights()

    def update_line_highlights(self):
        """Update line widths/colors based on hover & metadata table selection."""
        scatter_colors = None
        index_to_pos = {}

        if hasattr(self, 'scatter') and self.scatter is not None:
            scatter_colors = self.scatter.get_facecolors()
            index_to_pos = getattr(self, 'index_to_scatter_pos', {})

        # Indices selected in the metadata table
        selected_indices_from_list = []
        if hasattr(self, "metadata_table"):
            selected_items = self.metadata_table.selectedItems()
            selected_rows = {item.row() for item in selected_items}
            for row in selected_rows:
                item0 = self.metadata_table.item(row, 0)
                if item0 is None:
                    continue
                idx = item0.data(Qt.UserRole)
                if idx is not None:
                    selected_indices_from_list.append(idx)

        color_attr = self.color_combo.currentText() if hasattr(self, "color_combo") else "None"

        for idx, line in self.selected_index_to_line.items():
            if idx == self.hovered_index or idx in selected_indices_from_list:
                # Highlighted state
                line.set_linewidth(2.5)
                line.set_color('white')
                line.set_alpha(1.0)
            else:
                # Default state mirrors scatter colors
                line.set_linewidth(1)

                if scatter_colors is not None and len(scatter_colors) > 0:
                    pos = index_to_pos.get(idx)
                    if pos is not None and pos < len(scatter_colors):
                        line.set_color(scatter_colors[pos])
                    else:
                        line.set_color(DEFAULT_BLUE)
                else:
                    line.set_color(DEFAULT_BLUE)

                line.set_alpha(0.7)

        # Decide which spectrum should be on top:
        # 1) first selected in the metadata table, else
        # 2) currently hovered line, else keep previous top_index
        primary_idx = None
        if selected_indices_from_list:
            primary_idx = selected_indices_from_list[0]
        elif self.hovered_index is not None:
            primary_idx = self.hovered_index

        self.update_line_zorder(primary_idx)

        self.canvas_line.draw_idle()

        # Also reflect selection in the embedding plot
        self.update_scatter_highlight(selected_indices_from_list)

    def update_scatter_highlight(self, selected_indices):
        """Highlight selected spectra in the scatter plot and bring them on top."""
        if not hasattr(self, 'scatter') or self.scatter is None:
            return

        visible_indices = self.visible_indices or list(range(len(self.data_objects)))
        if not visible_indices:
            return

        # Keep your existing marker-shape logic (stars vs circles)
        # (Assumes you kept your custom update_scatter_markers)
        try:
            self.update_scatter_markers(selected_indices)
        except Exception:
            pass

        # Overlay: draw selected points in a separate scatter on top (2D only)
        if self.embedding_dim == 2 and hasattr(self, "scatter_selected") and self.scatter_selected is not None:
            coords = []
            for idx in selected_indices:
                if 0 <= idx < len(self.embedding_result):
                    coords.append(self.embedding_result[idx, :2])

            if coords:
                arr = np.vstack(coords)
                self.scatter_selected.set_offsets(arr)
            else:
                # no selected points → clear overlay
                self.scatter_selected.set_offsets(np.empty((0, 2)))

        self.canvas_scatter.draw_idle()

        # --- Ensure PCA loadings update immediately with selection ---
        if self.embedding_method == "PCA" and self.pca_loadings is not None:
            self.canvas_line.draw_idle()

    def clear_scatter_highlight(self):
        """Clear scatter plot highlights."""
        if hasattr(self, 'scatter') and self.scatter is not None:
            visible_indices = self.get_visible_indices()
            if visible_indices:
                # Reset markers
                try:
                    self.update_scatter_markers([])
                except Exception:
                    pass

            # Clear overlay
            if hasattr(self, "scatter_selected") and self.scatter_selected is not None and self.embedding_dim == 2:
                self.scatter_selected.set_offsets(np.empty((0, 2)))

            self.canvas_scatter.draw_idle()

    def _on_scatter_draw(self, event):
        """
        Called whenever the scatter FigureCanvas is drawn.
        We only update the line plot colors if a color-change triggered it
        (via _pending_line_color_update).
        """
        if not getattr(self, "_pending_line_color_update", False):
            return

        # Reset the flag: this draw has satisfied the pending update
        self._pending_line_color_update = False

        # Now the scatter's facecolors are fully up-to-date.
        # Rebuild the line plot and highlights using those colors.
        if self.selected_indices:
            self.update_line_plot()
            self.update_line_highlights()
        else:
            self.update_line_plot()

    def update_scatter_markers(self, selected_indices):
        """Update scatter markers: stars for selected, circles for others."""
        if not hasattr(self, 'scatter') or self.scatter is None:
            return

        visible_indices = self.get_visible_indices()
        if not visible_indices:
            return

        # Build new marker list (one per point)
        # Matplotlib supports per-point markers via PathCollection.set_paths()
        from matplotlib.markers import MarkerStyle
        from matplotlib.transforms import Affine2D

        circle = MarkerStyle("o").get_path().transformed(
            MarkerStyle("o").get_transform()
        )
        star_style = MarkerStyle("*")
        base_path = star_style.get_path()
        base_transform = star_style.get_transform()
        scaled_transform = base_transform + Affine2D().scale(2.0)  # make the star bigger
        star = base_path.transformed(scaled_transform)

        paths = []
        edgecolors = []
        for idx in visible_indices:
            if idx in selected_indices:
                paths.append(star)
                edgecolors.append('white')
            else:
                paths.append(circle)
                edgecolors.append('none')

        self.scatter.set_paths(paths)
        self.scatter.set_edgecolors(edgecolors)
        self.scatter.set_linewidths(2)
        self.canvas_scatter.draw_idle()

    def on_metadata_header_clicked(self, column):
        """Sort metadata table by the clicked column (ascending)."""
        if hasattr(self, "metadata_table"):
            self.metadata_table.sortItems(column, Qt.AscendingOrder)
            # Keep line highlights / scatter highlights in sync
            self.update_line_highlights()

    def _make_mpl_transparent(self, fig, canvas):
        fig.patch.set_alpha(0.0)
        canvas.setStyleSheet("background: transparent;")
        for ax in fig.get_axes():
            ax.set_facecolor("none")

    def save_metadata(self):
        metadata_list = [self.data_objects[idx].metadata for idx in self.selected_indices]
        filename, _ = QFileDialog.getSaveFileName(self, "Save Metadata", "", "JSON Files (*.json)")
        if filename:
            with open(filename, "w") as f:
                json.dump(metadata_list, f, indent=4)

    def update_preprocessing(self):
        """Re-process data with current preprocessing parameters, preserving filters and color selection."""
        # 1) Update preprocessor from UI
        self.update_preprocessor_from_ui()

        # 2) Remember current color selection
        prev_color_attr = None
        if hasattr(self, "color_combo"):
            prev_color_attr = self.color_combo.currentText()

        # 3) Remember current filters (order-preserving)f
        saved_filters = []
        if hasattr(self, "filters"):
            for f in self.filters:
                saved_filters.append((f.get("attr"), f.get("value")))

        # 4) Re-preprocess all original spectra
        self.data_objects = [
            self.preprocessor.preprocess(copy.deepcopy(obj))
            for obj in self.original_data_objects
        ]

        # 5) Align spectra to common grid
        self.align_spectra_to_common_grid()

        # 6) Clear embedding cache and current embedding
        self.embedding_cache.clear()
        self.embedding_result = None

        # 7) Recompute embedding with updated data
        self.compute_embedding()
        self.update_pca_loadings_plot()

        # 8) Refresh color options based on new metadata, but restore selection if possible
        if hasattr(self, "color_combo"):
            # Repopulate items
            self.populate_color_combo()
            # Try to restore previous color-by attribute
            if prev_color_attr and self.color_combo.findText(prev_color_attr) != -1:
                self.color_combo.blockSignals(True)
                self.color_combo.setCurrentText(prev_color_attr)
                self.color_combo.blockSignals(False)
            else:
                # Fallback if the old key no longer exists
                self.color_combo.blockSignals(True)
                self.color_combo.setCurrentText("None")
                self.color_combo.blockSignals(False)

        # 9) Rebuild filter combos using saved (attr, value) per row
        if hasattr(self, "filters_container_layout") and hasattr(self, "filters"):
            metadata_keys = set()
            for obj in self.data_objects:
                metadata_keys.update(obj.metadata.keys())
            metadata_keys = sorted(metadata_keys)

            for f, (saved_attr, saved_val) in zip(self.filters, saved_filters):
                attr_combo = f["attr_combo"]
                value_combo = f["value_combo"]

                # Re-populate attribute combo
                attr_combo.blockSignals(True)
                self.populate_filter_attr_combo(attr_combo)
                # Restore attribute if still available
                if saved_attr and saved_attr != "None" and saved_attr in metadata_keys:
                    attr_combo.setCurrentText(saved_attr)
                    f["attr"] = saved_attr

                    # Rebuild value combo for this attribute
                    values = sorted({
                        str(obj.metadata.get(saved_attr))
                        for obj in self.data_objects
                        if saved_attr in obj.metadata
                    })

                    value_combo.blockSignals(True)
                    value_combo.clear()
                    value_combo.addItem("All")
                    for v in values:
                        value_combo.addItem(v)
                    value_combo.setEnabled(True)

                    # Restore value if still valid
                    if saved_val and saved_val != "All" and value_combo.findText(saved_val) != -1:
                        value_combo.setCurrentText(saved_val)
                        f["value"] = saved_val
                    else:
                        value_combo.setCurrentText("All")
                        f["value"] = "All"
                    value_combo.blockSignals(False)
                else:
                    # Attribute no longer valid → reset this filter
                    attr_combo.setCurrentText("None")
                    f["attr"] = None

                    value_combo.blockSignals(True)
                    value_combo.clear()
                    value_combo.addItem("All")
                    value_combo.setEnabled(False)
                    f["value"] = "All"
                    value_combo.blockSignals(False)

                attr_combo.blockSignals(False)

        self.update_visible_indices()

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

    def style_dark_axes(self, ax):
        """Make all axis elements white for dark theme."""
        if ax is None:
            return

        # Axis labels & title
        ax.title.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")

        # Tick labels
        ax.tick_params(axis='both', colors='white')

        # Axis spines
        for spine in ax.spines.values():
            spine.set_color("white")

        # Grid (if enabled)
        ax.grid(color="white", alpha=0.15)

        # Legend
        leg = ax.get_legend()
        if leg is not None:
            leg.get_frame().set_facecolor("none")
            leg.get_frame().set_edgecolor("white")
            for text in leg.get_texts():
                text.set_color("white")

    def style_dark_3d_axes(self, ax):
        """Dark, semi-transparent 3D box panes + subtle white grid/ticks."""
        # Only apply to 3D axes
        if not hasattr(ax, "get_zlim"):
            return

        pane_rgba = (0.3, 0.3, 0.3, 0.25)  # darker + more transparent
        grid_rgba = (1.0, 1.0, 1.0, 0.12)

        # --- Newer Matplotlib: axis has .pane ---
        for axis in (getattr(ax, "xaxis", None), getattr(ax, "yaxis", None), getattr(ax, "zaxis", None)):
            if axis is None:
                continue

            pane = getattr(axis, "pane", None)
            if pane is not None:
                try:
                    pane.set_facecolor(pane_rgba)
                except Exception:
                    pass
                try:
                    pane.set_edgecolor((1, 1, 1, 0.25))
                except Exception:
                    pass

            # Grid color (3D uses _axinfo)
            try:
                axis._axinfo["grid"]["color"] = grid_rgba
            except Exception:
                pass

        # --- Older Matplotlib fallback: w_xaxis / w_yaxis / w_zaxis ---
        for wax in (getattr(ax, "w_xaxis", None), getattr(ax, "w_yaxis", None), getattr(ax, "w_zaxis", None)):
            if wax is None:
                continue
            try:
                wax.set_pane_color(pane_rgba)
            except Exception:
                pass
            try:
                wax._axinfo["grid"]["color"] = grid_rgba
            except Exception:
                pass

        # Labels/ticks white
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.zaxis.label.set_color("white")

        # Axis lines white-ish (best-effort across versions)
        for axis_name in ("xaxis", "yaxis", "zaxis"):
            axis_obj = getattr(ax, axis_name, None)
            line = getattr(axis_obj, "line", None)
            if line is not None:
                try:
                    line.set_color((1, 1, 1, 0.6))
                except Exception:
                    pass

    def ax_text_center(self, msg, title=None):
        """Centered overlay text that works for both 2D and 3D axes."""
        if getattr(self.ax_scatter, "name", "") == "3d":
            self.ax_scatter.text2D(
                0.5, 0.5, msg,
                ha="center", va="center",
                transform=self.ax_scatter.transAxes,
                color="white"
            )
        else:
            self.ax_scatter.text(
                0.5, 0.5, msg,
                ha="center", va="center",
                transform=self.ax_scatter.transAxes,
                color="white"
            )
        if title is not None:
            self.ax_scatter.set_title(title)


class ScrollableComboBox(QComboBox):
    """
    QComboBox that lets you change the current item with the mouse wheel
    (without having to open the dropdown). We consume the wheel event so
    the parent scroll area won't scroll instead.
    """
    def wheelEvent(self, event):
        if self.count() == 0:
            event.ignore()
            return

        delta = event.angleDelta().y()

        if delta > 0:
            # scroll up → previous item
            new_index = max(self.currentIndex() - 1, 0)
        elif delta < 0:
            # scroll down → next item
            new_index = min(self.currentIndex() + 1, self.count() - 1)
        else:
            event.ignore()
            return

        if new_index != self.currentIndex():
            self.setCurrentIndex(new_index)

        # Don't let the event bubble up to the scroll area
        event.accept()

class SortableTableWidgetItem(QTableWidgetItem):
    """QTableWidgetItem that sorts numerically when possible."""
    def __lt__(self, other):
        # If the other item isn't our type, fall back to default behavior
        if not isinstance(other, QTableWidgetItem):
            return super().__lt__(other)

        left_text = self.text()
        right_text = other.text()

        # Try numeric comparison first
        try:
            left_val = float(left_text)
            right_val = float(right_text)
            return left_val < right_val
        except ValueError:
            # Fallback: case-insensitive string comparison
            return left_text.lower() < right_text.lower()

if __name__ == "__main__":
    # Example usage - update these paths to your actual data
    data_folder = r'/Users/yifeigu/Library/CloudStorage/Box-Box/Carney Lab Shared/Data/Raman_Robot/2026_01_28_1'
    
    # Create default config
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

    preprocessor = SpectrumPreprocessorWrapper.from_dict(default_config)
    dataset = RamanRobotDataset(data_folder, preprocessor, augmentor=None)

    app = QApplication(sys.argv)
    apply_dark_theme(app)
    window = SpectraViewer(dataset.db, preprocessor)
    window.resize(1400, 800)
    window.show()
    sys.exit(app.exec_())