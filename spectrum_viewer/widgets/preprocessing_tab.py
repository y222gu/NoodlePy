"""Preprocessing controls tab widget."""

import json
import yaml
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QCheckBox, QDoubleSpinBox, QSpinBox, QComboBox,
    QFormLayout, QScrollArea, QFrame, QRadioButton, QButtonGroup,
    QFileDialog
)
from PyQt5.QtCore import pyqtSignal

from utils.raman_robot_dataset import SpectrumPreprocessorWrapper


class PreprocessingTab(QWidget):
    """Tab widget for preprocessing controls."""

    # Signals
    preprocessing_changed = pyqtSignal()  # Emitted when Re-process is clicked

    def __init__(self, preprocessor: SpectrumPreprocessorWrapper, parent=None):
        super().__init__(parent)

        self.preprocessor = preprocessor
        self._metadata_keys = []

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)

        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(10)

        # === AVERAGING GROUP ===
        avg_group = QGroupBox("Spectrum Averaging")
        avg_layout = QVBoxLayout(avg_group)
        avg_layout.setSpacing(6)

        self.cb_averaging = QCheckBox("Enable Averaging per Droplet")
        self.cb_averaging.setChecked(False)
        self.cb_averaging.stateChanged.connect(self._toggle_averaging_params)
        avg_layout.addWidget(self.cb_averaging)

        self.avg_params_widget = QWidget()
        avg_params_layout = QFormLayout(self.avg_params_widget)
        avg_params_layout.setContentsMargins(20, 0, 0, 0)

        self.avg_by_combo = QComboBox()
        self.avg_by_combo.addItem("None")
        self.avg_by_combo.setToolTip("Group spectra by this metadata field within each droplet")
        avg_params_layout.addRow("Average By:", self.avg_by_combo)

        self.avg_timing_combo = QComboBox()
        self.avg_timing_combo.addItems(["Before Preprocessing", "After Preprocessing"])
        self.avg_timing_combo.setToolTip("When to apply averaging relative to other preprocessing steps")
        avg_params_layout.addRow("Timing:", self.avg_timing_combo)

        avg_layout.addWidget(self.avg_params_widget)
        self.avg_params_widget.setVisible(False)

        scroll_layout.addWidget(avg_group)

        # === PREPROCESSING GROUP ===
        preproc_group = QGroupBox("Preprocessing Steps")
        preproc_layout = QVBoxLayout(preproc_group)
        preproc_layout.setSpacing(8)

        # Master enable/disable checkbox
        master_row = QHBoxLayout()
        self.cb_master_enable = QCheckBox("Enable All Preprocessing")
        self.cb_master_enable.setChecked(True)
        self.cb_master_enable.stateChanged.connect(self._toggle_master_preprocessing)
        self.cb_master_enable.setStyleSheet("font-weight: bold;")
        master_row.addWidget(self.cb_master_enable)
        master_row.addStretch()
        preproc_layout.addLayout(master_row)

        # Container for individual preprocessing steps
        self.preproc_steps_widget = QWidget()
        preproc_steps_layout = QVBoxLayout(self.preproc_steps_widget)
        preproc_steps_layout.setContentsMargins(0, 0, 0, 0)
        preproc_steps_layout.setSpacing(8)

        # ==== CROPPING ====
        self.cb_cropping = QCheckBox("Cropping")
        self.cb_cropping.setChecked(self.preprocessor.cropping)
        self.cb_cropping.stateChanged.connect(self._toggle_cropping_params)
        preproc_steps_layout.addWidget(self.cb_cropping)

        self.crop_params_widget = QWidget()
        crop_params_layout = QFormLayout(self.crop_params_widget)
        crop_params_layout.setContentsMargins(20, 0, 0, 0)

        self.start_raman_spin = QDoubleSpinBox()
        self.start_raman_spin.setRange(0.0, 5000.0)
        self.start_raman_spin.setSingleStep(0.1)
        self.start_raman_spin.setDecimals(3)
        self.start_raman_spin.setValue(self.preprocessor.start_raman_shift_cm)
        crop_params_layout.addRow("Start (cm^-1):", self.start_raman_spin)

        self.end_raman_spin = QDoubleSpinBox()
        self.end_raman_spin.setRange(0.0, 5000.0)
        self.end_raman_spin.setSingleStep(0.1)
        self.end_raman_spin.setDecimals(3)
        self.end_raman_spin.setValue(self.preprocessor.end_raman_shift_cm)
        crop_params_layout.addRow("End (cm^-1):", self.end_raman_spin)

        preproc_steps_layout.addWidget(self.crop_params_widget)

        # ==== BASELINE ====
        self.cb_baseline = QCheckBox("Baseline Correction (airPLS)")
        self.cb_baseline.setChecked(self.preprocessor.baseline_correction)
        self.cb_baseline.stateChanged.connect(self._toggle_baseline_params)
        preproc_steps_layout.addWidget(self.cb_baseline)

        self.baseline_params_widget = QWidget()
        baseline_params_layout = QFormLayout(self.baseline_params_widget)
        baseline_params_layout.setContentsMargins(20, 0, 0, 0)

        self.baseline_lam_spin = QDoubleSpinBox()
        self.baseline_lam_spin.setRange(1.0, 1000000.0)
        self.baseline_lam_spin.setSingleStep(100)
        self.baseline_lam_spin.setValue(self.preprocessor.baseline_lam)
        baseline_params_layout.addRow("Lambda:", self.baseline_lam_spin)

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

        preproc_steps_layout.addWidget(self.baseline_params_widget)

        # ==== COSMIC RAY REMOVAL ====
        self.cb_cosmic = QCheckBox("Remove Cosmic Rays (Modified Z-scores)")
        self.cb_cosmic.setChecked(self.preprocessor.remove_cosmic_rays)
        self.cb_cosmic.stateChanged.connect(self._toggle_cosmic_params)
        preproc_steps_layout.addWidget(self.cb_cosmic)

        self.cosmic_params_widget = QWidget()
        cosmic_params_layout = QFormLayout(self.cosmic_params_widget)
        cosmic_params_layout.setContentsMargins(20, 0, 0, 0)

        self.cosmic_threshold_spin = QDoubleSpinBox()
        self.cosmic_threshold_spin.setRange(1.0, 50.0)
        self.cosmic_threshold_spin.setSingleStep(0.5)
        self.cosmic_threshold_spin.setValue(self.preprocessor.cosmic_threshold)
        cosmic_params_layout.addRow("Threshold:", self.cosmic_threshold_spin)

        preproc_steps_layout.addWidget(self.cosmic_params_widget)

        # ==== SMOOTHING ====
        self.cb_smooth = QCheckBox("Smoothing (Savitzky-Golay)")
        self.cb_smooth.setChecked(self.preprocessor.smoothing)
        self.cb_smooth.stateChanged.connect(self._toggle_smooth_params)
        preproc_steps_layout.addWidget(self.cb_smooth)

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
        smooth_params_layout.addRow("Poly Order:", self.smooth_polyorder_spin)

        preproc_steps_layout.addWidget(self.smooth_params_widget)

        # ==== NORMALIZATION ====
        self.cb_norm = QCheckBox("Normalization")
        self.cb_norm.setChecked(self.preprocessor.normalization)
        self.cb_norm.stateChanged.connect(self._toggle_norm_params)
        preproc_steps_layout.addWidget(self.cb_norm)

        self.norm_params_widget = QWidget()
        norm_params_layout = QFormLayout(self.norm_params_widget)
        norm_params_layout.setContentsMargins(20, 0, 0, 0)

        self.norm_type_combo = QComboBox()
        self.norm_type_combo.addItems(["by_max", "by_area"])
        self.norm_type_combo.setCurrentText(self.preprocessor.normalization_type)
        norm_params_layout.addRow("Type:", self.norm_type_combo)

        preproc_steps_layout.addWidget(self.norm_params_widget)

        preproc_layout.addWidget(self.preproc_steps_widget)

        scroll_layout.addWidget(preproc_group)

        # Config buttons
        config_group = QGroupBox("Configuration")
        config_layout = QVBoxLayout(config_group)

        config_buttons_layout = QHBoxLayout()
        self.btn_load_config = QPushButton("Load Config")
        self.btn_load_config.clicked.connect(self._load_config)
        config_buttons_layout.addWidget(self.btn_load_config)

        self.btn_save_config = QPushButton("Save Config")
        self.btn_save_config.clicked.connect(self._save_config)
        config_buttons_layout.addWidget(self.btn_save_config)

        config_layout.addLayout(config_buttons_layout)

        # Re-process button
        self.btn_reprocess = QPushButton("Re-process Data")
        self.btn_reprocess.setStyleSheet("font-weight: bold; padding: 8px;")
        self.btn_reprocess.clicked.connect(self._on_reprocess)
        config_layout.addWidget(self.btn_reprocess)

        scroll_layout.addWidget(config_group)

        scroll_layout.addStretch()
        scroll_area.setWidget(scroll_widget)
        layout.addWidget(scroll_area)

        # Initialize visibility
        self._toggle_cropping_params()
        self._toggle_baseline_params()
        self._toggle_cosmic_params()
        self._toggle_norm_params()
        self._toggle_smooth_params()

    def set_metadata_keys(self, keys: list):
        """Set available metadata keys for averaging."""
        self._metadata_keys = keys

        # Update averaging combo
        current = self.avg_by_combo.currentText()
        self.avg_by_combo.blockSignals(True)
        self.avg_by_combo.clear()
        self.avg_by_combo.addItem("None")
        for key in sorted(keys):
            self.avg_by_combo.addItem(key)
        if current in keys:
            self.avg_by_combo.setCurrentText(current)
        self.avg_by_combo.blockSignals(False)

    def _toggle_averaging_params(self):
        self.avg_params_widget.setVisible(self.cb_averaging.isChecked())

    def _toggle_master_preprocessing(self):
        """Enable/disable all preprocessing steps."""
        enabled = self.cb_master_enable.isChecked()
        self.preproc_steps_widget.setEnabled(enabled)

    def _toggle_cropping_params(self):
        self.crop_params_widget.setVisible(self.cb_cropping.isChecked())

    def _toggle_baseline_params(self):
        self.baseline_params_widget.setVisible(self.cb_baseline.isChecked())

    def _toggle_cosmic_params(self):
        self.cosmic_params_widget.setVisible(self.cb_cosmic.isChecked())

    def _toggle_norm_params(self):
        self.norm_params_widget.setVisible(self.cb_norm.isChecked())

    def _toggle_smooth_params(self):
        self.smooth_params_widget.setVisible(self.cb_smooth.isChecked())

    def _on_reprocess(self):
        """Handle Re-process button click."""
        self.update_preprocessor()
        self.preprocessing_changed.emit()

    def get_averaging_config(self) -> dict:
        """Get averaging configuration."""
        return {
            'enabled': self.cb_averaging.isChecked(),
            'by': self.avg_by_combo.currentText() if self.avg_by_combo.currentText() != "None" else None,
            'timing': self.avg_timing_combo.currentText()
        }

    def is_preprocessing_enabled(self) -> bool:
        """Check if master preprocessing is enabled."""
        return self.cb_master_enable.isChecked()

    def update_preprocessor(self):
        """Update preprocessor from UI values."""
        # If master is disabled, turn off all preprocessing
        if not self.cb_master_enable.isChecked():
            self.preprocessor.cropping = False
            self.preprocessor.baseline_correction = False
            self.preprocessor.remove_cosmic_rays = False
            self.preprocessor.normalization = False
            self.preprocessor.smoothing = False
            return

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

    def _load_config(self):
        """Load preprocessing config from file."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Config", "", "Config Files (*.yaml *.yml *.json)"
        )
        if filename:
            with open(filename, 'r') as f:
                if filename.endswith(('.yaml', '.yml')):
                    config = yaml.safe_load(f)
                else:
                    config = json.load(f)

            self.preprocessor = SpectrumPreprocessorWrapper.from_dict(config)
            self._update_ui_from_preprocessor()
            print(f"Configuration loaded from {filename}")

    def _save_config(self):
        """Save preprocessing config to file."""
        self.update_preprocessor()
        config = self.preprocessor.to_dict()

        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Config", "", "YAML Files (*.yaml);;JSON Files (*.json)"
        )
        if filename:
            with open(filename, 'w') as f:
                if filename.endswith('.json'):
                    json.dump(config, f, indent=4)
                else:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            print(f"Configuration saved to {filename}")

    def _update_ui_from_preprocessor(self):
        """Update UI elements from preprocessor values."""
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
