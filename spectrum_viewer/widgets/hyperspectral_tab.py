"""Hyperspectral mapping tab widget."""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox,
    QComboBox, QSlider, QDoubleSpinBox, QScrollArea, QFrame
)
from PyQt5.QtCore import pyqtSignal, Qt

from ..plots.hyperspectral_map import HyperspectralMapWidget


class HyperspectralTab(QWidget):
    """Tab widget for hyperspectral mapping visualization."""

    # Signals
    point_selected = pyqtSignal(int)  # Emits selected spectrum index
    wavenumber_changed = pyqtSignal(float)  # Emits current wavenumber

    def __init__(self, parent=None):
        super().__init__(parent)

        self.wavenumber_min = 600.0
        self.wavenumber_max = 1800.0
        self.current_wavenumber = 1000.0

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # Controls group
        controls_group = QGroupBox("Hyperspectral Maps")
        controls_layout = QVBoxLayout(controls_group)
        controls_layout.setSpacing(8)

        # Wavenumber slider
        wn_row = QHBoxLayout()
        wn_row.addWidget(QLabel("Wavenumber:"))

        self.wn_slider = QSlider(Qt.Horizontal)
        self.wn_slider.setRange(0, 1000)  # Will be scaled to actual range
        self.wn_slider.setValue(500)
        self.wn_slider.valueChanged.connect(self._on_slider_changed)
        wn_row.addWidget(self.wn_slider, stretch=1)

        self.wn_spinbox = QDoubleSpinBox()
        self.wn_spinbox.setRange(self.wavenumber_min, self.wavenumber_max)
        self.wn_spinbox.setValue(self.current_wavenumber)
        self.wn_spinbox.setSuffix(" cm^-1")
        self.wn_spinbox.setDecimals(1)
        self.wn_spinbox.valueChanged.connect(self._on_spinbox_changed)
        wn_row.addWidget(self.wn_spinbox)

        controls_layout.addLayout(wn_row)

        # Color By selection
        color_row = QHBoxLayout()
        color_row.addWidget(QLabel("Color By:"))

        self.color_by_combo = QComboBox()
        self.color_by_combo.addItems(['Intensity', 'Scatter Plot Colors'])
        self.color_by_combo.currentTextChanged.connect(self._on_color_by_changed)
        color_row.addWidget(self.color_by_combo)

        color_row.addStretch()
        controls_layout.addLayout(color_row)

        # Colormap selection
        cmap_row = QHBoxLayout()
        cmap_row.addWidget(QLabel("Colormap:"))

        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(['turbo', 'viridis', 'plasma', 'inferno', 'magma', 'hot', 'cool'])
        self.cmap_combo.setCurrentText('turbo')
        self.cmap_combo.currentTextChanged.connect(self._on_cmap_changed)
        cmap_row.addWidget(self.cmap_combo)

        cmap_row.addStretch()
        controls_layout.addLayout(cmap_row)

        layout.addWidget(controls_group)

        # Hyperspectral maps container
        self.map_widget = HyperspectralMapWidget()
        self.map_widget.point_clicked.connect(self._on_point_clicked)
        layout.addWidget(self.map_widget, stretch=1)

    def set_wavenumber_range(self, wn_min: float, wn_max: float):
        """Set the wavenumber range for the slider."""
        self.wavenumber_min = wn_min
        self.wavenumber_max = wn_max

        self.wn_spinbox.setRange(wn_min, wn_max)

        # Set initial value to middle of range
        mid = (wn_min + wn_max) / 2
        self.current_wavenumber = mid
        self.wn_spinbox.setValue(mid)
        self._update_slider_from_spinbox()

    def set_samples(self, samples: dict, data_manager):
        """
        Set up sample maps.

        Args:
            samples: Dictionary mapping sample ID to list of spectrum indices
            data_manager: ViewerDataManager instance
        """
        self.map_widget.set_samples(samples, data_manager)
        self._update_all_maps()

    def update_point_colors(self, scatter_colors, index_to_pos):
        """Update scatter plot colors for maps."""
        self.map_widget.update_point_colors(scatter_colors, index_to_pos)

    def update_selection(self, selected_indices, scatter_colors, index_to_pos):
        """Update lasso selection on maps."""
        self.map_widget.set_selection(selected_indices, scatter_colors, index_to_pos)

    def clear_selection(self):
        """Clear lasso selection on maps."""
        self.map_widget.clear_selection()

    def _on_slider_changed(self, value):
        """Handle slider value change."""
        # Map slider value (0-1000) to wavenumber range
        fraction = value / 1000.0
        wn = self.wavenumber_min + fraction * (self.wavenumber_max - self.wavenumber_min)

        self.wn_spinbox.blockSignals(True)
        self.wn_spinbox.setValue(wn)
        self.wn_spinbox.blockSignals(False)

        self.current_wavenumber = wn
        self.wavenumber_changed.emit(wn)
        self._update_all_maps()

    def _on_spinbox_changed(self, value):
        """Handle spinbox value change."""
        self.current_wavenumber = value
        self._update_slider_from_spinbox()
        self.wavenumber_changed.emit(value)
        self._update_all_maps()

    def _update_slider_from_spinbox(self):
        """Update slider position from spinbox value."""
        if self.wavenumber_max == self.wavenumber_min:
            fraction = 0.5
        else:
            fraction = (self.current_wavenumber - self.wavenumber_min) / (self.wavenumber_max - self.wavenumber_min)

        self.wn_slider.blockSignals(True)
        self.wn_slider.setValue(int(fraction * 1000))
        self.wn_slider.blockSignals(False)

    def _on_color_by_changed(self, text):
        """Handle color-by mode change."""
        mode = 'scatter' if text == 'Scatter Plot Colors' else 'intensity'
        # Disable colormap combo when using scatter colors (colors come from Color Mapping tab)
        self.cmap_combo.setEnabled(mode == 'intensity')
        self.map_widget.set_color_mode(mode)

    def _on_cmap_changed(self, cmap_name):
        """Handle colormap change."""
        self.map_widget.set_colormap(cmap_name)
        self._update_all_maps()

    def _update_all_maps(self):
        """Update all sample maps with current wavenumber."""
        self.map_widget.update_wavenumber(self.current_wavenumber)

    def _on_point_clicked(self, sample_id, spectrum_idx):
        """Handle point click on a map."""
        self.point_selected.emit(spectrum_idx)

    def highlight_point(self, spectrum_idx: int):
        """Highlight a specific spectrum point on hyperspectral maps."""
        self.map_widget.highlight_point(spectrum_idx)

    def clear_highlight(self):
        """Clear highlight from hyperspectral maps."""
        self.map_widget.clear_highlight()

    def refresh(self):
        """Refresh all maps."""
        self.map_widget.refresh_all()

    def clear(self):
        """Clear all maps."""
        self.map_widget.clear()
