"""Color mapping controls tab widget."""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QGroupBox, QComboBox
)
from PyQt5.QtCore import pyqtSignal


class ColorMappingTab(QWidget):
    """Tab widget for color mapping controls."""

    # Signals
    color_option_changed = pyqtSignal()  # Emitted when color-by or colormap changes

    def __init__(self, parent=None):
        super().__init__(parent)

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        group = QGroupBox("Color Mapping")
        group_layout = QVBoxLayout(group)
        group_layout.setSpacing(8)

        # Color by selection
        group_layout.addWidget(QLabel("Color By:"))
        self.color_combo = QComboBox()
        self.color_combo.currentTextChanged.connect(self._on_changed)
        group_layout.addWidget(self.color_combo)

        # Colormap selection
        group_layout.addWidget(QLabel("Color Map:"))
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(["tab10", "turbo", "hsv", "cool", "viridis", "plasma"])
        self.cmap_combo.setCurrentText("tab10")
        self.cmap_combo.currentTextChanged.connect(self._on_changed)
        group_layout.addWidget(self.cmap_combo)

        layout.addWidget(group)
        layout.addStretch(1)

    def _on_changed(self, *_):
        """Handle any change."""
        self.color_option_changed.emit()

    def populate_color_options(self, metadata_keys: list):
        """Populate color-by combo with metadata keys."""
        self.color_combo.blockSignals(True)
        current = self.color_combo.currentText()
        self.color_combo.clear()
        self.color_combo.addItem("None")
        for key in sorted(metadata_keys):
            self.color_combo.addItem(key)
        # Restore selection if possible
        idx = self.color_combo.findText(current)
        if idx >= 0:
            self.color_combo.setCurrentIndex(idx)
        self.color_combo.blockSignals(False)

    def add_color_option(self, option: str):
        """Add a color option (e.g., 'cluster', 'outlier')."""
        if self.color_combo.findText(option) == -1:
            self.color_combo.addItem(option)

    def set_color_option(self, option: str):
        """Set the current color-by option."""
        idx = self.color_combo.findText(option)
        if idx >= 0:
            self.color_combo.setCurrentIndex(idx)

    def get_color_attr(self) -> str:
        """Get current color-by attribute."""
        return self.color_combo.currentText()

    def get_cmap(self) -> str:
        """Get current colormap name."""
        return self.cmap_combo.currentText()
