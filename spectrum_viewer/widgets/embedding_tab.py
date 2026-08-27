"""Embedding/dimensionality reduction controls tab widget."""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QGroupBox, QComboBox, QSpinBox
)
from PyQt5.QtCore import pyqtSignal


class EmbeddingTab(QWidget):
    """Tab widget for dimensionality reduction controls."""

    # Signals
    method_changed = pyqtSignal(str)  # Emits new method name
    dim_changed = pyqtSignal(int)  # Emits 2 or 3
    pca_loadings_changed = pyqtSignal(int)  # Emits number of PCs for loadings

    def __init__(self, parent=None):
        super().__init__(parent)

        self.current_method = "PCA"
        self.current_dim = 2
        self.n_pcs_for_loading = 5

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        group = QGroupBox("Dim Reduction Options")
        group_layout = QVBoxLayout(group)
        group_layout.setSpacing(8)

        # Dimensionality selection
        group_layout.addWidget(QLabel("Dimensionality:"))
        self.combo_dim = QComboBox()
        self.combo_dim.addItems(["2D", "3D"])
        self.combo_dim.currentTextChanged.connect(self._on_dim_changed)
        group_layout.addWidget(self.combo_dim)

        # Method selection
        group_layout.addWidget(QLabel("Method:"))
        self.combo_method = QComboBox()
        self.combo_method.addItems(["PCA", "T-SNE", "UMAP"])
        self.combo_method.currentTextChanged.connect(self._on_method_changed)
        group_layout.addWidget(self.combo_method)

        # PCA loadings PCs
        group_layout.addWidget(QLabel("PCA Loadings PCs:"))
        self.spin_pca_loadings = QSpinBox()
        self.spin_pca_loadings.setRange(1, 20)
        self.spin_pca_loadings.setValue(self.n_pcs_for_loading)
        self.spin_pca_loadings.setKeyboardTracking(False)
        self.spin_pca_loadings.valueChanged.connect(self._on_pca_loadings_changed)
        group_layout.addWidget(self.spin_pca_loadings)

        layout.addWidget(group)
        layout.addStretch(1)

    def _on_dim_changed(self, text):
        """Handle dimension change."""
        self.current_dim = 2 if text == "2D" else 3
        self.dim_changed.emit(self.current_dim)

    def _on_method_changed(self, text):
        """Handle method change."""
        self.current_method = text
        self.method_changed.emit(text)

    def _on_pca_loadings_changed(self, value):
        """Handle PCA loadings PC count change."""
        self.n_pcs_for_loading = value
        self.pca_loadings_changed.emit(value)

    def get_method(self) -> str:
        """Get current embedding method."""
        return self.current_method

    def get_dim(self) -> int:
        """Get current dimensionality (2 or 3)."""
        return self.current_dim

    def get_n_pcs(self) -> int:
        """Get number of PCs for loadings plot."""
        return self.n_pcs_for_loading
