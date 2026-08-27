"""Filter tab widget combining metadata filters and hidden spectra management."""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QGroupBox, QScrollArea, QFrame
)
from PyQt5.QtCore import pyqtSignal
from typing import List, Dict, Any, Set

from .hidden_spectra_widget import HiddenSpectraWidget


class FilterTab(QWidget):
    """Tab widget for filtering spectra by metadata and managing hidden spectra."""

    # Signals
    filter_changed = pyqtSignal()  # Emitted when filters change
    restore_requested = pyqtSignal(int)  # Emits set index to restore
    restore_all_requested = pyqtSignal()  # Restore all hidden spectra

    def __init__(self, parent=None):
        super().__init__(parent)

        self.filters: List[Dict[str, Any]] = []
        self._metadata_keys: List[str] = []

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # === METADATA FILTERS GROUP ===
        filter_group = QGroupBox("Metadata Filters")
        filter_layout = QVBoxLayout(filter_group)
        filter_layout.setSpacing(6)

        # Filter rows container
        self.filter_container = QWidget()
        self.filter_rows_layout = QVBoxLayout(self.filter_container)
        self.filter_rows_layout.setContentsMargins(0, 0, 0, 0)
        self.filter_rows_layout.setSpacing(4)
        filter_layout.addWidget(self.filter_container)

        # Add filter button
        btn_row = QHBoxLayout()
        self.btn_add_filter = QPushButton("+ Add Filter")
        self.btn_add_filter.clicked.connect(self.add_filter_row)
        btn_row.addWidget(self.btn_add_filter)
        btn_row.addStretch()

        self.btn_clear_filters = QPushButton("Clear All Filters")
        self.btn_clear_filters.clicked.connect(self._clear_all_filters)
        btn_row.addWidget(self.btn_clear_filters)

        filter_layout.addLayout(btn_row)
        layout.addWidget(filter_group)

        # === HIDDEN SPECTRA GROUP ===
        self.hidden_spectra_widget = HiddenSpectraWidget()
        self.hidden_spectra_widget.restore_requested.connect(self.restore_requested.emit)
        self.hidden_spectra_widget.restore_all_requested.connect(self.restore_all_requested.emit)
        layout.addWidget(self.hidden_spectra_widget)

        layout.addStretch()

    def set_metadata_keys(self, keys: List[str]):
        """Set available metadata keys for filtering."""
        self._metadata_keys = ["None"] + sorted(keys)

        # Update existing filter combos
        for f in self.filters:
            combo = f.get('attr_combo')
            if combo:
                current = combo.currentText()
                combo.blockSignals(True)
                combo.clear()
                combo.addItems(self._metadata_keys)
                if current in self._metadata_keys:
                    combo.setCurrentText(current)
                combo.blockSignals(False)

    def add_filter_row(self):
        """Add a new filter row."""
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(4)

        # Attribute combo
        attr_combo = QComboBox()
        attr_combo.addItems(self._metadata_keys)
        attr_combo.setMinimumWidth(100)
        row_layout.addWidget(attr_combo)

        # Value combo
        value_combo = QComboBox()
        value_combo.setMinimumWidth(100)
        row_layout.addWidget(value_combo)

        # Remove button
        btn_remove = QPushButton("×")
        btn_remove.setFixedWidth(24)
        btn_remove.setStyleSheet("font-weight: bold;")
        row_layout.addWidget(btn_remove)

        # Store filter info
        filter_info = {
            'widget': row_widget,
            'attr_combo': attr_combo,
            'value_combo': value_combo,
            'attr': None
        }
        self.filters.append(filter_info)

        # Connect signals
        attr_combo.currentTextChanged.connect(
            lambda text, f=filter_info: self._on_attr_changed(f, text)
        )
        value_combo.currentTextChanged.connect(
            lambda text, f=filter_info: self._on_value_changed(f, text)
        )
        btn_remove.clicked.connect(
            lambda checked, f=filter_info: self._remove_filter(f)
        )

        self.filter_rows_layout.addWidget(row_widget)

    def _on_attr_changed(self, filter_info: Dict, attr: str):
        """Handle attribute selection change."""
        filter_info['attr'] = attr if attr != "None" else None
        self.filter_changed.emit()

    def _on_value_changed(self, filter_info: Dict, value: str):
        """Handle value selection change."""
        filter_info['value'] = value if value else None
        self.filter_changed.emit()

    def _remove_filter(self, filter_info: Dict):
        """Remove a filter row."""
        if filter_info in self.filters:
            self.filters.remove(filter_info)
            filter_info['widget'].deleteLater()
            self.filter_changed.emit()

    def _clear_all_filters(self):
        """Clear all filter rows."""
        for f in self.filters[:]:
            f['widget'].deleteLater()
        self.filters.clear()
        self.filter_changed.emit()

    def populate_filter_values(self, filter_info: Dict, values: List[str]):
        """Populate value combo for a filter."""
        combo = filter_info.get('value_combo')
        if combo:
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems([""] + values)
            if current in values:
                combo.setCurrentText(current)
            combo.blockSignals(False)

    def get_active_filters(self) -> List[tuple]:
        """Get list of active (attr, value) filter pairs."""
        active = []
        for f in self.filters:
            attr = f.get('attr')
            value = f.get('value')
            if attr and attr != "None" and value:
                active.append((attr, value))
        return active

    # === Hidden Spectra Widget Passthrough Methods ===

    def add_hidden_set(self, indices: Set[int], label: str = None):
        """Add a new set of hidden spectra."""
        self.hidden_spectra_widget.add_hidden_set(indices, label)

    def remove_hidden_set(self, set_index: int) -> Set[int]:
        """Remove and return a hidden set by index."""
        return self.hidden_spectra_widget.remove_hidden_set(set_index)

    def clear_hidden_sets(self) -> Set[int]:
        """Clear all hidden sets and return all hidden indices."""
        return self.hidden_spectra_widget.clear_all()

    def get_all_hidden_indices(self) -> Set[int]:
        """Return union of all hidden indices."""
        return self.hidden_spectra_widget.get_all_hidden_indices()

    def has_hidden_spectra(self) -> bool:
        """Check if any spectra are hidden."""
        return self.hidden_spectra_widget.has_hidden_spectra()

    def reset_hidden(self):
        """Reset hidden spectra state."""
        self.hidden_spectra_widget.reset()
