"""Widget for managing hidden (temporarily removed) spectra."""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QScrollArea, QFrame
)
from PyQt5.QtCore import pyqtSignal, Qt
from typing import List, Set, Dict, Any


class HiddenSpectraWidget(QWidget):
    """Widget for managing sets of hidden spectra with restore functionality."""

    # Signals
    restore_requested = pyqtSignal(int)  # Emits set index to restore
    restore_all_requested = pyqtSignal()  # Restore all hidden spectra

    def __init__(self, parent=None):
        super().__init__(parent)

        self._hidden_sets: List[Dict[str, Any]] = []  # List of {'indices': set, 'label': str}
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.group = QGroupBox("Hidden Spectra")
        group_layout = QVBoxLayout(self.group)
        group_layout.setSpacing(6)

        # Info label
        self.info_label = QLabel("No hidden spectra")
        self.info_label.setStyleSheet("color: #888888; font-size: 10px;")
        group_layout.addWidget(self.info_label)

        # Scroll area for hidden set entries
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setMaximumHeight(120)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                background-color: transparent;
            }
        """)

        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_widget)
        self.scroll_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_layout.setSpacing(4)
        self.scroll_layout.addStretch()

        self.scroll_area.setWidget(self.scroll_widget)
        group_layout.addWidget(self.scroll_area)

        # Restore All button
        self.btn_restore_all = QPushButton("Restore All")
        self.btn_restore_all.clicked.connect(self._on_restore_all)
        self.btn_restore_all.setEnabled(False)
        self.btn_restore_all.setStyleSheet("font-size: 10px;")
        group_layout.addWidget(self.btn_restore_all)

        layout.addWidget(self.group)

        # Initially hide the group when empty
        self.group.setVisible(False)

    def add_hidden_set(self, indices: Set[int], label: str = None):
        """Add a new set of hidden spectra."""
        set_index = len(self._hidden_sets)

        if label is None:
            label = f"Set {set_index + 1}: {len(indices)} spectra"

        self._hidden_sets.append({
            'indices': indices.copy(),
            'label': label
        })

        # Create UI entry
        entry_widget = self._create_entry_widget(set_index, label, len(indices))

        # Insert before the stretch
        self.scroll_layout.insertWidget(self.scroll_layout.count() - 1, entry_widget)

        self._update_ui_state()

    def _create_entry_widget(self, set_index: int, label: str, count: int) -> QWidget:
        """Create a widget for a single hidden set entry."""
        entry = QWidget()
        entry.setObjectName(f"hidden_set_{set_index}")
        entry_layout = QHBoxLayout(entry)
        entry_layout.setContentsMargins(4, 2, 4, 2)
        entry_layout.setSpacing(8)

        # Label
        lbl = QLabel(f"{label}")
        lbl.setStyleSheet("font-size: 10px; color: #cccccc;")
        entry_layout.addWidget(lbl, stretch=1)

        # Restore button
        btn_restore = QPushButton("Restore")
        btn_restore.setFixedWidth(60)
        btn_restore.setStyleSheet("font-size: 10px; padding: 2px 6px;")
        btn_restore.clicked.connect(lambda checked, idx=set_index: self._on_restore(idx))
        entry_layout.addWidget(btn_restore)

        entry.setStyleSheet("""
            QWidget {
                background-color: #2a2a2a;
                border-radius: 4px;
            }
        """)

        return entry

    def _on_restore(self, set_index: int):
        """Handle restore button click for a specific set."""
        self.restore_requested.emit(set_index)

    def _on_restore_all(self):
        """Handle restore all button click."""
        self.restore_all_requested.emit()

    def remove_hidden_set(self, set_index: int) -> Set[int]:
        """Remove and return a hidden set by index."""
        if set_index < 0 or set_index >= len(self._hidden_sets):
            return set()

        removed = self._hidden_sets.pop(set_index)

        # Rebuild UI
        self._rebuild_entries()
        self._update_ui_state()

        return removed['indices']

    def clear_all(self) -> Set[int]:
        """Clear all hidden sets and return all hidden indices."""
        all_indices = self.get_all_hidden_indices()
        self._hidden_sets.clear()
        self._rebuild_entries()
        self._update_ui_state()
        return all_indices

    def _rebuild_entries(self):
        """Rebuild all entry widgets after a change."""
        # Remove all entries except the stretch
        while self.scroll_layout.count() > 1:
            item = self.scroll_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Recreate entries
        for i, hidden_set in enumerate(self._hidden_sets):
            entry_widget = self._create_entry_widget(
                i, hidden_set['label'], len(hidden_set['indices'])
            )
            self.scroll_layout.insertWidget(i, entry_widget)

    def _update_ui_state(self):
        """Update UI state based on current hidden sets."""
        total_hidden = sum(len(s['indices']) for s in self._hidden_sets)
        num_sets = len(self._hidden_sets)

        if total_hidden == 0:
            self.info_label.setText("No hidden spectra")
            self.btn_restore_all.setEnabled(False)
            self.group.setVisible(False)
        else:
            self.info_label.setText(f"Total: {total_hidden} spectra in {num_sets} set(s)")
            self.btn_restore_all.setEnabled(True)
            self.group.setVisible(True)

    def get_all_hidden_indices(self) -> Set[int]:
        """Return union of all hidden indices."""
        all_indices = set()
        for hidden_set in self._hidden_sets:
            all_indices.update(hidden_set['indices'])
        return all_indices

    def get_hidden_set_count(self) -> int:
        """Return number of hidden sets."""
        return len(self._hidden_sets)

    def has_hidden_spectra(self) -> bool:
        """Check if any spectra are hidden."""
        return len(self._hidden_sets) > 0

    def reset(self):
        """Reset to initial state (clears all without returning indices)."""
        self._hidden_sets.clear()
        self._rebuild_entries()
        self._update_ui_state()
