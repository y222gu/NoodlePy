"""Experiment loader widget for selecting data folders."""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QGroupBox, QListWidget, QListWidgetItem, QAbstractItemView,
    QListView, QTreeView
)
from PyQt5.QtCore import pyqtSignal, Qt
from typing import List, Dict, Any


class ExperimentLoaderWidget(QWidget):
    """Widget for loading and managing multiple experiment data folders."""

    # Signals
    experiment_loaded = pyqtSignal(str)  # Legacy signal for backward compatibility
    experiment_added = pyqtSignal(str)  # Emits folder path when added
    experiment_removed = pyqtSignal(int)  # Emits experiment_id when removed
    experiments_cleared = pyqtSignal()  # All experiments removed

    def __init__(self, parent=None):
        super().__init__(parent)

        self.current_path: str = ""
        self._experiments: Dict[int, Dict[str, Any]] = {}  # {experiment_id: {'path', 'folder_name', 'spectrum_count'}}
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        group = QGroupBox("Experiments")
        group_layout = QVBoxLayout(group)
        group_layout.setSpacing(8)

        # Button row
        btn_row = QHBoxLayout()

        self.btn_add = QPushButton("Add Experiment...")
        self.btn_add.clicked.connect(self._on_add_clicked)
        btn_row.addWidget(self.btn_add)

        self.btn_remove = QPushButton("Remove Selected")
        self.btn_remove.clicked.connect(self._on_remove_clicked)
        self.btn_remove.setEnabled(False)
        btn_row.addWidget(self.btn_remove)

        self.btn_clear = QPushButton("Clear All")
        self.btn_clear.clicked.connect(self._on_clear_clicked)
        self.btn_clear.setEnabled(False)
        btn_row.addWidget(self.btn_clear)

        group_layout.addLayout(btn_row)

        # Experiment list
        self.experiment_list = QListWidget()
        self.experiment_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.experiment_list.setMaximumHeight(120)
        self.experiment_list.setStyleSheet("""
            QListWidget {
                background-color: #2d2d2d;
                border: 1px solid #444444;
                border-radius: 4px;
                font-size: 11px;
            }
            QListWidget::item {
                padding: 4px 8px;
                border-bottom: 1px solid #3a3a3a;
            }
            QListWidget::item:selected {
                background-color: #0d6efd;
            }
            QListWidget::item:hover {
                background-color: #3a3a3a;
            }
        """)
        self.experiment_list.itemSelectionChanged.connect(self._on_selection_changed)
        group_layout.addWidget(self.experiment_list)

        # Totals info row
        info_row = QHBoxLayout()

        self.experiment_count_label = QLabel("Experiments: 0")
        self.experiment_count_label.setStyleSheet("font-size: 10px;")
        info_row.addWidget(self.experiment_count_label)

        self.sample_count_label = QLabel("Samples: -")
        self.sample_count_label.setStyleSheet("font-size: 10px;")
        info_row.addWidget(self.sample_count_label)

        self.spectrum_count_label = QLabel("Spectra: -")
        self.spectrum_count_label.setStyleSheet("font-size: 10px;")
        info_row.addWidget(self.spectrum_count_label)

        info_row.addStretch()
        group_layout.addLayout(info_row)

        layout.addWidget(group)

    def _on_add_clicked(self):
        """Handle add button click - allows selecting multiple folders."""
        dialog = QFileDialog(self, "Select Experiment Folder(s)")
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        dialog.setOption(QFileDialog.ShowDirsOnly, True)

        # Enable multi-selection in the file dialog
        file_view = dialog.findChild(QListView, 'listView')
        if file_view:
            file_view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        tree_view = dialog.findChild(QTreeView, 'treeView')
        if tree_view:
            tree_view.setSelectionMode(QAbstractItemView.ExtendedSelection)

        if self.current_path:
            dialog.setDirectory(self.current_path)

        if dialog.exec_() == QFileDialog.Accepted:
            folders = dialog.selectedFiles()
            for folder in folders:
                self.current_path = folder
                # Emit the new signal for adding experiments
                self.experiment_added.emit(folder)
                # Also emit legacy signal for backward compatibility
                self.experiment_loaded.emit(folder)

    def _on_remove_clicked(self):
        """Handle remove button click."""
        selected_items = self.experiment_list.selectedItems()
        if not selected_items:
            return

        item = selected_items[0]
        experiment_id = item.data(Qt.UserRole)
        if experiment_id is not None:
            self.experiment_removed.emit(experiment_id)

    def _on_clear_clicked(self):
        """Handle clear all button click."""
        self.experiments_cleared.emit()

    def _on_selection_changed(self):
        """Handle list selection change."""
        has_selection = len(self.experiment_list.selectedItems()) > 0
        self.btn_remove.setEnabled(has_selection)

    def add_experiment(self, experiment_id: int, folder_name: str, spectrum_count: int, path: str):
        """Add an experiment to the list."""
        self._experiments[experiment_id] = {
            'path': path,
            'folder_name': folder_name,
            'spectrum_count': spectrum_count
        }

        item = QListWidgetItem(f"{folder_name} ({spectrum_count} spectra)")
        item.setData(Qt.UserRole, experiment_id)
        item.setToolTip(path)
        self.experiment_list.addItem(item)

        self._update_buttons()
        self._update_experiment_count()

    def remove_experiment(self, experiment_id: int):
        """Remove an experiment from the list."""
        if experiment_id in self._experiments:
            del self._experiments[experiment_id]

        # Find and remove the list item
        for i in range(self.experiment_list.count()):
            item = self.experiment_list.item(i)
            if item.data(Qt.UserRole) == experiment_id:
                self.experiment_list.takeItem(i)
                break

        self._update_buttons()
        self._update_experiment_count()

    def clear_experiments(self):
        """Clear all experiments from the list."""
        self._experiments.clear()
        self.experiment_list.clear()
        self._update_buttons()
        self._update_experiment_count()
        self.sample_count_label.setText("Samples: -")
        self.spectrum_count_label.setText("Spectra: -")

    def _update_buttons(self):
        """Update button states based on current experiments."""
        has_experiments = len(self._experiments) > 0
        self.btn_clear.setEnabled(has_experiments)
        self.btn_remove.setEnabled(
            has_experiments and len(self.experiment_list.selectedItems()) > 0
        )

    def _update_experiment_count(self):
        """Update the experiment count label."""
        count = len(self._experiments)
        self.experiment_count_label.setText(f"Experiments: {count}")

    def set_info(self, sample_count: int, spectrum_count: int):
        """Update info labels after data is loaded."""
        self.sample_count_label.setText(f"Samples: {sample_count}")
        self.spectrum_count_label.setText(f"Spectra: {spectrum_count}")

    def set_path(self, path: str):
        """Set the current path programmatically."""
        self.current_path = path

    def get_experiment_count(self) -> int:
        """Return number of loaded experiments."""
        return len(self._experiments)

    def get_experiments(self) -> List[Dict[str, Any]]:
        """Return list of experiment info dicts."""
        return [
            {'id': exp_id, **info}
            for exp_id, info in self._experiments.items()
        ]
