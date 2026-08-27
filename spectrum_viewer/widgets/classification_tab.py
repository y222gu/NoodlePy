"""Classification tab widget for training MLP models."""

import json
import numpy as np
from typing import Optional, List, Dict, Any
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QComboBox, QSpinBox, QDoubleSpinBox, QPushButton, QGroupBox,
    QLineEdit, QProgressBar, QTextEdit, QScrollArea,
    QFrame, QFileDialog, QSplitter
)
from PyQt5.QtCore import pyqtSignal, Qt, QThread, pyqtSlot
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

from ..utils.dark_theme import style_dark_axes


class MLPClassifier(nn.Module):
    """Configurable MLP classifier."""

    def __init__(self, input_dim: int, hidden_layers: List[int], num_classes: int,
                 dropout_rate: float = 0.0, activation: str = 'relu'):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())

            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TrainingWorker(QThread):
    """Worker thread for model training."""

    progress_update = pyqtSignal(int, float, float)
    training_complete = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, model, train_loader, val_loader, criterion, optimizer,
                 num_epochs, device, l1_lambda=0.0):
        super().__init__()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.device = device
        self.l1_lambda = l1_lambda
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def run(self):
        try:
            self.model.to(self.device)
            train_losses = []
            val_losses = []

            for epoch in range(self.num_epochs):
                if self._stop_requested:
                    break

                # Training
                self.model.train()
                train_loss = 0.0
                for X_batch, y_batch in self.train_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                    self.optimizer.zero_grad()
                    outputs = self.model(X_batch)
                    loss = self.criterion(outputs, y_batch)

                    if self.l1_lambda > 0:
                        l1_norm = sum(p.abs().sum() for p in self.model.parameters())
                        loss = loss + self.l1_lambda * l1_norm

                    loss.backward()
                    self.optimizer.step()
                    train_loss += loss.item()

                train_loss /= len(self.train_loader)
                train_losses.append(train_loss)

                # Validation
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for X_batch, y_batch in self.val_loader:
                        X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                        outputs = self.model(X_batch)
                        loss = self.criterion(outputs, y_batch)
                        val_loss += loss.item()

                val_loss /= len(self.val_loader)
                val_losses.append(val_loss)

                self.progress_update.emit(epoch + 1, train_loss, val_loss)

            # Final evaluation
            self.model.eval()
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for X_batch, y_batch in self.val_loader:
                    X_batch = X_batch.to(self.device)
                    outputs = self.model(X_batch)
                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(y_batch.numpy())

            results = {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'predictions': np.array(all_preds),
                'labels': np.array(all_labels),
                'accuracy': accuracy_score(all_labels, all_preds),
                'confusion_matrix': confusion_matrix(all_labels, all_preds),
                'stopped_early': self._stop_requested
            }

            self.training_complete.emit(results)

        except Exception as e:
            import traceback
            self.error_occurred.emit(f"{str(e)}\n{traceback.format_exc()}")


class ClassificationTab(QWidget):
    """Tab widget for MLP classification training."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self._data_matrix: Optional[np.ndarray] = None
        self._embedding_matrix: Optional[np.ndarray] = None
        self._labels: Optional[np.ndarray] = None
        self._label_encoder: Optional[LabelEncoder] = None
        self._visible_indices: List[int] = []
        self._metadata_keys: List[str] = []
        self._data_objects: List[Any] = []

        self._model: Optional[MLPClassifier] = None
        self._training_worker: Optional[TrainingWorker] = None
        self._last_results: Optional[Dict] = None

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Main splitter
        splitter = QSplitter(Qt.Vertical)

        # === TOP: Settings (scrollable) ===
        settings_widget = QWidget()
        settings_layout = QVBoxLayout(settings_widget)
        settings_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(8)

        # === DATA SOURCE GROUP ===
        data_group = QGroupBox("Data Source")
        data_layout = QGridLayout(data_group)
        data_layout.setSpacing(6)

        data_layout.addWidget(QLabel("Input Data:"), 0, 0)
        self.combo_data_source = QComboBox()
        self.combo_data_source.addItems(['Preprocessed Spectra', 'Dim Reduced Embeddings'])
        data_layout.addWidget(self.combo_data_source, 0, 1)

        data_layout.addWidget(QLabel("Label (Target):"), 1, 0)
        self.combo_label = QComboBox()
        self.combo_label.currentTextChanged.connect(self._on_label_changed)
        data_layout.addWidget(self.combo_label, 1, 1)

        data_layout.addWidget(QLabel("Patient ID Field:"), 2, 0)
        self.combo_patient_id = QComboBox()
        self.combo_patient_id.setToolTip("Field used for patient-level train/val split (prevents data leakage)")
        self.combo_patient_id.currentTextChanged.connect(self._on_patient_id_changed)
        data_layout.addWidget(self.combo_patient_id, 2, 1)

        self.label_info = QLabel("Classes: -")
        self.label_info.setStyleSheet("color: #888888; font-size: 10px;")
        data_layout.addWidget(self.label_info, 3, 0, 1, 2)

        self.patient_info = QLabel("Patients: -")
        self.patient_info.setStyleSheet("color: #888888; font-size: 10px;")
        data_layout.addWidget(self.patient_info, 4, 0, 1, 2)

        self.data_count_label = QLabel("Visible Samples: 0")
        self.data_count_label.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        data_layout.addWidget(self.data_count_label, 5, 0, 1, 2)

        scroll_layout.addWidget(data_group)

        # === MODEL ARCHITECTURE GROUP ===
        arch_group = QGroupBox("Model Architecture")
        arch_layout = QGridLayout(arch_group)
        arch_layout.setSpacing(6)

        arch_layout.addWidget(QLabel("Hidden Layers:"), 0, 0)
        self.edit_layers = QLineEdit("64, 32")
        self.edit_layers.setPlaceholderText("e.g., 128, 64, 32")
        arch_layout.addWidget(self.edit_layers, 0, 1)

        arch_layout.addWidget(QLabel("Activation:"), 1, 0)
        self.combo_activation = QComboBox()
        self.combo_activation.addItems(['relu', 'leaky_relu', 'tanh', 'sigmoid'])
        arch_layout.addWidget(self.combo_activation, 1, 1)

        arch_layout.addWidget(QLabel("Dropout Rate:"), 2, 0)
        self.spin_dropout = QDoubleSpinBox()
        self.spin_dropout.setRange(0.0, 0.9)
        self.spin_dropout.setSingleStep(0.1)
        self.spin_dropout.setValue(0.2)
        arch_layout.addWidget(self.spin_dropout, 2, 1)

        scroll_layout.addWidget(arch_group)

        # === TRAINING SETTINGS GROUP ===
        train_group = QGroupBox("Training Settings")
        train_layout = QGridLayout(train_group)
        train_layout.setSpacing(6)

        train_layout.addWidget(QLabel("Train/Val Split:"), 0, 0)
        self.spin_val_split = QDoubleSpinBox()
        self.spin_val_split.setRange(0.1, 0.5)
        self.spin_val_split.setSingleStep(0.05)
        self.spin_val_split.setValue(0.2)
        train_layout.addWidget(self.spin_val_split, 0, 1)

        train_layout.addWidget(QLabel("Epochs:"), 1, 0)
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 999999)  # No practical upper limit
        self.spin_epochs.setValue(100)
        train_layout.addWidget(self.spin_epochs, 1, 1)

        train_layout.addWidget(QLabel("Batch Size:"), 2, 0)
        self.spin_batch_size = QSpinBox()
        self.spin_batch_size.setRange(1, 512)
        self.spin_batch_size.setValue(32)
        train_layout.addWidget(self.spin_batch_size, 2, 1)

        train_layout.addWidget(QLabel("Learning Rate:"), 3, 0)
        self.spin_lr = QDoubleSpinBox()
        self.spin_lr.setRange(0.00001, 1.0)
        self.spin_lr.setSingleStep(0.001)
        self.spin_lr.setDecimals(5)
        self.spin_lr.setValue(0.001)
        train_layout.addWidget(self.spin_lr, 3, 1)

        scroll_layout.addWidget(train_group)

        # === LOSS & REGULARIZATION GROUP ===
        loss_group = QGroupBox("Loss & Regularization")
        loss_layout = QGridLayout(loss_group)
        loss_layout.setSpacing(6)

        loss_layout.addWidget(QLabel("Loss Function:"), 0, 0)
        self.combo_loss = QComboBox()
        self.combo_loss.addItems(['CrossEntropyLoss', 'NLLLoss'])
        loss_layout.addWidget(self.combo_loss, 0, 1)

        loss_layout.addWidget(QLabel("Optimizer:"), 1, 0)
        self.combo_optimizer = QComboBox()
        self.combo_optimizer.addItems(['Adam', 'SGD', 'AdamW', 'RMSprop'])
        loss_layout.addWidget(self.combo_optimizer, 1, 1)

        loss_layout.addWidget(QLabel("L2 (Weight Decay):"), 2, 0)
        self.spin_weight_decay = QDoubleSpinBox()
        self.spin_weight_decay.setRange(0.0, 1.0)
        self.spin_weight_decay.setSingleStep(0.0001)
        self.spin_weight_decay.setDecimals(5)
        self.spin_weight_decay.setValue(0.0001)
        loss_layout.addWidget(self.spin_weight_decay, 2, 1)

        loss_layout.addWidget(QLabel("L1 Lambda:"), 3, 0)
        self.spin_l1_lambda = QDoubleSpinBox()
        self.spin_l1_lambda.setRange(0.0, 1.0)
        self.spin_l1_lambda.setSingleStep(0.0001)
        self.spin_l1_lambda.setDecimals(5)
        self.spin_l1_lambda.setValue(0.0)
        loss_layout.addWidget(self.spin_l1_lambda, 3, 1)

        scroll_layout.addWidget(loss_group)

        # === BUTTONS ===
        btn_layout = QHBoxLayout()
        self.btn_train = QPushButton("Train Model")
        self.btn_train.clicked.connect(self._on_train_clicked)
        btn_layout.addWidget(self.btn_train)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self._on_stop_clicked)
        self.btn_stop.setEnabled(False)
        btn_layout.addWidget(self.btn_stop)

        scroll_layout.addLayout(btn_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        scroll_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #888888; font-size: 10px;")
        scroll_layout.addWidget(self.status_label)

        # Save buttons
        save_layout = QHBoxLayout()
        self.btn_save_model = QPushButton("Save Model")
        self.btn_save_model.clicked.connect(self._save_model)
        self.btn_save_model.setEnabled(False)
        save_layout.addWidget(self.btn_save_model)

        self.btn_save_results = QPushButton("Save Results")
        self.btn_save_results.clicked.connect(self._save_results)
        self.btn_save_results.setEnabled(False)
        save_layout.addWidget(self.btn_save_results)

        scroll_layout.addLayout(save_layout)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        settings_layout.addWidget(scroll)

        splitter.addWidget(settings_widget)

        # === BOTTOM: Results (plots) ===
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        results_layout.setContentsMargins(0, 0, 0, 0)

        # Results text
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(80)
        self.results_text.setStyleSheet("""
            QTextEdit {
                background-color: #2a2a2a;
                color: #cccccc;
                font-family: monospace;
                font-size: 11px;
            }
        """)
        results_layout.addWidget(self.results_text)

        # Plots: Loss curve and Confusion matrix side by side
        plots_layout = QHBoxLayout()

        # Loss curve plot
        self.loss_fig = Figure(figsize=(4, 3))
        self.loss_fig.patch.set_facecolor('none')
        self.loss_ax = self.loss_fig.add_subplot(111)
        self.loss_ax.set_facecolor('none')
        self.loss_canvas = FigureCanvas(self.loss_fig)
        self.loss_canvas.setStyleSheet("background: transparent;")
        style_dark_axes(self.loss_ax)
        self.loss_ax.set_title("Loss Curve", color='white')
        plots_layout.addWidget(self.loss_canvas)

        # Confusion matrix plot
        self.cm_fig = Figure(figsize=(4, 3))
        self.cm_fig.patch.set_facecolor('none')
        self.cm_ax = self.cm_fig.add_subplot(111)
        self.cm_ax.set_facecolor('none')
        self.cm_canvas = FigureCanvas(self.cm_fig)
        self.cm_canvas.setStyleSheet("background: transparent;")
        style_dark_axes(self.cm_ax)
        self.cm_ax.set_title("Confusion Matrix", color='white')
        plots_layout.addWidget(self.cm_canvas)

        results_layout.addLayout(plots_layout)
        splitter.addWidget(results_widget)

        splitter.setSizes([300, 300])
        layout.addWidget(splitter)

    def set_data(self, data_objects: List[Any], visible_indices: List[int],
                 embedding_matrix: Optional[np.ndarray] = None):
        """Set the data for classification."""
        self._data_objects = data_objects
        self._visible_indices = visible_indices
        self._embedding_matrix = embedding_matrix

        # Update visible sample count display
        n_visible = len(visible_indices) if visible_indices else 0
        n_total = len(data_objects) if data_objects else 0
        self.data_count_label.setText(f"Visible Samples: {n_visible} / {n_total} total")

        if visible_indices and data_objects:
            self._data_matrix = np.array([
                data_objects[i].intensity for i in visible_indices
            ])
        else:
            self._data_matrix = None

        if data_objects:
            keys = set()
            for i in visible_indices:
                keys.update(data_objects[i].metadata.keys())
            self._metadata_keys = sorted(keys)

            # Update label combo
            current_label = self.combo_label.currentText()
            self.combo_label.blockSignals(True)
            self.combo_label.clear()
            self.combo_label.addItems(self._metadata_keys)
            if current_label in self._metadata_keys:
                self.combo_label.setCurrentText(current_label)
            self.combo_label.blockSignals(False)
            self._on_label_changed(self.combo_label.currentText())

            # Update patient ID combo
            current_patient_id = self.combo_patient_id.currentText()
            self.combo_patient_id.blockSignals(True)
            self.combo_patient_id.clear()
            self.combo_patient_id.addItems(self._metadata_keys)
            # Try to auto-select a patient ID field
            if current_patient_id in self._metadata_keys:
                self.combo_patient_id.setCurrentText(current_patient_id)
            else:
                # Auto-detect common patient ID field names
                for field in ['patient', 'patient_id', 'patient_number', 'subject_id', 'subject']:
                    if field in self._metadata_keys:
                        self.combo_patient_id.setCurrentText(field)
                        break
            self.combo_patient_id.blockSignals(False)
            self._on_patient_id_changed(self.combo_patient_id.currentText())

    def _on_label_changed(self, label_key: str):
        """Handle label selection change."""
        if not label_key or not self._data_objects or not self._visible_indices:
            self.label_info.setText("Classes: -")
            return

        labels = []
        for i in self._visible_indices:
            val = self._data_objects[i].metadata.get(label_key)
            if val is not None:
                labels.append(str(val))

        unique_labels = sorted(set(labels))
        n_classes = len(unique_labels)
        n_samples = len(labels)

        self.label_info.setText(
            f"Classes: {n_classes} | Samples: {n_samples} | "
            f"Labels: {', '.join(unique_labels[:5])}{'...' if n_classes > 5 else ''}"
        )

    def _on_patient_id_changed(self, patient_id_key: str):
        """Handle patient ID field selection change."""
        if not patient_id_key or not self._data_objects or not self._visible_indices:
            self.patient_info.setText("Patients: -")
            return

        patient_ids = set()
        for i in self._visible_indices:
            val = self._data_objects[i].metadata.get(patient_id_key)
            if val is not None:
                patient_ids.add(str(val))

        n_patients = len(patient_ids)
        n_samples = len(self._visible_indices)
        avg_per_patient = n_samples / n_patients if n_patients > 0 else 0

        self.patient_info.setText(
            f"Patients: {n_patients} | Avg samples/patient: {avg_per_patient:.1f}"
        )

    def _parse_hidden_layers(self) -> List[int]:
        """Parse hidden layers from text input."""
        text = self.edit_layers.text().strip()
        if not text:
            return [64, 32]
        try:
            layers = [int(x.strip()) for x in text.split(',') if x.strip()]
            return layers if layers else [64, 32]
        except ValueError:
            return [64, 32]

    def _prepare_data(self):
        """Prepare data for training with patient-level information."""
        if not self._visible_indices or not self._data_objects:
            raise ValueError("No data available")

        label_key = self.combo_label.currentText()
        if not label_key:
            raise ValueError("No label selected")

        use_embeddings = self.combo_data_source.currentText() == 'Dim Reduced Embeddings'

        if use_embeddings:
            if self._embedding_matrix is None:
                raise ValueError("No embedding data available")
            X = self._embedding_matrix[np.array(self._visible_indices)]
            valid_mask = ~np.any(np.isnan(X), axis=1)
            X = X[valid_mask]
            valid_indices = list(np.array(self._visible_indices)[valid_mask])
        else:
            X = self._data_matrix
            valid_indices = list(self._visible_indices)

        # Get patient ID field from UI
        patient_id_key = self.combo_patient_id.currentText()
        if not patient_id_key:
            raise ValueError("No patient ID field selected")

        # Get labels and patient IDs for each sample
        labels = []
        patient_ids = []
        for i in valid_indices:
            val = self._data_objects[i].metadata.get(label_key)
            labels.append(str(val) if val is not None else 'unknown')
            # Use selected patient ID field
            patient_id = self._data_objects[i].metadata.get(patient_id_key)
            patient_ids.append(str(patient_id) if patient_id is not None else str(i))

        self._label_encoder = LabelEncoder()
        y = self._label_encoder.fit_transform(labels)

        return X, y, np.array(patient_ids), np.array(labels)

    def _on_train_clicked(self):
        """Start model training."""
        try:
            X, y, patient_ids, labels_str = self._prepare_data()

            n_samples, input_dim = X.shape
            n_classes = len(self._label_encoder.classes_)

            if n_samples < 10:
                self.status_label.setText("Error: Need at least 10 samples")
                return

            val_split = self.spin_val_split.value()

            # Patient-level split to avoid data leakage
            unique_patients = np.unique(patient_ids)
            n_patients = len(unique_patients)

            if n_patients < 2:
                self.status_label.setText("Error: Need at least 2 patients for train/val split")
                return

            # Get label for each patient (use majority label if mixed)
            patient_labels = {}
            for patient in unique_patients:
                patient_mask = patient_ids == patient
                patient_label_values = labels_str[patient_mask]
                # Use most common label for this patient
                unique, counts = np.unique(patient_label_values, return_counts=True)
                patient_labels[patient] = unique[np.argmax(counts)]

            patient_label_array = np.array([patient_labels[p] for p in unique_patients])

            # Split patients, stratified by label if possible
            try:
                train_patients, val_patients = train_test_split(
                    unique_patients, test_size=val_split, random_state=42,
                    stratify=patient_label_array
                )
            except ValueError:
                # Stratification failed (too few samples per class), do random split
                train_patients, val_patients = train_test_split(
                    unique_patients, test_size=val_split, random_state=42
                )

            # Get indices for train and val based on patient membership
            train_mask = np.isin(patient_ids, train_patients)
            val_mask = np.isin(patient_ids, val_patients)

            X_train, y_train = X[train_mask], y[train_mask]
            X_val, y_val = X[val_mask], y[val_mask]

            n_train_patients = len(train_patients)
            n_val_patients = len(val_patients)

            # Validate we have enough samples
            if len(X_train) == 0:
                self.status_label.setText("Error: No training samples after patient split")
                return
            if len(X_val) == 0:
                self.status_label.setText("Error: No validation samples after patient split")
                return

            X_train_t = torch.FloatTensor(X_train)
            y_train_t = torch.LongTensor(y_train)
            X_val_t = torch.FloatTensor(X_val)
            y_val_t = torch.LongTensor(y_val)

            batch_size = min(self.spin_batch_size.value(), len(X_train))
            val_batch_size = min(batch_size, len(X_val))
            train_dataset = TensorDataset(X_train_t, y_train_t)
            val_dataset = TensorDataset(X_val_t, y_val_t)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=val_batch_size)

            hidden_layers = self._parse_hidden_layers()
            self._model = MLPClassifier(
                input_dim=input_dim,
                hidden_layers=hidden_layers,
                num_classes=n_classes,
                dropout_rate=self.spin_dropout.value(),
                activation=self.combo_activation.currentText()
            )

            loss_name = self.combo_loss.currentText()
            if loss_name == 'CrossEntropyLoss':
                criterion = nn.CrossEntropyLoss()
            else:
                criterion = nn.NLLLoss()

            opt_name = self.combo_optimizer.currentText()
            lr = self.spin_lr.value()
            weight_decay = self.spin_weight_decay.value()

            if opt_name == 'Adam':
                optimizer = optim.Adam(self._model.parameters(), lr=lr, weight_decay=weight_decay)
            elif opt_name == 'AdamW':
                optimizer = optim.AdamW(self._model.parameters(), lr=lr, weight_decay=weight_decay)
            elif opt_name == 'SGD':
                optimizer = optim.SGD(self._model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
            else:
                optimizer = optim.RMSprop(self._model.parameters(), lr=lr, weight_decay=weight_decay)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            num_epochs = self.spin_epochs.value()
            self._training_worker = TrainingWorker(
                model=self._model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                num_epochs=num_epochs,
                device=device,
                l1_lambda=self.spin_l1_lambda.value()
            )

            self._training_worker.progress_update.connect(self._on_progress_update)
            self._training_worker.training_complete.connect(self._on_training_complete)
            self._training_worker.error_occurred.connect(self._on_training_error)

            self.btn_train.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.btn_save_model.setEnabled(False)
            self.btn_save_results.setEnabled(False)
            self.progress_bar.setRange(0, num_epochs)
            self.progress_bar.setValue(0)
            self.status_label.setText(
                f"Training on {device}... "
                f"(Train: {len(X_train)} samples/{n_train_patients} patients, "
                f"Val: {len(X_val)} samples/{n_val_patients} patients)"
            )
            self.results_text.clear()

            # Clear plots
            self.loss_fig.clear()
            self.loss_ax = self.loss_fig.add_subplot(111)
            self.loss_ax.set_facecolor('none')
            style_dark_axes(self.loss_ax)
            self.loss_ax.set_title("Loss Curve", color='white')
            self.loss_canvas.draw()

            self.cm_fig.clear()
            self.cm_ax = self.cm_fig.add_subplot(111)
            self.cm_ax.set_facecolor('none')
            style_dark_axes(self.cm_ax)
            self.cm_ax.set_title("Confusion Matrix", color='white')
            self.cm_canvas.draw()

            self._training_worker.start()

        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()

    def _on_stop_clicked(self):
        """Stop training."""
        if self._training_worker:
            self._training_worker.stop()
            self.status_label.setText("Stopping...")

    @pyqtSlot(int, float, float)
    def _on_progress_update(self, epoch: int, train_loss: float, val_loss: float):
        """Handle training progress update."""
        self.progress_bar.setValue(epoch)
        self.status_label.setText(
            f"Epoch {epoch}/{self.spin_epochs.value()} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

    @pyqtSlot(dict)
    def _on_training_complete(self, results: dict):
        """Handle training completion."""
        self.btn_train.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_save_model.setEnabled(True)
        self.btn_save_results.setEnabled(True)

        self._last_results = results
        accuracy = results['accuracy']
        stopped = results.get('stopped_early', False)

        status = "Training stopped early" if stopped else "Training complete"
        self.status_label.setText(f"{status} | Accuracy: {accuracy:.2%}")

        # Display results
        report_text = f"Validation Accuracy: {accuracy:.2%}\n"
        classes = self._label_encoder.classes_
        cm = results['confusion_matrix']

        for i, cls in enumerate(classes):
            if i < len(cm):
                tp = cm[i, i]
                total = cm[i].sum()
                precision = tp / cm[:, i].sum() if cm[:, i].sum() > 0 else 0
                recall = tp / total if total > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                report_text += f"{cls}: P={precision:.2f}, R={recall:.2f}, F1={f1:.2f}\n"

        self.results_text.setText(report_text)

        # Plot loss curves
        self._plot_loss_curves(results['train_losses'], results['val_losses'])

        # Plot confusion matrix
        self._plot_confusion_matrix(cm, classes)

    def _plot_loss_curves(self, train_losses: List[float], val_losses: List[float]):
        """Plot training and validation loss curves."""
        self.loss_fig.clear()
        self.loss_ax = self.loss_fig.add_subplot(111)
        self.loss_ax.set_facecolor('none')
        style_dark_axes(self.loss_ax)

        epochs = range(1, len(train_losses) + 1)
        self.loss_ax.plot(epochs, train_losses, label='Train Loss', color='#1f77b4', linewidth=1.5)
        self.loss_ax.plot(epochs, val_losses, label='Val Loss', color='#ff7f0e', linewidth=1.5)

        self.loss_ax.set_xlabel('Epoch', color='white')
        self.loss_ax.set_ylabel('Loss', color='white')
        self.loss_ax.set_title('Training & Validation Loss', color='white')
        legend = self.loss_ax.legend(facecolor='none', edgecolor='white', labelcolor='white')
        legend.get_frame().set_alpha(0.5)
        self.loss_ax.grid(True, alpha=0.3, color='white')

        try:
            self.loss_fig.tight_layout()
        except (ValueError, np.linalg.LinAlgError):
            pass  # Ignore layout errors when figure size is invalid
        self.loss_canvas.draw()

    def _plot_confusion_matrix(self, cm: np.ndarray, classes: np.ndarray):
        """Plot confusion matrix with proper labels."""
        # Clear figure and recreate axes to properly handle colorbar
        self.cm_fig.clear()
        self.cm_ax = self.cm_fig.add_subplot(111)
        self.cm_ax.set_facecolor('none')
        style_dark_axes(self.cm_ax)

        n_classes = len(classes)
        im = self.cm_ax.imshow(cm, interpolation='nearest', cmap='Blues')

        # Add colorbar with white label
        cbar = self.cm_fig.colorbar(im, ax=self.cm_ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(colors='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        for label in cbar.ax.get_yticklabels():
            label.set_color('white')

        # Set ticks with white color
        self.cm_ax.set_xticks(np.arange(n_classes))
        self.cm_ax.set_yticks(np.arange(n_classes))
        self.cm_ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=8, color='white')
        self.cm_ax.set_yticklabels(classes, fontsize=8, color='white')

        # Labels
        self.cm_ax.set_xlabel('Predicted Label', color='white')
        self.cm_ax.set_ylabel('True Label', color='white')
        self.cm_ax.set_title('Confusion Matrix', color='white')

        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(n_classes):
            for j in range(n_classes):
                self.cm_ax.text(j, i, format(cm[i, j], 'd'),
                               ha="center", va="center",
                               color="white" if cm[i, j] > thresh else "black",
                               fontsize=8)

        try:
            self.cm_fig.tight_layout()
        except (ValueError, np.linalg.LinAlgError):
            pass  # Ignore layout errors when figure size is invalid
        self.cm_canvas.draw()

    @pyqtSlot(str)
    def _on_training_error(self, error_msg: str):
        """Handle training error."""
        self.btn_train.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.status_label.setText(f"Error: {error_msg[:50]}...")
        self.results_text.setText(f"Training failed:\n{error_msg}")

    def _save_model(self):
        """Save the trained model."""
        if self._model is None:
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Model", "", "PyTorch Model (*.pt *.pth)"
        )
        if filename:
            # Save model state dict and metadata
            save_data = {
                'model_state_dict': self._model.state_dict(),
                'model_config': {
                    'input_dim': self._model.network[0].in_features,
                    'hidden_layers': self._parse_hidden_layers(),
                    'num_classes': self._model.network[-1].out_features,
                    'dropout_rate': self.spin_dropout.value(),
                    'activation': self.combo_activation.currentText()
                },
                'label_encoder_classes': list(self._label_encoder.classes_) if self._label_encoder else []
            }
            torch.save(save_data, filename)
            print(f"Model saved to {filename}")

    def _save_results(self):
        """Save training results including confusion matrix."""
        if self._last_results is None:
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Results", "", "JSON Files (*.json)"
        )
        if filename:
            results_to_save = {
                'accuracy': float(self._last_results['accuracy']),
                'confusion_matrix': self._last_results['confusion_matrix'].tolist(),
                'classes': list(self._label_encoder.classes_) if self._label_encoder else [],
                'train_losses': self._last_results['train_losses'],
                'val_losses': self._last_results['val_losses'],
                'training_config': {
                    'hidden_layers': self._parse_hidden_layers(),
                    'activation': self.combo_activation.currentText(),
                    'dropout': self.spin_dropout.value(),
                    'epochs': self.spin_epochs.value(),
                    'batch_size': self.spin_batch_size.value(),
                    'learning_rate': self.spin_lr.value(),
                    'optimizer': self.combo_optimizer.currentText(),
                    'loss_function': self.combo_loss.currentText(),
                    'weight_decay': self.spin_weight_decay.value(),
                    'l1_lambda': self.spin_l1_lambda.value(),
                    'val_split': self.spin_val_split.value()
                }
            }
            with open(filename, 'w') as f:
                json.dump(results_to_save, f, indent=4)
            print(f"Results saved to {filename}")
