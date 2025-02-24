import sys
import json
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QListWidget, QListWidgetItem, 
                             QFileDialog, QComboBox, QLabel, QGroupBox, QCheckBox)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Use the "fast" style.
plt.style.use('fast')

########################################
# Dummy preprocessor class definition
########################################

class SpectrumPreprocessor:
    def __init__(self, cropping=True, 
                 baseline_correction=False,
                 remove_cosmic_rays=True,
                 normalization=False,
                 smoothing=False):
        self.cropping = cropping
        self.baseline_correction = baseline_correction
        self.remove_cosmic_rays = remove_cosmic_rays
        self.normalization = normalization
        self.smoothing = smoothing

    def preprocess(self, obj):
        # Dummy preprocessing: work on a copy.
        new_intensity = obj.intensity.copy()
        new_raman = obj.raman_shift_cm.copy()
        if self.cropping:
            n = int(len(new_intensity) * 0.7)
            new_intensity = new_intensity[:n]
            new_raman = new_raman[:n]
        if self.remove_cosmic_rays:
            new_intensity = new_intensity - np.random.normal(0, 0.01, size=new_intensity.shape)
        # Other options are left unchanged in this dummy.
        return SpectraData(new_intensity, new_raman, obj.metadata)

########################################
# Data class for spectra
########################################

class SpectraData:
    def __init__(self, intensity, raman_shift_cm, metadata):
        """
        intensity: 1D numpy array of intensity values
        raman_shift_cm: 1D numpy array of Raman shift values (in cm^-1)
        metadata: dictionary, e.g.
            {
                "patient_id": 489,
                "sample_type": "plasma",
                "date": "20230622",
                "position": "16",
                "staging": 0,
                "gender": "Male",
                "race": "White",
                "spectrum_id": 2
            }
        """
        self.intensity = intensity
        self.raman_shift_cm = raman_shift_cm
        self.metadata = metadata

########################################
# Viewer class
########################################

class SpectraViewer(QMainWindow):
    def __init__(self, data_objects, preprocessor):
        super().__init__()
        self.setWindowTitle("Spectra Viewer")
        # Store the untouched data.
        self.original_data_objects = data_objects  
        self.preprocessor = preprocessor
        # Preprocess initially.
        self.data_objects = [self.preprocessor.preprocess(obj) for obj in self.original_data_objects]
        
        self.selected_indices = []          # indices selected from the scatter plot
        self.selected_index_to_line = {}    # mapping from data index to its plotted spectrum line
        self.hovered_index = None           # index of the spectrum currently hovered in the line plot

        self.embedding_method = "T-SNE"     # default embedding method
        self.embedding_dim = 2              # default dimension (2D)
        self.embedding_result = None        # will hold computed embedding
        self.color_by = "None"              # default: no coloring

        # Lasso instance (for 2D selection); initially None.
        self.lasso = None

        # For panning in 2D.
        self._pan_active = False
        self._pan_press_event = None

        self.initUI()
        self.compute_embedding()
        self.plot_embedding()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        ########################################
        # LEFT PANEL: Preprocessing Options, 
        # Embedding & Color Controls, Scatter Plot, and Navigation Buttons.
        ########################################
        left_layout = QVBoxLayout()
        
        # Preprocessing Options Group Box.
        preproc_group = QGroupBox("Preprocessing Options")
        preproc_layout = QVBoxLayout()
        self.cb_cropping = QCheckBox("Cropping")
        self.cb_cropping.setChecked(self.preprocessor.cropping)
        self.cb_baseline = QCheckBox("Baseline Correction")
        self.cb_baseline.setChecked(self.preprocessor.baseline_correction)
        self.cb_cosmic = QCheckBox("Remove Cosmic Rays")
        self.cb_cosmic.setChecked(self.preprocessor.remove_cosmic_rays)
        self.cb_norm = QCheckBox("Normalization")
        self.cb_norm.setChecked(self.preprocessor.normalization)
        self.cb_smooth = QCheckBox("Smoothing")
        self.cb_smooth.setChecked(self.preprocessor.smoothing)
        for cb in [self.cb_cropping, self.cb_baseline, self.cb_cosmic, self.cb_norm, self.cb_smooth]:
            preproc_layout.addWidget(cb)
        self.btn_recalculate = QPushButton("Recalculate")
        self.btn_recalculate.clicked.connect(self.update_preprocessing)
        preproc_layout.addWidget(self.btn_recalculate)
        preproc_group.setLayout(preproc_layout)
        left_layout.addWidget(preproc_group)
        
        # Embedding and Color Controls.
        control_layout = QHBoxLayout()
        method_label = QLabel("Embedding:")
        self.embedding_combo = QComboBox()
        self.embedding_combo.addItems(["T-SNE", "PCA"])
        self.embedding_combo.currentIndexChanged.connect(self.on_embedding_change)
        control_layout.addWidget(method_label)
        control_layout.addWidget(self.embedding_combo)
        
        dim_label = QLabel("Dimension:")
        self.dim_combo = QComboBox()
        self.dim_combo.addItems(["2D", "3D"])
        self.dim_combo.currentIndexChanged.connect(self.on_dimension_change)
        control_layout.addWidget(dim_label)
        control_layout.addWidget(self.dim_combo)
        
        color_label = QLabel("Color by:")
        self.color_combo = QComboBox()
        self.color_combo.addItems(["None", "staging", "position", "patient_id", "date"])
        self.color_combo.currentIndexChanged.connect(self.on_color_change)
        control_layout.addWidget(color_label)
        control_layout.addWidget(self.color_combo)
        left_layout.addLayout(control_layout)
        
        # Increase the size of the embedding scatter plot.
        self.fig_scatter = Figure(figsize=(7, 6))
        self.canvas_scatter = FigureCanvas(self.fig_scatter)
        left_layout.addWidget(self.canvas_scatter)
        
        # Connect scroll and mouse events for zoom and pan.
        self.canvas_scatter.mpl_connect('scroll_event', self.on_scroll)
        self.canvas_scatter.mpl_connect('button_press_event', self.on_pan_press)
        self.canvas_scatter.mpl_connect('motion_notify_event', self.on_pan_motion)
        self.canvas_scatter.mpl_connect('button_release_event', self.on_pan_release)
        
        # Navigation Buttons.
        nav_layout = QHBoxLayout()
        self.btn_home = QPushButton("Home")
        self.btn_home.clicked.connect(self.reset_view)
        nav_layout.addWidget(self.btn_home)
        self.btn_select = QPushButton("Select")
        self.btn_select.setCheckable(True)
        self.btn_select.toggled.connect(self.toggle_lasso)
        nav_layout.addWidget(self.btn_select)
        left_layout.addLayout(nav_layout)
        
        # Clear selection button.
        self.btn_clear = QPushButton("Clear Selection")
        self.btn_clear.clicked.connect(self.clear_selection)
        left_layout.addWidget(self.btn_clear)
        
        ########################################
        # RIGHT PANEL: 2D Spectra Plot and Metadata List.
        ########################################
        right_layout = QVBoxLayout()
        self.fig_line = Figure(figsize=(5, 3))
        self.canvas_line = FigureCanvas(self.fig_line)
        self.ax_line = self.fig_line.add_subplot(111)
        self.ax_line.set_title("Select points to view spectra")
        right_layout.addWidget(self.canvas_line)
        self.canvas_line.mpl_connect('motion_notify_event', self.on_line_hover)
        self.canvas_line.mpl_connect('pick_event', self.on_line_pick)
        
        self.metadata_list = QListWidget()
        self.metadata_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.metadata_list.itemSelectionChanged.connect(self.update_line_highlights)
        self.btn_save_metadata = QPushButton("Save Metadata")
        self.btn_save_metadata.clicked.connect(self.save_metadata)
        meta_layout = QVBoxLayout()
        meta_layout.addWidget(self.metadata_list)
        meta_layout.addWidget(self.btn_save_metadata)
        meta_widget = QWidget()
        meta_widget.setLayout(meta_layout)
        right_layout.addWidget(meta_widget)
        
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)
        
        # For 3D selection via picking, connect pick events.
        self.canvas_scatter.mpl_connect('pick_event', self.on_scatter_pick)

    def reset_view(self):
        """Reset the view by replotting the embedding (restoring original view)."""
        self.plot_embedding()

    def toggle_lasso(self, checked):
        """Toggle the lasso tool in 2D mode. When off, panning is active."""
        if self.embedding_dim != 2:
            return
        if checked:
            # Activate lasso selection.
            if self.lasso is None:
                self.lasso = LassoSelector(self.ax_scatter, onselect=self.onselect)
                if hasattr(self.lasso, 'line'):
                    self.lasso.line.set_color('gold')
        else:
            # Deactivate lasso.
            if self.lasso is not None:
                self.lasso.disconnect_events()
                self.lasso = None

    def on_color_change(self, index):
        self.color_by = self.color_combo.currentText()
        self.plot_embedding()

    def update_preprocessing(self):
        """Update preprocessor settings and reapply preprocessing when Recalculate is clicked."""
        self.preprocessor.cropping = self.cb_cropping.isChecked()
        self.preprocessor.baseline_correction = self.cb_baseline.isChecked()
        self.preprocessor.remove_cosmic_rays = self.cb_cosmic.isChecked()
        self.preprocessor.normalization = self.cb_norm.isChecked()
        self.preprocessor.smoothing = self.cb_smooth.isChecked()
        
        self.data_objects = [self.preprocessor.preprocess(obj) for obj in self.original_data_objects]
        self.compute_embedding()
        self.plot_embedding()
        self.clear_selection()

    def compute_embedding(self):
        X = np.array([obj.intensity for obj in self.data_objects])
        n_components = self.embedding_dim
        if self.embedding_method == "T-SNE":
            self.embedding_result = TSNE(n_components=n_components, random_state=42).fit_transform(X)
        else:
            self.embedding_result = PCA(n_components=n_components).fit_transform(X)

    def plot_embedding(self):
        self.fig_scatter.clear()
        if self.embedding_dim == 2:
            self.ax_scatter = self.fig_scatter.add_subplot(111)
        else:
            self.ax_scatter = self.fig_scatter.add_subplot(111, projection='3d')
            
        if self.embedding_result is None:
            return

        # Determine colors.
        categorical = False
        if self.color_by == "None":
            point_colors = None
        elif self.color_by in ["staging", "position", "date", "patient_id"]:
            categorical = True
            values = [obj.metadata.get(self.color_by, None) for obj in self.data_objects]
            unique_vals = sorted(set(values))
            cmap = plt.cm.get_cmap('viridis_r', len(unique_vals))
            point_colors = [unique_vals.index(v) for v in values]
            cat_unique = unique_vals
        # elif self.color_by == "patient_id":
        #     values = [obj.metadata.get(self.color_by, None) for obj in self.data_objects]
        #     try:
        #         numeric_values = [float(v) for v in values]
        #         point_colors = numeric_values
        #     except Exception:
        #         categorical = True
        #         unique_vals = sorted(set(values))
        #         cmap = plt.cm.get_cmap('viridis_r', len(unique_vals))
        #         point_colors = [unique_vals.index(v) for v in values]
        #         cat_unique = unique_vals

        # Plot scatter.
        if self.embedding_dim == 2:
            if not categorical and point_colors is not None:
                sc = self.ax_scatter.scatter(
                    self.embedding_result[:, 0],
                    self.embedding_result[:, 1],
                    c=point_colors,
                    cmap='viridis',
                    alpha=0.6,
                    picker=True
                )
                cbar = self.fig_scatter.colorbar(sc, ax=self.ax_scatter)
                cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                cbar.set_label(self.color_by)
            elif categorical:
                sc = self.ax_scatter.scatter(
                    self.embedding_result[:, 0],
                    self.embedding_result[:, 1],
                    c=point_colors,
                    cmap=cmap,
                    alpha=0.6,
                    picker=True
                )
                cbar = self.fig_scatter.colorbar(sc, ax=self.ax_scatter, ticks=range(len(cat_unique)))
                if self.color_by == "staging":
                    ticklabels = []
                    for v in cat_unique:
                        if v == 0:
                            ticklabels.append("healthy")
                        elif v == 1:
                            ticklabels.append("early stage")
                        elif v == 2:
                            ticklabels.append("late stage")
                        else:
                            ticklabels.append(str(v))
                else:
                    ticklabels = [str(v) for v in cat_unique]
                cbar.set_ticks(range(len(cat_unique)))
                cbar.set_ticklabels(ticklabels)
                cbar.ax.tick_params(labelsize=10)
                cbar.set_label(self.color_by)
            else:
                self.ax_scatter.scatter(
                    self.embedding_result[:, 0],
                    self.embedding_result[:, 1],
                    color='C0',
                    alpha=0.6,
                    picker=True
                )
            self.ax_scatter.set_title(f"2D {self.embedding_method} of Spectra Intensities")
            self.canvas_scatter.draw()
            # In 2D, if the Select button is toggled, add the lasso.
            if self.btn_select.isChecked():
                self.lasso = LassoSelector(self.ax_scatter, onselect=self.onselect)
                if hasattr(self.lasso, 'line'):
                    self.lasso.line.set_color('gold')
        else:
            if not categorical and point_colors is not None:
                sc = self.ax_scatter.scatter(
                    self.embedding_result[:, 0],
                    self.embedding_result[:, 1],
                    self.embedding_result[:, 2],
                    c=point_colors,
                    cmap='viridis',
                    alpha=0.6,
                    picker=True
                )
                cbar = self.fig_scatter.colorbar(sc, ax=self.ax_scatter)
                cbar.set_label(self.color_by)
            elif categorical:
                sc = self.ax_scatter.scatter(
                    self.embedding_result[:, 0],
                    self.embedding_result[:, 1],
                    self.embedding_result[:, 2],
                    c=point_colors,
                    cmap=cmap,
                    alpha=0.6,
                    picker=True
                )
                cbar = self.fig_scatter.colorbar(sc, ax=self.ax_scatter, ticks=range(len(cat_unique)))
                if self.color_by == "staging":
                    ticklabels = []
                    for v in cat_unique:
                        if v == 0:
                            ticklabels.append("healthy")
                        elif v == 1:
                            ticklabels.append("early stage")
                        elif v == 2:
                            ticklabels.append("late stage")
                        else:
                            ticklabels.append(str(v))
                else:
                    ticklabels = [str(v) for v in cat_unique]
                cbar.set_ticks(range(len(cat_unique)))
                cbar.set_ticklabels(ticklabels)
                cbar.ax.tick_params(labelsize=10)
                cbar.set_label(self.color_by)
            else:
                self.ax_scatter.scatter(
                    self.embedding_result[:, 0],
                    self.embedding_result[:, 1],
                    self.embedding_result[:, 2],
                    color='C0',
                    alpha=0.6,
                    picker=True
                )
            self.ax_scatter.set_title(f"3D {self.embedding_method} of Spectra Intensities")
            self.canvas_scatter.draw()

    def on_embedding_change(self, index):
        self.embedding_method = self.embedding_combo.currentText()
        self.clear_selection()
        self.compute_embedding()
        self.plot_embedding()

    def on_dimension_change(self, index):
        text = self.dim_combo.currentText()
        self.embedding_dim = 2 if text == "2D" else 3
        self.clear_selection()
        self.compute_embedding()
        self.plot_embedding()

    def onselect(self, verts):
        if self.embedding_dim != 2:
            return
        path = Path(verts)
        ind = np.nonzero(path.contains_points(self.embedding_result))[0]
        self.selected_indices = ind
        self.update_line_plot()

    def on_scatter_pick(self, event):
        # Process pick events only in 3D mode when the "Select" button is active.
        if self.embedding_dim != 3:
            return
        if not self.btn_select.isChecked():
            return
        if hasattr(event, 'ind') and len(event.ind) > 0:
            picked_index = event.ind[0]
            if picked_index not in self.selected_indices:
                self.selected_indices.append(picked_index)
            self.update_line_plot()

    def on_scroll(self, event):
        ax = event.inaxes
        if ax is None:
            return
        if self.embedding_dim == 2:
            cur_xlim = ax.get_xlim()
            cur_ylim = ax.get_ylim()
            xdata = event.xdata
            ydata = event.ydata
            scale_factor = 0.9 if event.button == 'up' else 1.1
            new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
            new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
            ax.set_xlim([xdata - new_width/2, xdata + new_width/2])
            ax.set_ylim([ydata - new_height/2, ydata + new_height/2])
        else:
            if hasattr(ax, 'dist'):
                if event.button == 'up':
                    ax.dist *= 0.9
                elif event.button == 'down':
                    ax.dist *= 1.1
        self.canvas_scatter.draw_idle()

    def on_pan_press(self, event):
        # If in 2D mode and not in lasso mode, start panning.
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
        # Calculate shift.
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
        self.ax_line.clear()
        self.selected_index_to_line = {}
        self.metadata_list.clear()
        if len(self.selected_indices) == 0:
            self.ax_line.set_title("No points selected")
        else:
            for idx in self.selected_indices:
                obj = self.data_objects[idx]
                line, = self.ax_line.plot(obj.raman_shift_cm, obj.intensity, color='C0', picker=5)
                self.selected_index_to_line[idx] = line
                item = QListWidgetItem(str(obj.metadata))
                item.setData(Qt.UserRole, idx)
                self.metadata_list.addItem(item)
            self.ax_line.set_title("Spectra for Selected Points")
            self.ax_line.set_xlabel("Raman Shift")
            self.ax_line.set_ylabel("Intensity")
        self.canvas_line.draw_idle()

    def clear_selection(self):
        self.selected_indices = []
        self.ax_line.clear()
        self.ax_line.set_title("Select points to view spectra")
        self.metadata_list.clear()
        self.selected_index_to_line = {}
        self.canvas_line.draw_idle()

    def on_line_hover(self, event):
        if event.inaxes != self.ax_line:
            self.hovered_index = None
            self.update_line_highlights()
            return
        hovered_index = None
        for idx, line in self.selected_index_to_line.items():
            contains, _ = line.contains(event)
            if contains:
                hovered_index = idx
                break
        self.hovered_index = hovered_index
        self.update_line_highlights()

    def on_line_pick(self, event):
        artist = event.artist
        picked_index = None
        for idx, line in self.selected_index_to_line.items():
            if artist == line:
                picked_index = idx
                break
        if picked_index is not None:
            self.metadata_list.clearSelection()
            for i in range(self.metadata_list.count()):
                item = self.metadata_list.item(i)
                if item.data(Qt.UserRole) == picked_index:
                    item.setSelected(True)
                    self.metadata_list.scrollToItem(item)
                    break
        self.update_line_highlights()

    def update_line_highlights(self):
        selected_items = self.metadata_list.selectedItems()
        selected_indices_from_list = [item.data(Qt.UserRole) for item in selected_items]
        for idx, line in self.selected_index_to_line.items():
            if idx == self.hovered_index or idx in selected_indices_from_list:
                line.set_linewidth(3)
                line.set_color('orange')
            else:
                line.set_linewidth(1)
                line.set_color('C0')
        self.canvas_line.draw_idle()

    def save_metadata(self):
        metadata_list = [self.data_objects[idx].metadata for idx in self.selected_indices]
        filename, _ = QFileDialog.getSaveFileName(self, "Save Metadata", "", "JSON Files (*.json)")
        if filename:
            with open(filename, "w") as f:
                json.dump(metadata_list, f, indent=4)

if __name__ == "__main__":
    # Create dummy data.
    num_samples = 50
    data_objects = []
    for i in range(num_samples):
        intensity = np.random.rand(100)
        raman_shift = np.linspace(100, 3000, 100)
        metadata = {
            "patient_id": np.random.randint(100, 600),
            "sample_type": "plasma",
            "date": "20230622",
            "position": str(np.random.randint(1, 20)),
            "staging": np.random.choice([0,1,2]),
            "gender": "Male",
            "race": "White",
            "spectrum_id": i
        }
        data_objects.append(SpectraData(intensity, raman_shift, metadata))
        
    preprocessor = SpectrumPreprocessor(cropping=True, 
                                        baseline_correction=False,
                                        remove_cosmic_rays=True,
                                        normalization=False,
                                        smoothing=False)
    app = QApplication(sys.argv)
    window = SpectraViewer(data_objects, preprocessor)
    window.resize(1300, 650)
    window.show()
    sys.exit(app.exec_())
