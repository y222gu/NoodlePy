import sys
import json
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QListWidget, QListWidgetItem, 
                             QFileDialog, QComboBox, QLabel)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # Import required for 3D plotting
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# Use a dark background style
plt.style.use('fast')

# Define a simple data class for the spectra
class SpectraData:
    def __init__(self, intensity, raman_shift_cm, metadata):
        """
        intensity: 1D numpy array of intensity values
        raman_shift_cm: 1D numpy array of corresponding Raman shift values (in cm^-1)
        metadata: any additional information (e.g., a string or dict)
        """
        self.intensity = intensity
        self.raman_shift_cm = raman_shift_cm
        self.metadata = metadata

class Spectra3DViewer(QMainWindow):
    def __init__(self, data_objects):
        super().__init__()
        self.setWindowTitle("3D Spectra Viewer")
        self.data_objects = data_objects
        self.selected_indices = []          # indices of points selected via pick
        self.selected_index_to_line = {}    # mapping for the spectra plot (2D)
        self.hovered_index = None
        self.embedding_method = "T-SNE"     # default
        self.embedding_result = None        # will hold the 3D embedding
        self.initUI()
        self.compute_embedding()
        self.plot_embedding()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # LEFT SIDE: 3D scatter plot with combo box and clear button.
        left_layout = QVBoxLayout()
        method_layout = QHBoxLayout()
        method_label = QLabel("Embedding:")
        self.embedding_combo = QComboBox()
        self.embedding_combo.addItems(["T-SNE", "PCA"])
        self.embedding_combo.currentIndexChanged.connect(self.on_embedding_change)
        method_layout.addWidget(method_label)
        method_layout.addWidget(self.embedding_combo)
        left_layout.addLayout(method_layout)

        # Create a 3D scatter plot
        self.fig_scatter = Figure(figsize=(5, 4))
        self.canvas_scatter = FigureCanvas(self.fig_scatter)
        # Create a 3D axes using projection='3d'
        self.ax_scatter = self.fig_scatter.add_subplot(111, projection='3d')
        left_layout.addWidget(self.canvas_scatter)
        
        self.btn_clear = QPushButton("Clear Selection")
        self.btn_clear.clicked.connect(self.clear_selection)
        left_layout.addWidget(self.btn_clear)

        # RIGHT SIDE: 2D line plot and metadata list as before.
        right_layout = QVBoxLayout()
        self.fig_line = Figure(figsize=(5, 3))
        self.canvas_line = FigureCanvas(self.fig_line)
        self.ax_line = self.fig_line.add_subplot(111)
        self.ax_line.set_title("Select a point (via click) to view its spectrum")
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

        # Connect picking on the 3D scatter plot
        self.canvas_scatter.mpl_connect('pick_event', self.on_scatter_pick)

    def compute_embedding(self):
        X = np.array([obj.intensity for obj in self.data_objects])
        if self.embedding_method == "T-SNE":
            self.embedding_result = TSNE(n_components=3, random_state=42).fit_transform(X)
        else:  # PCA
            self.embedding_result = PCA(n_components=3).fit_transform(X)

    def plot_embedding(self):
        self.ax_scatter.clear()
        # Create a 3D scatter plot; note that picking in 3D is enabled by setting picker=True
        self.scatter = self.ax_scatter.scatter(
            self.embedding_result[:, 0],
            self.embedding_result[:, 1],
            self.embedding_result[:, 2],
            alpha=0.6,
            picker=True
        )
        title = f"3D {self.embedding_method} of Spectra Intensities"
        self.ax_scatter.set_title(title)
        self.canvas_scatter.draw()

    def on_embedding_change(self, index):
        self.embedding_method = self.embedding_combo.currentText()
        self.clear_selection()
        self.compute_embedding()
        self.plot_embedding()

    def on_scatter_pick(self, event):
        """
        Called when a point in the 3D scatter plot is clicked.
        We rely on the pick event to determine which data index was clicked.
        """
        # The event.ind attribute contains the indices of the picked points.
        if hasattr(event, 'ind') and len(event.ind) > 0:
            picked_index = event.ind[0]
            if picked_index not in self.selected_indices:
                self.selected_indices.append(picked_index)
            self.update_line_plot()

    def update_line_plot(self):
        self.ax_line.clear()
        self.selected_index_to_line = {}
        self.metadata_list.clear()
        if len(self.selected_indices) == 0:
            self.ax_line.set_title("No points selected")
        else:
            for idx in self.selected_indices:
                obj = self.data_objects[idx]
                # Plot each spectrum in default blue using 'C0'
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
        self.ax_line.set_title("Select a point (via click) to view its spectrum")
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
    num_samples = 50
    data_objects = []
    for i in range(num_samples):
        intensity = np.random.rand(100)
        raman_shift = np.linspace(100, 3000, 100)
        metadata = f"Sample {i}"
        data_objects.append(SpectraData(intensity, raman_shift, metadata))
        
    app = QApplication(sys.argv)
    window = Spectra3DViewer(data_objects)
    window.resize(1200, 600)
    window.show()
    sys.exit(app.exec_())