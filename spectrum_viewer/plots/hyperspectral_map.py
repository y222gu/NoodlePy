"""Hyperspectral mapping plot widget."""

import io
import math
import pickle
import numpy as np
from typing import List, Dict, Optional, Set
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QGridLayout, QSizePolicy, QScrollArea, QFrame,
    QPushButton, QDialog, QHBoxLayout
)
from PyQt5.QtCore import pyqtSignal, Qt
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar
)
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from ..utils.dark_theme import style_dark_axes, make_mpl_transparent


class MapSnapshotDialog(QDialog):
    """Non-modal dialog that displays a copy of a matplotlib figure with navigation toolbar."""

    def __init__(self, source_fig: Figure, title: str = "Map Snapshot", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(700, 600)
        self.setAttribute(Qt.WA_DeleteOnClose)

        # Dark background styling
        self.setStyleSheet(
            "QDialog { background-color: #1e1e1e; color: #dcdcdc; }"
            "QToolBar { background-color: #2a2a2a; border: none; }"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Copy figure via pickle roundtrip
        try:
            buf = io.BytesIO()
            pickle.dump(source_fig, buf)
            buf.seek(0)
            self._fig = pickle.load(buf)
        except Exception:
            # Fallback: render source as PNG and display in a new figure
            self._fig = Figure(figsize=(8, 6))
            ax = self._fig.add_subplot(111)
            png_buf = io.BytesIO()
            source_fig.savefig(png_buf, format='png', dpi=150,
                               facecolor='#1e1e1e', bbox_inches='tight')
            png_buf.seek(0)
            from matplotlib.image import imread
            img = imread(png_buf)
            ax.imshow(img)
            ax.axis('off')
            self._fig.tight_layout()

        # Apply dark background to the figure
        self._fig.patch.set_facecolor('#1e1e1e')
        for ax in self._fig.get_axes():
            style_dark_axes(ax)

        self._canvas = FigureCanvas(self._fig)
        self._toolbar = NavigationToolbar(self._canvas, self)

        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas, stretch=1)


class HyperspectralMapWidget(QWidget):
    """Widget for displaying hyperspectral intensity maps."""

    # Signals
    point_clicked = pyqtSignal(int, int)  # Emits (sample_id, spectrum_index)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.sample_maps: Dict[int, 'SampleMapCanvas'] = {}
        self.current_wavenumber: float = 1000.0
        self.colormap: str = 'turbo'
        self._color_mode: str = 'intensity'  # 'intensity' or 'scatter'
        self._scatter_colors: Optional[np.ndarray] = None
        self._index_to_pos: Dict[int, int] = {}
        self._selected_indices: Optional[Set[int]] = None

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Scroll area for multiple sample maps
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)

        self.scroll_widget = QWidget()
        self.scroll_layout = QGridLayout(self.scroll_widget)
        self.scroll_layout.setContentsMargins(5, 5, 5, 5)
        self.scroll_layout.setSpacing(10)

        self.scroll_area.setWidget(self.scroll_widget)
        layout.addWidget(self.scroll_area)

    def set_samples(self, samples: Dict[int, List[int]], data_manager):
        """
        Set up sample maps arranged in a grid.

        Args:
            samples: Dictionary mapping sample ID to list of spectrum indices
            data_manager: ViewerDataManager instance for data access
        """
        # Clear existing maps
        for widget in self.sample_maps.values():
            widget.setParent(None)
            widget.deleteLater()
        self.sample_maps.clear()

        # Remove any remaining items from grid layout
        while self.scroll_layout.count():
            item = self.scroll_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Collect valid sample ids
        sorted_ids = [sid for sid in sorted(samples.keys()) if len(samples[sid]) > 0]
        n_maps = len(sorted_ids)

        # Determine grid columns: 1 col for 1 map, 2 cols for 2-3, 3 cols for 4+
        if n_maps <= 1:
            n_cols = 1
        elif n_maps <= 3:
            n_cols = 2
        else:
            n_cols = 3

        # Create and place maps in grid
        for i, sample_id in enumerate(sorted_ids):
            indices = samples[sample_id]

            map_widget = SampleMapCanvas(sample_id, indices, data_manager, self)
            map_widget.set_colormap(self.colormap)
            map_widget.point_clicked.connect(
                lambda idx, sid=sample_id: self.point_clicked.emit(sid, idx)
            )

            self.sample_maps[sample_id] = map_widget
            row = i // n_cols
            col = i % n_cols
            self.scroll_layout.addWidget(map_widget, row, col)

    def update_wavenumber(self, wavenumber: float):
        """Update all maps to show intensity at new wavenumber."""
        self.current_wavenumber = wavenumber

        # Compute global vmin/vmax across all maps for unified colorbar
        all_intensities = []
        for map_widget in self.sample_maps.values():
            if map_widget._x_coords is not None and len(map_widget._x_coords) > 0:
                intensities = map_widget.data_manager.get_intensity_at_wavenumber(
                    wavenumber, map_widget.indices
                )
                all_intensities.append(intensities)

        if all_intensities:
            combined = np.concatenate(all_intensities)
            global_vmin = float(np.nanmin(combined))
            global_vmax = float(np.nanmax(combined))
        else:
            global_vmin = None
            global_vmax = None

        for map_widget in self.sample_maps.values():
            map_widget.update_wavenumber(wavenumber, vmin=global_vmin, vmax=global_vmax)

    def set_colormap(self, cmap_name: str):
        """Set colormap for all maps."""
        self.colormap = cmap_name
        for map_widget in self.sample_maps.values():
            map_widget.set_colormap(cmap_name)  # also sets _needs_full_redraw

    def set_color_mode(self, mode: str):
        """Set color mode for all maps ('intensity' or 'scatter')."""
        self._color_mode = mode
        for map_widget in self.sample_maps.values():
            map_widget._color_mode = mode
            map_widget._needs_full_redraw = True
        self.refresh_all()

    def update_point_colors(self, scatter_colors, index_to_pos):
        """Update scatter plot colors for all maps."""
        self._scatter_colors = scatter_colors
        self._index_to_pos = index_to_pos
        for map_widget in self.sample_maps.values():
            map_widget._scatter_colors = scatter_colors
            map_widget._index_to_pos = index_to_pos
        if self._color_mode == 'scatter':
            self.refresh_all()

    def set_selection(self, selected_indices, scatter_colors, index_to_pos):
        """Set lasso selection highlighting on all maps."""
        self._selected_indices = set(selected_indices) if selected_indices else None
        self._scatter_colors = scatter_colors
        self._index_to_pos = index_to_pos
        for map_widget in self.sample_maps.values():
            map_widget._scatter_colors = scatter_colors
            map_widget._index_to_pos = index_to_pos
            map_widget._selected_set = self._selected_indices
            map_widget._needs_full_redraw = True
        self.refresh_all()

    def clear_selection(self):
        """Clear lasso selection highlighting on all maps."""
        self._selected_indices = None
        for map_widget in self.sample_maps.values():
            map_widget._selected_set = None
            map_widget._needs_full_redraw = True
        self.refresh_all()

    def highlight_point(self, spectrum_idx: int):
        """Highlight a specific spectrum point across all sample maps."""
        for map_widget in self.sample_maps.values():
            map_widget.highlight_point(spectrum_idx)

    def clear_highlight(self):
        """Clear highlight from all sample maps."""
        for map_widget in self.sample_maps.values():
            map_widget.clear_highlight()

    def refresh_all(self):
        """Refresh all sample maps."""
        for map_widget in self.sample_maps.values():
            map_widget.plot()

    def clear(self):
        """Clear all sample maps."""
        for widget in self.sample_maps.values():
            widget.setParent(None)
            widget.deleteLater()
        self.sample_maps.clear()

        # Remove any remaining items from grid layout
        while self.scroll_layout.count():
            item = self.scroll_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()


class SampleMapCanvas(QWidget):
    """Canvas for a single sample's hyperspectral map."""

    point_clicked = pyqtSignal(int)  # Emits spectrum index

    def __init__(self, sample_id, indices: List[int], data_manager, parent=None):
        super().__init__(parent)

        self.sample_id = sample_id
        self.indices = indices
        self.data_manager = data_manager
        self.current_wavenumber: float = 1000.0
        self.colormap: str = 'turbo'

        self._color_mode: str = 'intensity'
        self._x_coords: Optional[np.ndarray] = None
        self._y_coords: Optional[np.ndarray] = None
        self._intensities: Optional[np.ndarray] = None
        self._scatter_colors: Optional[np.ndarray] = None
        self._index_to_pos: Dict[int, int] = {}
        self._selected_set: Optional[Set[int]] = None
        self._global_vmin: Optional[float] = None
        self._global_vmax: Optional[float] = None

        # Fast-path rendering state
        self._scatter_artist = None
        self._cbar = None
        self._needs_full_redraw: bool = True
        self._highlight_artist = None

        self._label = self._build_label()

        self._init_ui()
        self._load_coordinates()

    def _build_label(self) -> str:
        """Build a descriptive label from the first spectrum's metadata."""
        if not self.indices:
            return f"Droplet {self.sample_id}"
        obj = self.data_manager.data_objects[self.indices[0]]
        patient = obj.metadata.get('patient', '?')
        droplet_id = obj.metadata.get('droplet_id', obj.metadata.get('sample', '?'))
        staging = obj.metadata.get('cancer_stage', obj.metadata.get('staging', ''))
        label = f"Patient {patient} | Droplet {droplet_id}"
        if staging:
            label += f" | {staging}"
        return label

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.fig = Figure(figsize=(6, 5))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)

        make_mpl_transparent(self.fig, self.canvas)

        self.canvas.mpl_connect('pick_event', self._on_pick)

        self.canvas.setMinimumHeight(300)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout.addWidget(self.canvas)

        # Snapshot overlay button
        self._snapshot_btn = QPushButton("\u2922", self.canvas)  # Unicode expand icon
        self._snapshot_btn.setFixedSize(28, 28)
        self._snapshot_btn.setToolTip("Open snapshot in new window")
        self._snapshot_btn.setStyleSheet(
            "QPushButton { background: rgba(60,60,60,180); color: white; "
            "border: 1px solid #555; border-radius: 4px; font-size: 16px; }"
            "QPushButton:hover { background: rgba(80,80,80,220); }"
        )
        self._snapshot_btn.clicked.connect(self._open_snapshot)
        self._reposition_snapshot_btn()

    def resizeEvent(self, event):
        """Reposition snapshot button on resize."""
        super().resizeEvent(event)
        self._reposition_snapshot_btn()

    def _reposition_snapshot_btn(self):
        """Position the snapshot button in the bottom-right corner of the canvas."""
        if hasattr(self, '_snapshot_btn') and hasattr(self, 'canvas'):
            cw = self.canvas.width()
            ch = self.canvas.height()
            self._snapshot_btn.move(cw - 34, ch - 34)

    def _open_snapshot(self):
        """Open a snapshot of the current figure in a new dialog."""
        dlg = MapSnapshotDialog(self.fig, title=self._label, parent=self.window())
        dlg.show()

    def _load_coordinates(self):
        """Load spatial coordinates from metadata."""
        self._x_coords, self._y_coords = self.data_manager.get_spatial_coordinates(self.indices)

    def update_wavenumber(self, wavenumber: float, vmin: float = None, vmax: float = None):
        """Update map for new wavenumber with optional global intensity range."""
        self.current_wavenumber = wavenumber
        self._global_vmin = vmin
        self._global_vmax = vmax

        # Use fast path if possible: intensity mode, no selection, scatter artist exists
        if (not self._needs_full_redraw
                and self._color_mode == 'intensity'
                and self._scatter_artist is not None
                and (self._selected_set is None or len(self._selected_set) == 0)):
            self._update_intensity_fast(vmin, vmax)
        else:
            self.plot()

    def _update_intensity_fast(self, vmin, vmax):
        """Fast-path update: only update scatter facecolors and colorbar clim."""
        self._intensities = self.data_manager.get_intensity_at_wavenumber(
            self.current_wavenumber, self.indices
        )

        if vmin is None:
            vmin = np.nanmin(self._intensities)
        if vmax is None:
            vmax = np.nanmax(self._intensities)

        # Update scatter facecolors
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap(self.colormap)
        new_colors = cmap(norm(self._intensities))
        self._scatter_artist.set_facecolors(new_colors)
        self._scatter_artist.set_clim(vmin, vmax)

        # Update colorbar
        if self._cbar is not None:
            self._cbar.mappable.set_clim(vmin, vmax)

        # Update title
        title_suffix = f" @ {self.current_wavenumber:.1f} cm^-1"
        self.ax.set_title(f"{self._label}{title_suffix}")
        style_dark_axes(self.ax)

        self.canvas.draw_idle()

    def set_colormap(self, cmap_name: str):
        """Set the colormap."""
        self.colormap = cmap_name
        self._needs_full_redraw = True

    def _get_point_scatter_colors(self):
        """Get scatter plot colors for each point in this sample."""
        if self._scatter_colors is None or not self._index_to_pos:
            return None
        point_colors = []
        for idx in self.indices:
            pos = self._index_to_pos.get(idx)
            if pos is not None and pos < len(self._scatter_colors):
                point_colors.append(self._scatter_colors[pos])
            else:
                point_colors.append([0.5, 0.5, 0.5, 1.0])
        return np.array(point_colors)

    def plot(self):
        """Plot the hyperspectral map."""
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self._scatter_artist = None
        self._cbar = None
        self._needs_full_redraw = False

        try:
            self.fig.patch.set_alpha(0.0)
        except Exception:
            pass

        self.ax.set_facecolor("none")

        if self._x_coords is None or len(self._x_coords) == 0:
            self.ax.set_title(f"{self._label}: No spatial data", color="white")
            self.ax.axis("off")
            self.canvas.draw_idle()
            return

        dot_size = 20
        has_selection = self._selected_set is not None and len(self._selected_set) > 0

        if self._color_mode == 'scatter':
            point_colors = self._get_point_scatter_colors()
            if point_colors is None:
                self._plot_intensity(dot_size, has_selection)
            else:
                self._plot_with_colors(point_colors, dot_size, has_selection)
            title_suffix = " (Scatter Colors)"
        else:
            self._plot_intensity(dot_size, has_selection)
            title_suffix = f" @ {self.current_wavenumber:.1f} cm^-1"

        self.ax.set_title(f"{self._label}{title_suffix}")
        self.ax.set_xlabel("X position [\u00b5m]")
        self.ax.set_ylabel("Y position [\u00b5m]")

        style_dark_axes(self.ax)

        # Make aspect ratio equal
        self.ax.set_aspect('equal', adjustable='box')

        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _plot_intensity(self, dot_size, has_selection):
        """Plot with intensity colormap."""
        self._intensities = self.data_manager.get_intensity_at_wavenumber(
            self.current_wavenumber, self.indices
        )

        # Use global vmin/vmax if provided, else fall back to local
        vmin = self._global_vmin if self._global_vmin is not None else np.nanmin(self._intensities)
        vmax = self._global_vmax if self._global_vmax is not None else np.nanmax(self._intensities)

        if has_selection:
            sel_mask = np.array([idx in self._selected_set for idx in self.indices])

            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            cmap = plt.get_cmap(self.colormap)
            intensity_rgba = cmap(norm(self._intensities))

            n = len(self.indices)
            facecolors = np.zeros((n, 4))
            edgecolors = np.zeros((n, 4))

            for i in range(n):
                if sel_mask[i]:
                    facecolors[i] = intensity_rgba[i]
                    edgecolors[i] = intensity_rgba[i]
                else:
                    facecolors[i] = [0, 0, 0, 0]  # transparent face
                    edgecolors[i] = [0.5, 0.5, 0.5, 1.0]  # grey edge

            self.ax.scatter(
                self._x_coords, self._y_coords,
                c=facecolors, edgecolors=edgecolors,
                s=dot_size, linewidths=0.8, picker=5
            )
        else:
            scatter = self.ax.scatter(
                self._x_coords, self._y_coords,
                c=self._intensities,
                cmap=self.colormap,
                vmin=vmin, vmax=vmax,
                s=dot_size, picker=5
            )
            self._scatter_artist = scatter

            # Add colorbar
            cbar = self.fig.colorbar(scatter, ax=self.ax, shrink=0.8)
            cbar.set_label("Intensity", color="white")
            cbar.ax.tick_params(colors="white")
            for spine in cbar.ax.spines.values():
                spine.set_color("white")
            self._cbar = cbar

    def _plot_with_colors(self, point_colors, dot_size, has_selection):
        """Plot with explicit per-point colors (from scatter plot)."""
        if has_selection:
            sel_mask = np.array([idx in self._selected_set for idx in self.indices])
            n = len(self.indices)
            facecolors = np.zeros((n, 4))
            edgecolors = np.zeros((n, 4))

            for i in range(n):
                if sel_mask[i]:
                    facecolors[i] = point_colors[i]
                    edgecolors[i] = point_colors[i]
                else:
                    facecolors[i] = [0, 0, 0, 0]  # transparent face
                    edgecolors[i] = [0.5, 0.5, 0.5, 1.0]  # grey edge

            self.ax.scatter(
                self._x_coords, self._y_coords,
                c=facecolors, edgecolors=edgecolors,
                s=dot_size, linewidths=0.8, picker=5
            )
        else:
            self.ax.scatter(
                self._x_coords, self._y_coords,
                c=point_colors,
                s=dot_size, picker=5
            )

    def highlight_point(self, spectrum_idx: int):
        """Highlight a specific spectrum point with a white circle."""
        if spectrum_idx not in self.indices:
            self.clear_highlight()
            return

        local_pos = self.indices.index(spectrum_idx)
        x = self._x_coords[local_pos]
        y = self._y_coords[local_pos]

        # Remove previous highlight if any
        self.clear_highlight()

        self._highlight_artist = self.ax.scatter(
            [x], [y], s=120, facecolors='none', edgecolors='white',
            linewidths=2.5, zorder=20
        )
        self.canvas.draw_idle()

    def clear_highlight(self):
        """Remove the highlight circle."""
        if hasattr(self, '_highlight_artist') and self._highlight_artist is not None:
            try:
                self._highlight_artist.remove()
            except Exception:
                pass
            self._highlight_artist = None
            self.canvas.draw_idle()

    def _on_pick(self, event):
        """Handle point picking."""
        ind = event.ind
        if len(ind) > 0 and ind[0] < len(self.indices):
            self.point_clicked.emit(self.indices[ind[0]])


class HyperspectralGridMap(QWidget):
    """Widget for grid-based hyperspectral mapping (for grid scan patterns)."""

    point_clicked = pyqtSignal(int)  # Emits spectrum index

    def __init__(self, sample_id, indices: List[int], data_manager, parent=None):
        super().__init__(parent)

        self.sample_id = sample_id
        self.indices = indices
        self.data_manager = data_manager
        self.current_wavenumber: float = 1000.0
        self.colormap: str = 'turbo'

        self._grid_data: Optional[np.ndarray] = None
        self._rows: List[int] = []
        self._cols: List[int] = []

        self._label = self._build_label()

        self._init_ui()
        self._analyze_grid_pattern()

    def _build_label(self) -> str:
        """Build a descriptive label from the first spectrum's metadata."""
        if not self.indices:
            return f"Droplet {self.sample_id}"
        obj = self.data_manager.data_objects[self.indices[0]]
        patient = obj.metadata.get('patient', '?')
        droplet_id = obj.metadata.get('droplet_id', obj.metadata.get('sample', '?'))
        staging = obj.metadata.get('cancer_stage', obj.metadata.get('staging', ''))
        label = f"Patient {patient} | Droplet {droplet_id}"
        if staging:
            label += f" | {staging}"
        return label

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.fig = Figure(figsize=(6, 5))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)

        make_mpl_transparent(self.fig, self.canvas)

        self.canvas.setMinimumHeight(350)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        layout.addWidget(self.canvas)

    def _analyze_grid_pattern(self):
        """Analyze data to determine grid pattern from metadata."""
        rows = []
        cols = []

        for idx in self.indices:
            obj = self.data_manager.data_objects[idx]
            row = obj.metadata.get('row', obj.metadata.get('grid_row', None))
            col = obj.metadata.get('column', obj.metadata.get('grid_col', None))

            if row is not None and col is not None:
                rows.append(int(row))
                cols.append(int(col))
            else:
                # Fall back to index-based positioning
                rows.append(len(rows) // 10)
                cols.append(len(rows) % 10)

        self._rows = rows
        self._cols = cols

    def update_wavenumber(self, wavenumber: float):
        """Update map for new wavenumber."""
        self.current_wavenumber = wavenumber
        self.plot()

    def set_colormap(self, cmap_name: str):
        """Set the colormap."""
        self.colormap = cmap_name

    def plot(self):
        """Plot the grid-based hyperspectral map."""
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)

        try:
            self.fig.patch.set_alpha(0.0)
        except Exception:
            pass

        self.ax.set_facecolor("none")

        if not self._rows or not self._cols:
            self.ax.set_title(f"{self._label}: No grid data", color="white")
            self.ax.axis("off")
            self.canvas.draw_idle()
            return

        # Get intensities at current wavenumber
        intensities = self.data_manager.get_intensity_at_wavenumber(
            self.current_wavenumber, self.indices
        )

        # Create grid
        n_rows = max(self._rows) + 1
        n_cols = max(self._cols) + 1
        grid = np.full((n_rows, n_cols), np.nan)

        for i, (row, col) in enumerate(zip(self._rows, self._cols)):
            grid[row, col] = intensities[i]

        # Plot as image
        im = self.ax.imshow(grid, cmap=self.colormap, aspect='equal', origin='lower')

        self.ax.set_title(f"{self._label} @ {self.current_wavenumber:.1f} cm^-1")
        self.ax.set_xlabel("Column")
        self.ax.set_ylabel("Row")

        # Add colorbar
        cbar = self.fig.colorbar(im, ax=self.ax, shrink=0.8)
        cbar.set_label("Intensity", color="white")
        cbar.ax.tick_params(colors="white")
        for spine in cbar.ax.spines.values():
            spine.set_color("white")

        style_dark_axes(self.ax)

        self.fig.tight_layout()
        self.canvas.draw_idle()
