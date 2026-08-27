"""Scatter plot widget for embedding visualization."""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any, Callable
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSizePolicy, QPushButton
from PyQt5.QtCore import pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D
from matplotlib import colors, cm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from ..utils.dark_theme import style_dark_axes, style_dark_3d_axes, make_mpl_transparent, DEFAULT_BLUE


class ScatterPlotWidget(QWidget):
    """Widget for displaying embedding scatter plots with lasso selection."""

    # Signals
    selection_changed = pyqtSignal(list)  # Emits list of selected indices
    point_clicked = pyqtSignal(int)  # Emits clicked point index

    def __init__(self, parent=None):
        super().__init__(parent)

        self.embedding_result: Optional[np.ndarray] = None
        self.visible_indices: List[int] = []
        self.selected_indices: List[int] = []
        self.index_to_scatter_pos: Dict[int, int] = {}

        self.embedding_dim = 2
        self.embedding_method = "PCA"
        self.color_attr = "None"
        self.cmap_name = "tab10"

        self.scatter = None
        self.scatter_selected = None
        self.scatter_cbar = None
        self.lasso = None
        self.lasso_enabled = False

        self._pan_active = False
        self._pan_press_event = None

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.fig = Figure(figsize=(5, 5))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)

        make_mpl_transparent(self.fig, self.canvas)

        self.canvas.mpl_connect('scroll_event', self._on_scroll)
        self.canvas.mpl_connect('button_press_event', self._on_pan_press)
        self.canvas.mpl_connect('motion_notify_event', self._on_pan_motion)
        self.canvas.mpl_connect('button_release_event', self._on_pan_release)
        self.canvas.mpl_connect('pick_event', self._on_pick)

        self.canvas.setMinimumSize(400, 400)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout.addWidget(self.canvas)

        # Snapshot overlay button
        self._snapshot_btn = QPushButton("\u2922", self.canvas)
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
        from .hyperspectral_map import MapSnapshotDialog
        title = f"{self.embedding_method} Embedding"
        dlg = MapSnapshotDialog(self.fig, title=title, parent=self.window())
        dlg.show()

    def set_embedding(self, embedding: np.ndarray, dim: int = 2, method: str = "PCA"):
        """Set the embedding data."""
        self.embedding_result = embedding
        self.embedding_dim = dim
        self.embedding_method = method

    def set_visible_indices(self, indices: List[int]):
        """Set which indices are currently visible (filtered)."""
        self.visible_indices = indices
        self.index_to_scatter_pos = {idx: pos for pos, idx in enumerate(indices)}

    def set_color_options(self, color_attr: str, cmap_name: str = "tab10"):
        """Set color mapping options."""
        self.color_attr = color_attr
        self.cmap_name = cmap_name

    def enable_lasso(self, enabled: bool):
        """Enable or disable lasso selection."""
        self.lasso_enabled = enabled
        self._setup_lasso()

    def plot(self, data_objects: list, color_attr: str = "None", cmap_name: str = "tab10"):
        """
        Plot the embedding scatter.

        Args:
            data_objects: List of Spectrum objects for metadata access
            color_attr: Metadata attribute to color by, or "None"
            cmap_name: Matplotlib colormap name
        """
        self.color_attr = color_attr
        self.cmap_name = cmap_name

        # Clear and recreate axes
        self.fig.clear()
        if self.embedding_dim == 2:
            self.ax = self.fig.add_subplot(111)
        else:
            self.ax = self.fig.add_subplot(111, projection='3d')
            style_dark_3d_axes(self.ax)

        try:
            self.ax.set_box_aspect(1)
        except Exception:
            self.ax.set_aspect('equal', adjustable='box')

        self.ax.set_facecolor("none")
        style_dark_axes(self.ax)
        self.fig.patch.set_alpha(0.0)

        self.scatter_cbar = None

        # Check for valid embedding
        if self.embedding_result is None or len(self.embedding_result) == 0:
            self._show_message("No embedding data available.")
            self.canvas.draw_idle()
            return

        emb = self.embedding_result
        indices = self.visible_indices if self.visible_indices else list(range(len(emb)))

        if len(indices) == 0:
            self._show_message("No points match the current filter.")
            self.canvas.draw_idle()
            return

        # Update index mapping
        self.index_to_scatter_pos = {idx: pos for pos, idx in enumerate(indices)}

        # Calculate bounds
        x_min, x_max, y_min, y_max, z_min, z_max = self._calculate_bounds(emb)

        # Prepare colors
        colors_array, cmap_to_use, unique_vals = self._prepare_colors(
            data_objects, indices, color_attr, cmap_name
        )

        # Plot scatter
        self._plot_scatter(emb, indices, colors_array, cmap_to_use,
                          x_min, x_max, y_min, y_max, z_min, z_max, len(data_objects))

        # Add colorbar if needed
        if color_attr != "None" and cmap_to_use is not None and len(unique_vals) > 0:
            self._add_colorbar(cmap_to_use, unique_vals, color_attr)

        # Setup lasso selector
        self._setup_lasso()

        self.canvas.draw_idle()

    def _calculate_bounds(self, emb: np.ndarray):
        """Calculate axis bounds with margins."""
        mask = np.isfinite(emb[:, 0]) & np.isfinite(emb[:, 1])
        emb_valid = emb[mask]

        if emb_valid.size == 0:
            return -1, 1, -1, 1, None, None

        x_all = emb_valid[:, 0]
        y_all = emb_valid[:, 1]

        def with_margin(vmin, vmax):
            if vmin == vmax:
                return vmin - 1, vmax + 1
            span = vmax - vmin
            pad = 0.05 * span
            return vmin - pad, vmax + pad

        x_min, x_max = with_margin(float(x_all.min()), float(x_all.max()))
        y_min, y_max = with_margin(float(y_all.min()), float(y_all.max()))

        z_min = z_max = None
        if self.embedding_dim == 3 and emb.shape[1] >= 3:
            z_all = emb[:, 2]
            z_valid = z_all[np.isfinite(z_all)]
            if z_valid.size > 0:
                z_min, z_max = with_margin(float(z_valid.min()), float(z_valid.max()))

        return x_min, x_max, y_min, y_max, z_min, z_max

    def _prepare_colors(self, data_objects, indices, color_attr, cmap_name):
        """Prepare color array for scatter plot."""
        if color_attr == "None":
            return DEFAULT_BLUE, None, []

        values = [data_objects[i].metadata.get(color_attr, None) for i in indices]

        # Filter out None and NaN
        valid_vals = []
        for v in values:
            if v is None:
                continue
            try:
                if pd.isna(v):
                    continue
            except (TypeError, ValueError):
                pass
            valid_vals.append(v)

        try:
            unique_vals = sorted(set(valid_vals))
        except TypeError:
            unique_vals = sorted(set(str(v) for v in valid_vals))

        if len(unique_vals) == 0:
            return DEFAULT_BLUE, None, []

        color_map = {val: idx for idx, val in enumerate(unique_vals)}
        colors_array = []
        for v in values:
            if v is not None and not (isinstance(v, float) and pd.isna(v)):
                colors_array.append(color_map.get(v, -1))
            else:
                colors_array.append(-1)

        return colors_array, cmap_name, unique_vals

    def _plot_scatter(self, emb, indices, colors_array, cmap_to_use,
                     x_min, x_max, y_min, y_max, z_min, z_max, total_count):
        """Plot the actual scatter points."""
        if self.embedding_dim == 2:
            coords = emb[indices, :2]
            self.scatter = self.ax.scatter(
                coords[:, 0], coords[:, 1],
                c=colors_array, cmap=cmap_to_use,
                picker=5, zorder=1
            )
            self.scatter_selected = self.ax.scatter(
                [], [], marker='*', s=120,
                edgecolors='white', facecolors='none',
                linewidths=1.0, zorder=10, picker=False
            )
            self.ax.set_xlabel(f"{self.embedding_method} 1")
            self.ax.set_ylabel(f"{self.embedding_method} 2")
            self.ax.set_xlim(x_min, x_max)
            self.ax.set_ylim(y_min, y_max)
        else:
            coords = emb[indices, :3]
            self.scatter = self.ax.scatter(
                coords[:, 0], coords[:, 1], coords[:, 2],
                c=colors_array, cmap=cmap_to_use,
                picker=10, zorder=1
            )
            self.scatter_selected = self.ax.scatter(
                [], [], [], marker='*', s=120,
                edgecolors='white', facecolors='none',
                linewidths=5.0, zorder=10, picker=False
            )
            self.ax.set_xlabel(f"{self.embedding_method} 1")
            self.ax.set_ylabel(f"{self.embedding_method} 2")
            self.ax.set_zlabel(f"{self.embedding_method} 3")
            self.ax.zaxis.label.set_color("white")
            self.ax.set_xlim(x_min, x_max)
            self.ax.set_ylim(y_min, y_max)
            if z_min is not None and z_max is not None:
                self.ax.set_zlim(z_min, z_max)

        suffix = "" if len(indices) == total_count else f" (filtered: {len(indices)}/{total_count})"
        self.ax.set_title(f"{self.embedding_method}{suffix}")

    def _add_colorbar(self, cmap_name, unique_vals, color_attr):
        """Add colorbar for categorical coloring."""
        n_unique = len(unique_vals)
        cmap_obj = plt.get_cmap(cmap_name)
        norm = colors.Normalize(vmin=0, vmax=max(n_unique - 1, 1))
        sm = cm.ScalarMappable(norm=norm, cmap=cmap_obj)

        cax = inset_axes(
            self.ax,
            width="94%",
            height="7%",
            loc="lower center",
            bbox_to_anchor=(0.03, -0.22, 0.94, 1.0),
            bbox_transform=self.ax.transAxes,
            borderpad=0
        )

        self.scatter_cbar = self.fig.colorbar(sm, cax=cax, orientation="horizontal")
        self.scatter_cbar.set_ticks(list(range(n_unique)))
        self.scatter_cbar.set_ticklabels([str(v) for v in unique_vals])

        self.scatter_cbar.ax.tick_params(colors="white", labelsize=8)
        for t in self.scatter_cbar.ax.get_xticklabels():
            t.set_rotation(45)
            t.set_ha("right")

        self.scatter_cbar.set_label(color_attr, color="white", fontsize=9, labelpad=6)

        for spine in self.scatter_cbar.ax.spines.values():
            spine.set_color("white")

        self.fig.subplots_adjust(bottom=0.22)

    def _show_message(self, msg: str):
        """Show centered message on the plot."""
        if hasattr(self.ax, 'name') and self.ax.name == "3d":
            self.ax.text2D(0.5, 0.5, msg, ha="center", va="center",
                          transform=self.ax.transAxes, color="white")
        else:
            self.ax.text(0.5, 0.5, msg, ha="center", va="center",
                        transform=self.ax.transAxes, color="white")

    def _setup_lasso(self):
        """Setup lasso selector."""
        if self.lasso is not None:
            try:
                self.lasso.disconnect_events()
            except Exception:
                pass
            self.lasso = None

        if self.embedding_dim == 2 and self.lasso_enabled:
            self.lasso = LassoSelector(
                self.ax,
                self._on_lasso_select,
                useblit=True,
                props=dict(color="white", linewidth=1.5, alpha=0.95)
            )

    def _on_lasso_select(self, verts):
        """Handle lasso selection."""
        if self.embedding_dim != 2 or self.embedding_result is None:
            return

        indices = self.visible_indices if self.visible_indices else list(range(len(self.embedding_result)))
        if not indices:
            return

        path = Path(verts)
        points = self.embedding_result[indices, :2]
        selected_mask = path.contains_points(points)
        self.selected_indices = [indices[i] for i, s in enumerate(selected_mask) if s]

        self.selection_changed.emit(self.selected_indices)

    def update_highlight(self, selected_indices: List[int]):
        """Update highlight overlay for selected points."""
        if self.scatter_selected is None or self.embedding_result is None:
            return

        if self.embedding_dim == 2:
            coords = []
            for idx in selected_indices:
                if 0 <= idx < len(self.embedding_result):
                    coords.append(self.embedding_result[idx, :2])

            if coords:
                arr = np.vstack(coords)
                self.scatter_selected.set_offsets(arr)
            else:
                self.scatter_selected.set_offsets(np.empty((0, 2)))

        self.canvas.draw_idle()

    def update_markers(self, selected_indices: List[int]):
        """Update scatter markers: stars for selected, circles for others."""
        if self.scatter is None:
            return

        indices = self.visible_indices if self.visible_indices else []
        if not indices:
            return

        circle = MarkerStyle("o").get_path().transformed(
            MarkerStyle("o").get_transform()
        )
        star_style = MarkerStyle("*")
        base_path = star_style.get_path()
        base_transform = star_style.get_transform()
        scaled_transform = base_transform + Affine2D().scale(2.0)
        star = base_path.transformed(scaled_transform)

        paths = []
        edgecolors = []
        for idx in indices:
            if idx in selected_indices:
                paths.append(star)
                edgecolors.append('white')
            else:
                paths.append(circle)
                edgecolors.append('none')

        self.scatter.set_paths(paths)
        self.scatter.set_edgecolors(edgecolors)
        self.scatter.set_linewidths(2)
        self.canvas.draw_idle()

    def get_scatter_colors(self) -> np.ndarray:
        """Get the current scatter point colors."""
        if self.scatter is not None:
            try:
                return self.scatter.get_facecolors()
            except Exception:
                pass
        return None

    def _on_scroll(self, event):
        """Handle scroll zoom."""
        if event.inaxes is None:
            return

        if self.embedding_dim == 2:
            cur_xlim = self.ax.get_xlim()
            cur_ylim = self.ax.get_ylim()
            center_x = (cur_xlim[0] + cur_xlim[1]) / 2
            center_y = (cur_ylim[0] + cur_ylim[1]) / 2
            scale_factor = 0.9 if event.button == 'up' else 1.1
            new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
            new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
            self.ax.set_xlim([center_x - new_width/2, center_x + new_width/2])
            self.ax.set_ylim([center_y - new_height/2, center_y + new_height/2])
        else:
            if hasattr(self.ax, 'dist'):
                if event.button == 'up':
                    self.ax.dist *= 0.9
                elif event.button == 'down':
                    self.ax.dist *= 1.1

        self.canvas.draw_idle()

    def _on_pan_press(self, event):
        """Handle pan start."""
        if self.embedding_dim != 2:
            return
        if self.lasso_enabled:
            return
        if event.inaxes != self.ax:
            return
        if event.button != 1:
            return
        self._pan_active = True
        self._pan_press_event = event

    def _on_pan_motion(self, event):
        """Handle pan motion."""
        if not self._pan_active or event.inaxes != self.ax:
            return
        dx = event.xdata - self._pan_press_event.xdata
        dy = event.ydata - self._pan_press_event.ydata
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        self.ax.set_xlim(cur_xlim[0] - dx, cur_xlim[1] - dx)
        self.ax.set_ylim(cur_ylim[0] - dy, cur_ylim[1] - dy)
        self._pan_press_event = event
        self.canvas.draw_idle()

    def _on_pan_release(self, event):
        """Handle pan release."""
        self._pan_active = False
        self._pan_press_event = None

    def _on_pick(self, event):
        """Handle point picking."""
        if event.artist != self.scatter:
            return
        ind = event.ind
        if len(ind) > 0:
            indices = self.visible_indices if self.visible_indices else list(range(len(self.embedding_result)))
            if ind[0] < len(indices):
                self.point_clicked.emit(indices[ind[0]])

    def clear(self):
        """Clear the scatter plot."""
        self.embedding_result = None
        self.visible_indices = []
        self.selected_indices = []
        self.index_to_scatter_pos = {}
        self.scatter = None
        self.scatter_selected = None
        self.scatter_cbar = None

        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("none")
        style_dark_axes(self.ax)
        self.fig.patch.set_alpha(0.0)
        self._show_message("No data loaded")
        self.canvas.draw_idle()
