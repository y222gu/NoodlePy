"""Spectrum line plot widget for displaying selected spectra."""

import numpy as np
from typing import List, Dict, Optional, Any
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSizePolicy, QPushButton
from PyQt5.QtCore import pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ..utils.dark_theme import style_dark_axes, make_mpl_transparent, DEFAULT_BLUE


class SpectrumPlotWidget(QWidget):
    """Widget for displaying spectra line plots."""

    # Signals
    spectrum_clicked = pyqtSignal(int)  # Emits clicked spectrum index
    spectrum_hovered = pyqtSignal(int)  # Emits hovered spectrum index

    def __init__(self, parent=None):
        super().__init__(parent)

        self.selected_indices: List[int] = []
        self.selected_index_to_line: Dict[int, Any] = {}
        self.hovered_index: Optional[int] = None
        self.top_index: Optional[int] = None
        self.plot_mode = 'combined'  # 'combined' or 'grouped'

        self._spectra_xlim: tuple = (0, 1)
        self._shared_left: float = 0.10
        self._shared_right: float = 0.985

        self._wavenumber_lines = []
        self._current_wavenumber: Optional[float] = None

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.fig = Figure(figsize=(12, 4))
        self.canvas = FigureCanvas(self.fig)

        make_mpl_transparent(self.fig, self.canvas)

        self.canvas.mpl_connect('motion_notify_event', self._on_hover)
        self.canvas.mpl_connect('pick_event', self._on_pick)
        self.canvas.mpl_connect('button_press_event', self._on_click)

        self.canvas.setMinimumHeight(350)
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
        dlg = MapSnapshotDialog(self.fig, title="Spectra Snapshot", parent=self.window())
        dlg.show()

    def set_plot_mode(self, mode: str):
        """Set plot mode ('combined' or 'grouped')."""
        self.plot_mode = mode

    def plot(self, data_objects: list, indices: List[int],
             scatter_colors: np.ndarray = None,
             index_to_scatter_pos: Dict[int, int] = None,
             group_by: str = 'patient'):
        """
        Plot spectra for selected indices.

        Args:
            data_objects: List of Spectrum objects
            indices: List of indices to plot
            scatter_colors: Optional color array from scatter plot
            index_to_scatter_pos: Mapping from global index to scatter position
            group_by: Metadata key to group by in grouped mode
        """
        self.fig.clear()
        self.selected_indices = indices
        self.selected_index_to_line = {}

        try:
            self.fig.patch.set_alpha(0.0)
        except Exception:
            pass

        if not indices:
            ax = self.fig.add_subplot(111)
            ax.set_title("Select points to view spectra", color='white')
            ax.axis('off')
            self.canvas.draw_idle()
            return

        # Get x-axis range from first spectrum
        x = data_objects[indices[0]].raman_shift_cm
        xmin, xmax = float(np.min(x)), float(np.max(x))
        self._spectra_xlim = (xmin, xmax)

        def pick_color(idx, fallback):
            if scatter_colors is not None and len(scatter_colors) > 0 and index_to_scatter_pos:
                pos = index_to_scatter_pos.get(idx)
                if pos is not None and pos < len(scatter_colors):
                    return scatter_colors[pos]
            return fallback

        if self.plot_mode == 'combined':
            self._plot_combined(data_objects, indices, pick_color, xmin, xmax)
        else:
            self._plot_grouped(data_objects, indices, pick_color, xmin, xmax, group_by)

        self._update_line_zorder()
        self._draw_wavenumber_line()

        # Set margins
        self.fig.subplots_adjust(left=0.10, right=0.985, top=0.95, bottom=0.10, hspace=0.35)
        self.canvas.draw()

        sp = self.fig.subplotpars
        self._shared_left = sp.left
        self._shared_right = sp.right

    def _plot_combined(self, data_objects, indices, pick_color, xmin, xmax):
        """Plot all spectra in a single combined view."""
        ax = self.fig.add_subplot(111)
        ax.set_facecolor("none")

        for idx in indices:
            obj = data_objects[idx]
            c = pick_color(idx, DEFAULT_BLUE)

            line, = ax.plot(obj.raman_shift_cm, obj.intensity, color=c, alpha=0.7, picker=5)
            self.selected_index_to_line[idx] = line

        ax.set_title(f"All Selected Spectra ({len(indices)} total)")
        ax.set_xlabel("Raman Shift (cm^-1)")
        ax.set_ylabel("Intensity")
        style_dark_axes(ax)

        ax.set_xlim(xmin, xmax)
        ax.margins(x=0)
        ax.set_autoscalex_on(False)

    def _plot_grouped(self, data_objects, indices, pick_color, xmin, xmax, group_by):
        """Plot spectra grouped by metadata key."""
        groups = {}
        for idx in indices:
            key = data_objects[idx].metadata.get(group_by, "Unknown")
            groups.setdefault(key, []).append(idx)

        n_groups = len(groups)

        for plot_idx, (key, group_indices) in enumerate(sorted(groups.items())):
            ax = self.fig.add_subplot(n_groups, 1, plot_idx + 1)
            ax.set_facecolor("none")

            for idx in group_indices:
                obj = data_objects[idx]
                c = pick_color(idx, "white")

                line, = ax.plot(obj.raman_shift_cm, obj.intensity, color=c, alpha=0.7, picker=5)
                self.selected_index_to_line[idx] = line

            ax.set_title(f"{group_by.title()} {key} ({len(group_indices)} spectra)", fontsize=9, pad=3)
            ax.set_xlabel("Raman Shift (cm^-1)", fontsize=8)
            ax.set_ylabel("Intensity", fontsize=8)
            ax.tick_params(labelsize=7)
            style_dark_axes(ax)

            ax.set_xlim(xmin, xmax)
            ax.margins(x=0)
            ax.set_autoscalex_on(False)

            if plot_idx < n_groups - 1:
                ax.set_xlabel("")

    def update_highlights(self, highlighted_indices: List[int],
                         scatter_colors: np.ndarray = None,
                         index_to_scatter_pos: Dict[int, int] = None):
        """Update line highlights based on selection."""
        for idx, line in self.selected_index_to_line.items():
            if idx == self.hovered_index or idx in highlighted_indices:
                line.set_linewidth(2.5)
                line.set_color('white')
                line.set_alpha(1.0)
            else:
                line.set_linewidth(1)

                if scatter_colors is not None and len(scatter_colors) > 0 and index_to_scatter_pos:
                    pos = index_to_scatter_pos.get(idx)
                    if pos is not None and pos < len(scatter_colors):
                        line.set_color(scatter_colors[pos])
                    else:
                        line.set_color(DEFAULT_BLUE)
                else:
                    line.set_color(DEFAULT_BLUE)

                line.set_alpha(0.7)

        # Update z-order
        primary_idx = None
        if highlighted_indices:
            primary_idx = highlighted_indices[0]
        elif self.hovered_index is not None:
            primary_idx = self.hovered_index

        self._update_line_zorder(primary_idx)
        self.canvas.draw_idle()

    def _update_line_zorder(self, primary_idx: int = None):
        """Ensure one spectrum line is drawn on top."""
        if primary_idx is not None:
            self.top_index = primary_idx

        if not self.selected_index_to_line:
            return

        base_z = 1
        top_z = 10

        if self.top_index not in self.selected_index_to_line:
            self.top_index = None

        for idx, line in self.selected_index_to_line.items():
            if idx == self.top_index:
                line.set_zorder(top_z)
            else:
                line.set_zorder(base_z)

    def get_xlim(self) -> tuple:
        """Get current x-axis limits."""
        return self._spectra_xlim

    def get_margins(self) -> tuple:
        """Get left and right margins for alignment."""
        return self._shared_left, self._shared_right

    def set_wavenumber_line(self, wavenumber: Optional[float]):
        """Draw or update a vertical dashed line at the specified wavenumber."""
        self._current_wavenumber = wavenumber
        self._draw_wavenumber_line()
        self.canvas.draw_idle()

    def _draw_wavenumber_line(self):
        """Draw vertical dashed line at current wavenumber on all axes."""
        for line in self._wavenumber_lines:
            try:
                line.remove()
            except Exception:
                pass
        self._wavenumber_lines = []

        if self._current_wavenumber is None:
            return

        for ax in self.fig.get_axes():
            line = ax.axvline(
                self._current_wavenumber, color='yellow',
                linestyle='--', linewidth=1.0, alpha=0.7, zorder=5
            )
            self._wavenumber_lines.append(line)

    def _on_hover(self, event):
        """Handle hover events."""
        for ax in self.fig.get_axes():
            if event.inaxes == ax:
                for idx, line in self.selected_index_to_line.items():
                    if line.axes == ax:
                        contains, _ = line.contains(event)
                        if contains:
                            if self.hovered_index != idx:
                                self.hovered_index = idx
                                self.spectrum_hovered.emit(idx)
                            return

        if self.hovered_index is not None:
            self.hovered_index = None
            self.spectrum_hovered.emit(-1)

    def _on_pick(self, event):
        """Handle pick events on lines."""
        if event.artist not in self.selected_index_to_line.values():
            return

        idx = None
        for k, v in self.selected_index_to_line.items():
            if v is event.artist:
                idx = k
                break

        if idx is not None:
            self._update_line_zorder(idx)
            self.spectrum_clicked.emit(idx)

    def _on_click(self, event):
        """Handle click events."""
        if event.inaxes is None:
            return

        clicked_on_line = False
        for idx, line in self.selected_index_to_line.items():
            if line.axes == event.inaxes:
                contains, _ = line.contains(event)
                if contains:
                    clicked_on_line = True
                    break

        if not clicked_on_line:
            self.hovered_index = None
            self.spectrum_clicked.emit(-1)

    def clear(self):
        """Clear the plot."""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_title("Select points to view spectra", color='white')
        ax.axis('off')
        self.selected_index_to_line = {}
        self.canvas.draw_idle()
