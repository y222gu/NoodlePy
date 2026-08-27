"""PCA loadings plot widget."""

import numpy as np
from typing import Optional
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ..utils.dark_theme import style_dark_axes, make_mpl_transparent


class LoadingsPlotWidget(QWidget):
    """Widget for displaying PCA loadings."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.pca_loadings: Optional[np.ndarray] = None
        self.pca_explained_var: Optional[np.ndarray] = None
        self.wavenumber_grid: Optional[np.ndarray] = None
        self.n_pcs_to_show: int = 5

        self._spectra_xlim: tuple = (0, 1)
        self._shared_left: float = 0.10
        self._shared_right: float = 0.985

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.fig = Figure(figsize=(12, 2))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)

        make_mpl_transparent(self.fig, self.canvas)

        # Initial styling
        try:
            self.fig.patch.set_alpha(0.0)
        except Exception:
            pass
        self.ax.set_facecolor("none")
        self.ax.set_title("PCA loadings", color="white")
        self.ax.set_xlabel("Raman Shift (cm^-1)", color="white")
        self.ax.set_ylabel("Loading", color="white")
        self.ax.tick_params(colors="white")
        for spine in self.ax.spines.values():
            spine.set_color("white")
        self.ax.grid(color="white", alpha=0.15)

        self.canvas.setMinimumHeight(150)
        self.canvas.setMaximumHeight(260)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        layout.addWidget(self.canvas)

    def set_pca_data(self, loadings: np.ndarray, explained_var: np.ndarray,
                     wavenumber_grid: np.ndarray, n_pcs: int = 5):
        """
        Set PCA data for plotting.

        Args:
            loadings: PCA loadings array (n_components, n_features)
            explained_var: Explained variance ratio array
            wavenumber_grid: Wavenumber values for x-axis
            n_pcs: Number of PCs to display
        """
        self.pca_loadings = loadings
        self.pca_explained_var = explained_var
        self.wavenumber_grid = wavenumber_grid
        self.n_pcs_to_show = n_pcs

    def set_xlim(self, xlim: tuple):
        """Set x-axis limits to match spectrum plot."""
        self._spectra_xlim = xlim

    def set_margins(self, left: float, right: float):
        """Set left/right margins to match spectrum plot."""
        self._shared_left = left
        self._shared_right = right

    def plot(self):
        """Plot PCA loadings."""
        show = (
            self.pca_loadings is not None
            and self.pca_explained_var is not None
            and self.wavenumber_grid is not None
        )

        if not show:
            self._show_unavailable()
            return

        self.fig.clear()
        try:
            self.fig.patch.set_alpha(0.0)
        except Exception:
            pass

        ax = self.fig.add_subplot(111)
        ax.set_facecolor("none")

        x = self.wavenumber_grid
        n_show = min(int(self.n_pcs_to_show), self.pca_loadings.shape[0])

        for k in range(n_show):
            ev = self.pca_explained_var[k] if k < len(self.pca_explained_var) else None
            label = f"PC{k+1}" + (f" ({ev*100:.1f}%)" if ev is not None else "")
            ax.plot(x, self.pca_loadings[k, :], linewidth=1.0, label=label)

        ax.axhline(0, linewidth=0.8)
        ax.set_title("PCA loadings")
        ax.set_xlabel("Raman Shift (cm^-1)")
        ax.set_ylabel("Loading")
        ax.legend(fontsize=8, ncol=min(n_show, 3))

        style_dark_axes(ax)

        # Force exact same x-range
        xmin, xmax = self._spectra_xlim
        ax.set_xlim(xmin, xmax)
        ax.margins(x=0)
        ax.set_autoscalex_on(False)

        # Match margins with spectra plot
        self.fig.subplots_adjust(left=self._shared_left, right=self._shared_right,
                                 top=0.90, bottom=0.25)

        self.canvas.draw_idle()

    def _show_unavailable(self):
        """Show 'unavailable' message."""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_title("PCA loadings unavailable", color="white")
        ax.axis("off")
        self.canvas.draw_idle()

    def clear(self):
        """Clear the plot."""
        self.pca_loadings = None
        self.pca_explained_var = None
        self._show_unavailable()
