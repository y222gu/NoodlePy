"""Dark theme styling for PyQt5 and matplotlib."""

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPalette, QColor


DEFAULT_BLUE = "#496fa7"


def apply_dark_theme(app: QApplication):
    """Apply dark theme to a PyQt5 application."""
    app.setStyle("Fusion")

    dark = QPalette()
    dark.setColor(QPalette.Window, QColor(30, 30, 30))
    dark.setColor(QPalette.WindowText, QColor(220, 220, 220))
    dark.setColor(QPalette.Base, QColor(25, 25, 25))
    dark.setColor(QPalette.AlternateBase, QColor(35, 35, 35))
    dark.setColor(QPalette.ToolTipBase, QColor(220, 220, 220))
    dark.setColor(QPalette.ToolTipText, QColor(220, 220, 220))
    dark.setColor(QPalette.Text, QColor(220, 220, 220))
    dark.setColor(QPalette.Button, QColor(45, 45, 45))
    dark.setColor(QPalette.ButtonText, QColor(220, 220, 220))
    dark.setColor(QPalette.BrightText, QColor(255, 0, 0))
    dark.setColor(QPalette.Highlight, QColor(80, 120, 200))
    dark.setColor(QPalette.HighlightedText, QColor(0, 0, 0))

    app.setPalette(dark)


def style_dark_axes(ax):
    """Make all axis elements white for dark theme."""
    if ax is None:
        return

    # Axis labels & title
    ax.title.set_color("white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")

    # Tick labels
    ax.tick_params(axis='both', colors='white')

    # Axis spines
    for spine in ax.spines.values():
        spine.set_color("white")

    # Grid (if enabled)
    ax.grid(color="white", alpha=0.15)

    # Legend
    leg = ax.get_legend()
    if leg is not None:
        leg.get_frame().set_facecolor("none")
        leg.get_frame().set_edgecolor("white")
        for text in leg.get_texts():
            text.set_color("white")


def style_dark_3d_axes(ax):
    """Dark, semi-transparent 3D box panes + subtle white grid/ticks."""
    # Only apply to 3D axes
    if not hasattr(ax, "get_zlim"):
        return

    pane_rgba = (0.3, 0.3, 0.3, 0.25)  # darker + more transparent
    grid_rgba = (1.0, 1.0, 1.0, 0.12)

    # --- Newer Matplotlib: axis has .pane ---
    for axis in (getattr(ax, "xaxis", None), getattr(ax, "yaxis", None), getattr(ax, "zaxis", None)):
        if axis is None:
            continue

        pane = getattr(axis, "pane", None)
        if pane is not None:
            try:
                pane.set_facecolor(pane_rgba)
            except Exception:
                pass
            try:
                pane.set_edgecolor((1, 1, 1, 0.25))
            except Exception:
                pass

        # Grid color (3D uses _axinfo)
        try:
            axis._axinfo["grid"]["color"] = grid_rgba
        except Exception:
            pass

    # --- Older Matplotlib fallback: w_xaxis / w_yaxis / w_zaxis ---
    for wax in (getattr(ax, "w_xaxis", None), getattr(ax, "w_yaxis", None), getattr(ax, "w_zaxis", None)):
        if wax is None:
            continue
        try:
            wax.set_pane_color(pane_rgba)
        except Exception:
            pass
        try:
            wax._axinfo["grid"]["color"] = grid_rgba
        except Exception:
            pass

    # Labels/ticks white
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.zaxis.label.set_color("white")

    # Axis lines white-ish (best-effort across versions)
    for axis_name in ("xaxis", "yaxis", "zaxis"):
        axis_obj = getattr(ax, axis_name, None)
        line = getattr(axis_obj, "line", None)
        if line is not None:
            try:
                line.set_color((1, 1, 1, 0.6))
            except Exception:
                pass


def make_mpl_transparent(fig, canvas):
    """Make matplotlib figure and canvas transparent."""
    fig.patch.set_alpha(0.0)
    canvas.setStyleSheet("background: transparent;")
    for ax in fig.get_axes():
        ax.set_facecolor("none")
