import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # optional, just for pretty heatmaps
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)
from matplotlib import animation
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import pandas as pd
import os

# ---------- CONFIG ----------
DATA_DIR = Path(r"/Users/yifeigu/Downloads/2025_12_08/metadata")

# Example patterns:
# 1) patient_433_staging_control_sample_0_point_0_line_1_ring_1_rep_8_x-81.98_y-1087.11_metadata.txt

# ---------- HELPERS ----------

def parse_filename(filename: str) -> dict:
    """
    Parse metadata encoded in the filename.
    
    Generalized pattern that extracts key-value pairs from filenames like:
    "patient_538_staging_cancer_sample_0_point_17_rep_20_line_1_ring_9_x72.93_y-130.50.txt"
    
    Extracts: patient, staging, sample, point, rep, line, ring, x, y
    """
    name = os.path.basename(filename).replace('.txt', '')
    
    # Use regex to extract all key-value pairs
    # Match a word followed by underscore and a value (stops at next underscore+word pattern or end)

    split = name.split('_')

    # remove the last three
    split = split[:-3]

    # every two elements are a key-value pair
    matches = [(split[i], split[i + 1]) for i in range(0, len(split), 2)]

    d = {}
    for key, value in matches:
        key_lower = key.lower()
        
        # Try to convert to appropriate type
        try:
            # Check if it's a float (contains decimal point or negative)
            if '.' in value or (value.startswith('-') and value[1:].replace('.', '').isdigit()):
                d[key_lower] = float(value)
            # Check if it's an integer
            elif value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                d[key_lower] = int(value)
            else:
                # Keep as string
                d[key_lower] = value
        except ValueError:
            d[key_lower] = value
    
    # Ensure required fields exist
    required_fields = ['patient', 'staging', 'sample', 'point', 'rep', 'ring']
    for field in required_fields:
        if field not in d:
            raise ValueError(f"Required field '{field}' not found in filename: {filename}")
    
    # If line is not present, set it equal to point
    if 'line' not in d:
        d['line'] = d['point']
    
    return d


def normalize_key(key: str) -> str:
    """
    Normalize metadata keys to lowercase with underscores.
    
    Examples:
        'Prusa_position' -> 'prusa_position'
        'Integration time (ms)' -> 'integration_time_ms'
    """
    key = key.strip().lower()
    key = key.replace("(", "_").replace(")", "")
    key = key.replace(" ", "_")
    key = re.sub(r'_+', '_', key)  # Replace multiple underscores with single
    return key.strip('_')  # Remove leading/trailing underscores

def parse_metadata_file(path: Path) -> dict:
    """
    Parse the contents of a metadata file like:
      Number_of_repetitions = 3
      Integration_time = 5000
      Laser_power = 450
      Prusa_position = 13.02
      Nanodrive_position = 64.0
      Timestamp = 19:06:27
    or with ':' instead of '='.
    """
    meta = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Handle both  "key: value"  and  "key = value"  styles
            if ":" in line:
                key, value = line.split(":", 1)
            elif "=" in line:
                key, value = line.split("=", 1)
            else:
                continue

            key = normalize_key(key)
            value = value.strip().strip(",")

            # Try to convert to int or float
            try:
                if any(c in value for c in (".", "e", "E")):
                    num = float(value)
                    if num.is_integer():
                        num = int(num)
                    meta[key] = num
                else:
                    meta[key] = int(value)
            except ValueError:
                meta[key] = value  # keep as string if not numeric

    return meta


def load_all_metadata(data_dir: Path) -> pd.DataFrame:
    """
    Load metadata from all *_metadata.txt files in data_dir.
    Returns a pandas DataFrame with one row per file and a 'focus_position' column.
    """
    rows = []

    files = sorted(data_dir.glob("*_metadata.txt"))
    if not files:
        raise RuntimeError(f"No *_metadata.txt files found in {data_dir}")

    for path in files:
        fn_meta = parse_filename(path)
        file_meta = parse_metadata_file(path)

        row = {**fn_meta, **file_meta}

        # Look for prusa/nanodrive with or without unit suffixes
        prusa = (
            row.get("prusa_position_mm")
            or row.get("prusa_position")
        )
        nano = (
            row.get("nanodrive_position_um")
            or row.get("nanodrive_position")
        )

        if prusa is None or nano is None:
            raise KeyError(
                f"Missing prusa/nanodrive position in {path}. "
                f"Available keys: {list(row.keys())}"
            )

        # focus_position = prusa_position - 0.001 * nanodrive_position
        row["focus_position"] = prusa - 0.001 * nano

        rows.append(row)

    df = pd.DataFrame(rows)
    return df

def add_relative_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add relative timestamps based on the *latest* file time per rep.

    Definitions:
      - For each rep r, let T_max(r) = latest timestamp among all rows with that rep.
      - Let T0 = T_max(1) if rep 1 exists, otherwise T_max(min_rep).

    This function computes:
      - df['timestamp_rel_min']:
          per-row relative time (minutes) = (timestamp_row - T0)
      - df['rep_timestamp_rel_min']:
          per-rep relative time (minutes) = (T_max(rep) - T0)

    Works for:
      - full datetime strings, e.g. "2025-12-06 20:07:19"
      - time-only strings, e.g. "20:07:19"
      - already-parsed datetime objects
    """
    if "timestamp" not in df.columns:
        raise KeyError("No 'timestamp' column found in dataframe.")

    # Let pandas infer the format (handles both HH:MM:SS and full datetimes)
    ts = pd.to_datetime(df["timestamp"], errors="coerce")

    if ts.isna().any():
        bad = df.loc[ts.isna(), "timestamp"].unique()
        raise ValueError(
            "Some timestamps could not be parsed. Examples:\n"
            + "\n".join(map(str, bad[:10]))
        )

    df = df.copy()
    df["timestamp_parsed"] = ts  # optional, but handy for debugging

    # Latest timestamp per rep: T_max(rep)
    rep_max_ts = df.groupby("rep")["timestamp_parsed"].max()

    # Baseline: latest timestamp of rep 1 if it exists, else smallest rep
    if 1 in rep_max_ts.index:
        baseline_rep = 1
    else:
        baseline_rep = rep_max_ts.index.min()

    T0 = rep_max_ts.loc[baseline_rep]

    # Per-row relative time to T0
    df["timestamp_rel_min"] = (df["timestamp_parsed"] - T0).dt.total_seconds() / 60.0

    # Per-rep relative time to T0, using T_max(rep)
    rep_rel_min = (rep_max_ts - T0).dt.total_seconds() / 60.0
    df["rep_timestamp_rel_min"] = df["rep"].map(rep_rel_min)

    return df

def plot_focus_heatmaps_by_rep(df: pd.DataFrame):
    """
    For each rep, pivot to a line x ring matrix of focus_position
    and plot a heatmap. All heatmaps share the same colorbar range.
    """

    required_cols = {"line", "ring", "rep", "focus_position"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # Compute global color limits
    vmin = df["focus_position"].min()
    vmax = df["focus_position"].max()
    print(f"Global color scale: vmin={vmin:.4f}, vmax={vmax:.4f}")

    reps = sorted(df["rep"].unique())

    for rep_val in reps:
        df_rep = df[df["rep"] == rep_val]

        pivot = df_rep.pivot(index="line", columns="ring", values="focus_position")

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            pivot,
            annot=True,       # turn off if too crowded
            fmt=".3f",
            cmap="viridis",
            vmin=vmin,        # ← FIXED SCALE
            vmax=vmax,        # ← FIXED SCALE
            cbar_kws={"label": "Focus position"},
        )
        plt.title(f"Focus Position Heatmap (rep = {rep_val})")
        plt.xlabel("Ring")
        plt.ylabel("Line")
        plt.tight_layout()
        plt.show()

from matplotlib.animation import PillowWriter

def make_3d_surface_gif(df: pd.DataFrame,
                        outfile: str = "focus_3d.gif",
                        fps: int = 5,
                        smooth_factor: int = 3):
    """
    Create a smoothed 3D surface animation over (line, ring, focus_position)
    for each rep, and save as an animated GIF (no ffmpeg needed).

    - x-axis: ring
    - y-axis: line
    - z-axis: focus_position
    - frame index: rep
    """

    required_cols = {"line", "ring", "rep", "focus_position"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df = df.copy()
    df["line"] = df["line"].astype(int)
    df["ring"] = df["ring"].astype(int)
    reps = sorted(df["rep"].unique())

    # Use the first rep to define the coarse grid
    first = df[df["rep"] == reps[0]]
    pivot_first = first.pivot(index="line", columns="ring", values="focus_position")
    pivot_first = pivot_first.sort_index().sort_index(axis=1)

    lines = pivot_first.index.values    # y-axis
    rings = pivot_first.columns.values  # x-axis
    Z0 = pivot_first.values

    # Upsample for smooth surface
    Xs, Ys, Zs0 = _upsample_grid(Z0, rings, lines, factor=smooth_factor)

    # Global z/color limits
    vmin = df["focus_position"].min()
    vmax = df["focus_position"].max()

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Initial surface (store in list so we can reassign inside update)
    surf = [ax.plot_surface(
        Xs, Ys, Zs0,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        edgecolor="none",
        antialiased=True,
    )]

    ax.set_xlabel("Ring")
    ax.set_ylabel("Line")
    ax.set_zlabel("Focus position")
    ax.set_zlim(vmin, vmax)
    ax.set_title(f"3D Focus Surface – rep = {reps[0]}")
    ax.view_init(elev=30, azim=-15)  # fixed view

    # Single colorbar, created once
    cbar = fig.colorbar(surf[0], ax=ax, shrink=0.6, label="Focus position")

    def update(frame_idx):
        rep_val = reps[frame_idx]
        df_rep = df[df["rep"] == rep_val]
        pivot = df_rep.pivot(index="line", columns="ring", values="focus_position")
        pivot = pivot.reindex(index=lines, columns=rings)

        Z = pivot.values
        _, _, Zs = _upsample_grid(Z, rings, lines, factor=smooth_factor)

        # Remove old surface
        surf[0].remove()

        # Add new surface
        surf[0] = ax.plot_surface(
            Xs, Ys, Zs,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            edgecolor="none",
            antialiased=True,
        )

        ax.set_title(f"3D Focus Surface – rep = {rep_val}")
        # keep fixed view; no rotation
        return (surf[0],)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(reps),
        blit=False,
        repeat=True,
    )

    writer = PillowWriter(fps=fps)
    ani.save(outfile, writer=writer)
    plt.close(fig)
    print(f"Saved 3D GIF to {outfile}")

import numpy as np

def _upsample_grid(Z, rings, lines, factor=3):
    """
    Upsample a 2D grid Z by 'factor' in both directions using simple
    1D linear interpolation (no external deps besides numpy).

    Z shape: (n_lines, n_rings)
    rings:   array of ring positions (len = n_rings)
    lines:   array of line positions (len = n_lines)
    """

    n_lines, n_rings = Z.shape

    # Original index coordinates
    old_x = np.arange(n_rings)
    old_y = np.arange(n_lines)

    # New (upsampled) index coordinates
    new_x = np.linspace(0, n_rings - 1, factor * n_rings)
    new_y = np.linspace(0, n_lines - 1, factor * n_lines)

    # 1) interpolate along x (ring axis) for each row
    Zx = np.array([np.interp(new_x, old_x, row) for row in Z])

    # 2) interpolate along y (line axis) for each column
    Zxy = np.array(
        [np.interp(new_y, old_y, Zx[:, j]) for j in range(Zx.shape[1])]
    ).T  # shape: (factor*n_lines, factor*n_rings)

    # Map index space back to physical ring/line coordinates
    new_rings = np.interp(new_x, [0, n_rings - 1], [rings.min(), rings.max()])
    new_lines = np.interp(new_y, [0, n_lines - 1], [lines.min(), lines.max()])
    Xs, Ys = np.meshgrid(new_rings, new_lines)

    return Xs, Ys, Zxy


from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def interactive_3d_surface_viewer(df: pd.DataFrame, smooth_factor: int = 3):
    """
    Open a matplotlib window with a 3D surface plot and a slider to
    move through different reps.

    - x-axis: ring
    - y-axis: line
    - z-axis: focus_position
    - slider index: rep

    The camera/view angle is NOT changed automatically when using the slider,
    so any manual rotation you do is preserved as you move through reps.
    """
    required_cols = {"line", "ring", "rep", "focus_position"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df = df.copy()
    df["line"] = df["line"].astype(int)
    df["ring"] = df["ring"].astype(int)
    reps = sorted(df["rep"].unique())

    # If we have rep-level relative times, grab them for titles
    rep_rel_time = None
    if "rep_timestamp_rel_min" in df.columns:
        rep_rel_time = df.groupby("rep")["rep_timestamp_rel_min"].first().to_dict()

    # Use the first rep to define the coarse grid
    first = df[df["rep"] == reps[0]]
    pivot_first = first.pivot(index="line", columns="ring", values="focus_position")
    pivot_first = pivot_first.sort_index().sort_index(axis=1)

    lines = pivot_first.index.values    # y-axis
    rings = pivot_first.columns.values  # x-axis
    Z0 = pivot_first.values

    # Upsample for smooth surface
    Xs, Ys, Zs0 = _upsample_grid(Z0, rings, lines, factor=smooth_factor)

    # Global z/color limits
    vmin = df["focus_position"].min()
    vmax = df["focus_position"].max()

    # --- FIGURE + AXES SETUP ---
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Initial surface
    surf = [ax.plot_surface(
        Xs, Ys, Zs0,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        edgecolor="none",
        antialiased=True,
    )]

    ax.set_xlabel("Ring")
    ax.set_ylabel("Line (Edge → Center)")
    ax.set_zlabel("Focus position")
    ax.set_zlim(vmin, vmax)

    first_rep = reps[0]
    if rep_rel_time is not None:
        tmin = rep_rel_time[first_rep]
        ax.set_title(f"Focus Surface – rep={first_rep}, t={tmin:.2f} min")
    else:
        ax.set_title(f"Focus Surface – rep={first_rep}")

    # Single colorbar
    fig.colorbar(surf[0], ax=ax, shrink=0.6, label="Focus position")

    # Make room at bottom for slider
    plt.subplots_adjust(bottom=0.15)

    # Slider axis: [left, bottom, width, height]
    ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
    slider = Slider(
        ax=ax_slider,
        label="Frame (rep index)",
        valmin=0,
        valmax=len(reps) - 1,
        valinit=0,
        valstep=1,
    )

    def update(val):
        idx = int(slider.val)
        rep_val = reps[idx]

        df_rep = df[df["rep"] == rep_val]
        pivot = df_rep.pivot(index="line", columns="ring", values="focus_position")
        pivot = pivot.reindex(index=lines, columns=rings)

        Z = pivot.values
        _, _, Zs = _upsample_grid(Z, rings, lines, factor=smooth_factor)

        # Remove old surface and draw new one
        surf[0].remove()
        surf[0] = ax.plot_surface(
            Xs, Ys, Zs,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            edgecolor="none",
            antialiased=True,
        )

        # Update title with rep and time (but DO NOT change view)
        if rep_rel_time is not None:
            tmin = rep_rel_time[rep_val]
            ax.set_title(f"Focus Surface – rep={rep_val}, t={tmin:.2f} min")
        else:
            ax.set_title(f"Focus Surface – rep={rep_val}")

        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()

def slider_style_3d_surface_gif(df: pd.DataFrame,
                                outfile: str = "focus_slider.gif",
                                fps: int = 5,
                                smooth_factor: int = 3):
    """
    Render the same surfaces shown by `interactive_3d_surface_viewer` into
    an animated GIF, stepping through reps in order (like moving the slider).
    """
    required = {"line", "ring", "rep", "focus_position"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df = df.copy()
    df["line"] = df["line"].astype(int)
    df["ring"] = df["ring"].astype(int)
    reps = sorted(df["rep"].unique())

    rep_rel_time = None
    if "rep_timestamp_rel_min" in df.columns:
        rep_rel_time = df.groupby("rep")["rep_timestamp_rel_min"].first().to_dict()

    first = df[df["rep"] == reps[0]]
    pivot_first = first.pivot(index="line", columns="ring", values="focus_position")
    pivot_first = pivot_first.sort_index().sort_index(axis=1)

    lines = pivot_first.index.values
    rings = pivot_first.columns.values
    Z0 = pivot_first.values

    Xs, Ys, Zs0 = _upsample_grid(Z0, rings, lines, factor=smooth_factor)

    vmin = df["focus_position"].min()
    vmax = df["focus_position"].max()

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    surf = [ax.plot_surface(
        Xs, Ys, Zs0,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        edgecolor="none",
        antialiased=True,
    )]

    ax.set_xlabel("Ring")
    ax.set_ylabel("Line (Edge → Center)")
    ax.set_zlabel("Focus position")
    ax.set_zlim(5.4, 5.5)
    ax.view_init(elev=30, azim=75)
    fig.colorbar(surf[0], ax=ax, shrink=0.6, label="Focus position")

    def update(frame_idx):
        rep_val = reps[frame_idx]
        df_rep = df[df["rep"] == rep_val]
        pivot = df_rep.pivot(index="line", columns="ring", values="focus_position")
        pivot = pivot.reindex(index=lines, columns=rings)

        Z = pivot.values
        _, _, Zs = _upsample_grid(Z, rings, lines, factor=smooth_factor)

        surf[0].remove()
        surf[0] = ax.plot_surface(
            Xs, Ys, Zs,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            edgecolor="none",
            antialiased=True,
        )

        if rep_rel_time is not None:
            tmin = rep_rel_time[rep_val]
            ax.set_title(f"Focus Surface – rep={rep_val}, t={tmin:.2f} min")
        else:
            ax.set_title(f"Focus Surface – rep={rep_val}")
        return (surf[0],)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(reps),
        blit=False,
        repeat=True,
    )

    writer = PillowWriter(fps=fps)
    ani.save(outfile, writer=writer)
    plt.close(fig)
    print(f"Saved slider-style 3D GIF to {outfile}")

# ---------- MAIN ----------

if __name__ == "__main__":
    df_all = load_all_metadata(DATA_DIR)
    df_all = add_relative_timestamps(df_all)
    print("Loaded metadata from", len(df_all), "files")
    print(df_all.head())

    # plot_focus_heatmaps_by_rep(df_all)

    # 3D contour/surface GIF across reps
    # make_3d_surface_gif(df_all, outfile="focus_3d.gif", fps=5)

    # interactive_3d_surface_viewer(df_all, smooth_factor=3)
    slider_style_3d_surface_gif(df_all, outfile="focus_slider.gif", fps=5)
