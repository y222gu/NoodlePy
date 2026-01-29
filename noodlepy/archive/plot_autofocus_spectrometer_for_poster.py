import os
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Config
# ----------------------------
folder_path = r'/Users/yifeigu/Downloads/autofocus_spectra/5s'
true_focus_position = 6.10  # optional reference marker

# Optional smoothing (set to 1 to disable)
SMOOTH_WINDOW = 1

# ----------------------------
# Helpers
# ----------------------------
def read_spectrum_csv(path: str):
    """Read 'wavelength,intensity' per line."""
    wl, inten = [], []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            try:
                wl.append(float(parts[0]))
                inten.append(float(parts[1]))
            except ValueError:
                continue
    return np.asarray(wl, dtype=float), np.asarray(inten, dtype=float)

def smooth_moving_average(wl: np.ndarray, inten: np.ndarray, window: int):
    """Simple moving average; keeps arrays aligned."""
    if window <= 1 or len(inten) < window:
        return wl, inten
    kernel = np.ones(window) / window
    sm = np.convolve(inten, kernel, mode="valid")
    wl2 = wl[window // 2 : window // 2 + len(sm)]
    return wl2, sm

def calculate_ratio_score(wavelength: np.ndarray, intensity: np.ndarray) -> float:
    mask = (wavelength >= 790) & (wavelength <= 795)
    reflected_laser = intensity[mask]
    quartz_peak = intensity[(wavelength >= 808) & (wavelength <= 815)]
    denom = np.sum(reflected_laser)
    if denom == 0 or reflected_laser.size == 0 or quartz_peak.size == 0:
        return np.nan
    ratio =  denom / np.sum(quartz_peak)
    return float(ratio)

# ----------------------------
# Load files (sequential positions)
# ----------------------------
files = sorted([f for f in os.listdir(folder_path) if f.endswith(".txt")])

positions = []
ratio_scores = []
spectra = []  # list of (wl, inten, position)

for i, fn in enumerate(files, start=1):
    path = os.path.join(folder_path, fn)

    wl, inten = read_spectrum_csv(path)
    if wl.size == 0 or inten.size == 0:
        positions.append(i)
        ratio_scores.append(np.nan)
        spectra.append((np.array([]), np.array([]), i))
        continue

    wl, inten = smooth_moving_average(wl, inten, SMOOTH_WINDOW)
    score = calculate_ratio_score(wl, inten)

    positions.append(i)
    ratio_scores.append(score)
    spectra.append((wl, inten, i))

positions = np.asarray(positions, dtype=float)
ratio_scores = np.asarray(ratio_scores, dtype=float)

# Predicted focus: choose MIN ratio
predicted_pos = np.nan
if np.isfinite(ratio_scores).any():
    predicted_pos = positions[int(np.nanargmin(ratio_scores))]

# ----------------------------
# Compact stacked spectra plot
# ----------------------------
valid_spectra = [(wl, inten, pos) for (wl, inten, pos) in spectra if wl.size > 0]
n = len(valid_spectra)

if n == 0:
    print("No spectra found to plot.")
else:
    fig, axes = plt.subplots(
        n, 1,
        figsize=(4, max(2.0, 0.6 * n)),
        sharex=True,
        gridspec_kw={"hspace": 0.02}
    )
    if n == 1:
        axes = [axes]

    all_wl = np.concatenate([wl for wl, _, _ in valid_spectra])
    x_min, x_max = float(np.nanmin(all_wl)), float(np.nanmax(all_wl))

    for i, (ax, (wl, inten, pos)) in enumerate(zip(axes, valid_spectra)):
        ax.plot(wl, inten, lw=1.0)
        ax.set_ylim(700, 1700)

        # keep only left spine as y-axis, no ticks
        ax.spines["left"].set_visible(True)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False if i < n - 1 else True)

        ax.set_yticks([])
        ax.tick_params(axis="y", which="both", length=0, labelleft=False)

        if i < n - 1:
            ax.tick_params(labelbottom=False, bottom=False)
        else:
            ax.tick_params(bottom=True)

        ax.set_xlim(x_min, x_max)
        ax.margins(x=0)

    axes[-1].set_xlabel("Raman Shift (cm$^{-1}$)")
    fig.text(0.02, 0.5, "Intensity A.U.", va="center", rotation="vertical")
    fig.suptitle("Stacked Spectra (shared x, y arbitrary)", y=0.995)
    plt.tight_layout()
    plt.savefig("stacked_spectra_compact.svg", dpi=600, bbox_inches="tight")
    plt.show()

# ----------------------------
# Ratio focus score plot
# ----------------------------
fig2, ax2 = plt.subplots(figsize=(6, 4))
ax2.plot(positions, ratio_scores, "ko-", lw=0.5, ms=10)
ax2.set_xlabel("Position")
ax2.set_ylabel("Ratio score: sum(808–815) / sum(790–795)")
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("ratio_focus_score.svg", dpi=600, bbox_inches="tight")
plt.show()

print(f"Plotted {len(files)} files.")
print(f"Predicted focus position (min ratio): {int(predicted_pos) if np.isfinite(predicted_pos) else 'N/A'}")
