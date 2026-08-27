"""Temporal stability & Z-focus analysis for Raman spectroscopy data.

Part 1: Temporal stability over 30 consecutive acquisitions (_first30 files).
Part 2: Z-focus dependence across 20 Z positions (_z<N> files).
Part 3: Spatially-aware per-point analysis.
Part 4: Dimensionality reduction (PCA, t-SNE, UMAP) colored by Z.

All spectra are preprocessed before analysis:
  cosmic ray removal → cropping → baseline correction → smoothing → normalization
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import medfilt, savgol_filter
import pybaselines

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = Path("/Users/yifeigu/Documents/Carney_Lab/RamanPy/data/2026_03_05_1/spectra")
OUTPUT_DIR = Path("/Users/yifeigu/Documents/Carney_Lab/RamanPy/analysis/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
N_POINTS_PER_SPECTRUM = 1024
LASER_WAVELENGTH_NM = 785.0

# Biological signature peaks (cm^-1)
CHARACTERISTIC_SHIFTS = [1006, 1156, 1450, 1650]
SHIFT_LABELS = [
    '1006 (Phe)',
    '1156 (Carotenoid)',
    '1450 (CH$_2$ bend)',
    '1650 (Amide I)',
]

# ── Preprocessing config ──────────────────────────────────────────────────────
CROP_START_CM = 662.7
CROP_END_CM = 1784.1
COSMIC_THRESHOLD = 10
BASELINE_LAM = 100
BASELINE_DIFF_ORDER = 1
BASELINE_MAX_ITER = 15
BASELINE_TOL = 0.005
SMOOTH_WINDOW = 5
SMOOTH_POLYORDER = 3

# Filename regex
FNAME_RE = re.compile(
    r'point_(\d+)_rep_(\d+)_line_(\d+)_ring_(\d+)'
    r'_x([+-]?\d+(?:\.\d+)?)_y([+-]?\d+(?:\.\d+)?)'
    r'_(first30|z(\d+))\.txt$'
)


# ══════════════════════════════════════════════════════════════════════════════
# Utility functions
# ══════════════════════════════════════════════════════════════════════════════

def wavelength_to_raman_shift(wavelength_nm):
    return (1.0 / LASER_WAVELENGTH_NM - 1.0 / wavelength_nm) * 1e7


def load_spectra(filepath, n_points=N_POINTS_PER_SPECTRUM):
    data = np.loadtxt(filepath, delimiter=',')
    n_reps = len(data) // n_points
    wavelengths = data[:n_points, 0]
    intensities = data[:, 1].reshape(n_reps, n_points)
    return wavelengths, intensities


def find_nearest_idx(array, value):
    return np.argmin(np.abs(array - value))


def remove_cosmic_rays(intensity, threshold=COSMIC_THRESHOLD):
    median_filtered = medfilt(intensity, kernel_size=5)
    residual = intensity - median_filtered
    std_dev = np.std(residual)
    if std_dev == 0:
        return intensity
    mask = np.abs(residual) > (threshold * std_dev)
    result = intensity.copy()
    result[mask] = median_filtered[mask]
    return result


def preprocess_spectrum(intensity, raman_shifts, normalize=True):
    """Preprocessing: cosmic rays → crop → baseline → smooth → (optional) normalize."""
    result = remove_cosmic_rays(intensity)

    # Crop
    mask = (raman_shifts >= CROP_START_CM) & (raman_shifts <= CROP_END_CM)
    rs = raman_shifts[mask]
    result = result[mask]

    # Baseline correction (airPLS)
    fitter = pybaselines.Baseline(x_data=rs)
    baseline, _ = fitter.airpls(result, BASELINE_LAM, BASELINE_DIFF_ORDER,
                                 BASELINE_MAX_ITER, BASELINE_TOL)
    result = result - baseline

    # Smoothing
    result = savgol_filter(result, SMOOTH_WINDOW, SMOOTH_POLYORDER)

    # Normalize by max (optional)
    if normalize:
        rng = np.max(result) - np.min(result)
        if rng > 0:
            result = (result - np.min(result)) / rng

    return result, rs


def preprocess_batch(intensities_2d, raman_shifts, normalize=True):
    """Preprocess multiple spectra, return (processed_2d, cropped_raman_shifts)."""
    processed = []
    rs_out = None
    for i in range(intensities_2d.shape[0]):
        proc, rs = preprocess_spectrum(intensities_2d[i], raman_shifts, normalize=normalize)
        processed.append(proc)
        if rs_out is None:
            rs_out = rs
    return np.array(processed), rs_out


def setup_dark_figure(nrows=1, ncols=1, figsize=(12, 7)):
    plt.style.use('dark_background')
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=300)
    fig.patch.set_facecolor('black')
    if isinstance(axes, np.ndarray):
        for ax in axes.flat:
            ax.set_facecolor('black')
    else:
        axes.set_facecolor('black')
    return fig, axes


# ══════════════════════════════════════════════════════════════════════════════
# Load & preprocess data
# ══════════════════════════════════════════════════════════════════════════════
print("Scanning spectra files...")
first30_files = sorted(DATA_DIR.glob("*_first30.txt"))
z_files = sorted(DATA_DIR.glob("*_z[0-9]*.txt"))
print(f"  Found {len(first30_files)} first30 files, {len(z_files)} z-focus files")

# ── Load raw wavelength axis ──────────────────────────────────────────────────
sample_wl, _ = load_spectra(first30_files[0])
raw_raman_shifts = wavelength_to_raman_shift(sample_wl)

# ── Load & preprocess first30 data ───────────────────────────────────────────
print("\nLoading & preprocessing first30 spectra...")
first30_data = {}      # point_id -> preprocessed[30, n_cropped]
first30_data_raw = {}  # point_id -> raw[30, 1024]  (for comparison)
point_coords = {}
point_ring = {}
proc_raman_shifts = None

for i, f in enumerate(first30_files):
    m = FNAME_RE.search(f.name)
    if not m:
        continue
    point_id = int(m.group(1))
    ring = int(m.group(4))
    x, y = float(m.group(5)), float(m.group(6))
    _, intensities = load_spectra(f)

    first30_data_raw[point_id] = intensities
    proc, rs = preprocess_batch(intensities, raw_raman_shifts)
    first30_data[point_id] = proc
    point_coords[point_id] = (x, y)
    point_ring[point_id] = ring
    if proc_raman_shifts is None:
        proc_raman_shifts = rs

    if (i + 1) % 20 == 0:
        print(f"  {i + 1}/{len(first30_files)}...")

print(f"  Loaded {len(first30_data)} points")
print(f"  Preprocessed Raman shift range: {proc_raman_shifts.min():.1f} - {proc_raman_shifts.max():.1f} cm^-1")

# Compute distance from centroid
all_points = sorted(first30_data.keys())
all_xy = np.array([point_coords[p] for p in all_points])
centroid = all_xy.mean(axis=0)
point_dist = {p: np.sqrt((x - centroid[0])**2 + (y - centroid[1])**2)
              for p, (x, y) in point_coords.items()}
max_dist = max(point_dist.values())
print(f"  Centroid: ({centroid[0]:.1f}, {centroid[1]:.1f}), max dist: {max_dist:.1f} µm")

# ── Load & preprocess z-focus data ────────────────────────────────────────────
print("\nLoading & preprocessing z-focus spectra...")
z_data = {}          # point_id -> {z_val: preprocessed+normalized[n_cropped]}
z_data_raw = {}      # point_id -> {z_val: raw_median[1024]}
z_data_unnorm = {}   # point_id -> {z_val: preprocessed WITHOUT normalization[n_cropped]}

for i, f in enumerate(z_files):
    m = FNAME_RE.search(f.name)
    if not m:
        continue
    point_id = int(m.group(1))
    z_val = int(m.group(8))
    _, intensities = load_spectra(f)

    # Take median of raw reps, then preprocess
    median_raw = np.median(intensities, axis=0)
    proc, _ = preprocess_spectrum(median_raw, raw_raman_shifts, normalize=True)
    proc_unnorm, _ = preprocess_spectrum(median_raw, raw_raman_shifts, normalize=False)

    if point_id not in z_data:
        z_data[point_id] = {}
        z_data_raw[point_id] = {}
        z_data_unnorm[point_id] = {}
    z_data[point_id][z_val] = proc
    z_data_raw[point_id][z_val] = median_raw
    z_data_unnorm[point_id][z_val] = proc_unnorm

    if (i + 1) % 200 == 0:
        print(f"  {i + 1}/{len(z_files)}...")

z_values = sorted(next(iter(z_data.values())).keys())
z_arr = np.array(z_values)
z_points = sorted(z_data.keys())
print(f"  Loaded {len(z_data)} points, Z range: {z_values[0]} - {z_values[-1]}")

# ── Indices for characteristic peaks (on preprocessed/cropped axis) ───────────
char_indices = [find_nearest_idx(proc_raman_shifts, s) for s in CHARACTERISTIC_SHIFTS]
print(f"\nPeak indices (on cropped axis):")
for s, idx in zip(CHARACTERISTIC_SHIFTS, char_indices):
    print(f"  Target {s} cm⁻¹ → actual {proc_raman_shifts[idx]:.1f} cm⁻¹ (idx {idx})")

n_pts = len(all_points)
n_z_pts = len(z_points)
n_acq = first30_data[all_points[0]].shape[0]
acq_indices = np.arange(1, n_acq + 1)
rep_point = all_points[0]

# Color palettes
colors_line = plt.cm.Set1(np.linspace(0, 0.8, len(CHARACTERISTIC_SHIFTS)))
cmap_spatial = plt.cm.coolwarm
dist_norm = plt.Normalize(vmin=0, vmax=max_dist)


def add_distance_colorbar(fig, ax, label='Distance from centroid (µm)'):
    sm = plt.cm.ScalarMappable(cmap=cmap_spatial, norm=dist_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(label, fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    return cbar


# ══════════════════════════════════════════════════════════════════════════════
# PART 1: Temporal Stability Analysis (preprocessed)
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- Part 1: Temporal Stability ---")

# ── Figure 1: Raw vs Preprocessed temporal waterfall (2 rows) ─────────────────
print("  Figure 1: Raw vs preprocessed temporal waterfall...")
fig, (ax_raw, ax_proc) = setup_dark_figure(nrows=2, ncols=1, figsize=(12, 12))
cmap_acq = plt.cm.turbo
colors_acq = cmap_acq(np.linspace(0, 1, n_acq))

# Top: Raw spectra
for i in range(n_acq):
    ax_raw.plot(raw_raman_shifts, first30_data_raw[rep_point][i], color=colors_acq[i],
                linewidth=0.5, alpha=0.8)
ax_raw.set_xlabel('Raman Shift (cm$^{-1}$)', fontsize=13)
ax_raw.set_ylabel('Intensity (counts)', fontsize=13)
ax_raw.set_title(f'Raw — 30 Acquisitions (Point {rep_point})', fontsize=14)
ax_raw.tick_params(labelsize=11)

# Bottom: Preprocessed spectra
for i in range(n_acq):
    ax_proc.plot(proc_raman_shifts, first30_data[rep_point][i], color=colors_acq[i],
                 linewidth=0.5, alpha=0.8)
ax_proc.set_xlabel('Raman Shift (cm$^{-1}$)', fontsize=13)
ax_proc.set_ylabel('Normalized Intensity', fontsize=13)
ax_proc.set_title(f'Preprocessed — 30 Acquisitions (Point {rep_point})', fontsize=14)
ax_proc.tick_params(labelsize=11)

# Shared colorbar
sm = plt.cm.ScalarMappable(cmap=cmap_acq, norm=plt.Normalize(vmin=1, vmax=n_acq))
sm.set_array([])
cbar = fig.colorbar(sm, ax=[ax_raw, ax_proc], shrink=0.6, pad=0.02)
cbar.set_label('Acquisition Number', fontsize=12)

fig.suptitle('Temporal Stability: Raw vs Preprocessed', fontsize=16, y=1.01)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "fig1_temporal_waterfall.png", dpi=300,
            facecolor='black', bbox_inches='tight')
plt.close(fig)

# ── Figure 2: Peak intensity vs acquisition (mean ± std across points) ────────
print("  Figure 2: Peak intensity vs acquisition number...")
fig, ax = setup_dark_figure(figsize=(12, 7))

for idx, label, color in zip(char_indices, SHIFT_LABELS, colors_line):
    vals = np.array([first30_data[p][:, idx] for p in all_points])
    mean, std = vals.mean(axis=0), vals.std(axis=0)
    ax.plot(acq_indices, mean, color=color, linewidth=1.5, label=label)
    ax.fill_between(acq_indices, mean - std, mean + std, color=color, alpha=0.2)

ax.set_xlabel('Acquisition Number', fontsize=14)
ax.set_ylabel('Normalized Intensity', fontsize=14)
ax.set_title(f'Peak Intensity vs Acquisition (mean ± std, n={n_pts} points)', fontsize=15)
ax.legend(fontsize=11)
ax.tick_params(labelsize=12)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "fig2_temporal_peak_intensity.png", dpi=300,
            facecolor='black', bbox_inches='tight')
plt.close(fig)

# ── Figure 3: 1006 cm⁻¹ (Phe) intensity vs acquisition ──────────────────────
print("  Figure 3: Phe peak intensity vs acquisition...")
fig, ax = setup_dark_figure(figsize=(10, 6))
phe_idx = char_indices[0]  # 1006 cm⁻¹
vals = np.array([first30_data[p][:, phe_idx] for p in all_points])
mean, std = vals.mean(axis=0), vals.std(axis=0)

ax.plot(acq_indices, mean, color='#00d4ff', linewidth=2)
ax.fill_between(acq_indices, mean - std, mean + std, color='#00d4ff', alpha=0.2)

ax.set_xlabel('Acquisition Number', fontsize=14)
ax.set_ylabel('Normalized Intensity', fontsize=14)
ax.set_title(f'Phenylalanine (1006 cm$^{{-1}}$) Intensity vs Acquisition\n'
             f'(mean ± std, n={n_pts} points)', fontsize=15)
ax.tick_params(labelsize=12)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "fig3_temporal_phe.png", dpi=300,
            facecolor='black', bbox_inches='tight')
plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PART 2: Z-Focus Analysis (preprocessed)
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- Part 2: Z-Focus Analysis ---")

# ── Figure 4: Raw vs Preprocessed Z spectra (2 rows) ─────────────────────────
print("  Figure 4: Raw vs preprocessed Z spectra...")
fig, (ax_raw, ax_proc) = setup_dark_figure(nrows=2, ncols=1, figsize=(12, 12))
cmap_z = plt.cm.turbo
colors_z = cmap_z(np.linspace(0, 1, len(z_values)))

# Top: Raw spectra at different Z
for i, z in enumerate(z_values):
    ax_raw.plot(raw_raman_shifts, z_data_raw[rep_point][z], color=colors_z[i],
                linewidth=0.7, alpha=0.85)
ax_raw.set_xlabel('Raman Shift (cm$^{-1}$)', fontsize=13)
ax_raw.set_ylabel('Intensity (counts)', fontsize=13)
ax_raw.set_title(f'Raw — Spectra at Different Z (Point {rep_point})', fontsize=14)
ax_raw.tick_params(labelsize=11)

# Bottom: Preprocessed spectra at different Z
for i, z in enumerate(z_values):
    ax_proc.plot(proc_raman_shifts, z_data[rep_point][z], color=colors_z[i],
                 linewidth=0.7, alpha=0.85)
ax_proc.set_xlabel('Raman Shift (cm$^{-1}$)', fontsize=13)
ax_proc.set_ylabel('Normalized Intensity', fontsize=13)
ax_proc.set_title(f'Preprocessed — Spectra at Different Z (Point {rep_point})', fontsize=14)
ax_proc.tick_params(labelsize=11)

# Shared colorbar
sm = plt.cm.ScalarMappable(cmap=cmap_z, norm=plt.Normalize(vmin=z_arr.min(), vmax=z_arr.max()))
sm.set_array([])
cbar = fig.colorbar(sm, ax=[ax_raw, ax_proc], shrink=0.6, pad=0.02)
cbar.set_label('Z Position', fontsize=12)

fig.suptitle('Z-Focus: Raw vs Preprocessed', fontsize=16, y=1.01)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "fig4_z_spectra.png", dpi=300,
            facecolor='black', bbox_inches='tight')
plt.close(fig)

# ── Figure 5: Peak intensity vs Z (mean ± std) ───────────────────────────────
print("  Figure 5: Peak intensity vs Z position...")
fig, ax = setup_dark_figure(figsize=(12, 7))

for idx, label, color in zip(char_indices, SHIFT_LABELS, colors_line):
    vals = np.array([[z_data[p][z][idx] for z in z_values] for p in z_points])
    mean, std = vals.mean(axis=0), vals.std(axis=0)
    ax.plot(z_arr, mean, color=color, linewidth=1.5, marker='o', markersize=4, label=label)
    ax.fill_between(z_arr, mean - std, mean + std, color=color, alpha=0.2)

ax.set_xlabel('Z Position', fontsize=14)
ax.set_ylabel('Normalized Intensity', fontsize=14)
ax.set_title(f'Peak Intensity vs Z Position (mean ± std, n={n_z_pts} points)', fontsize=15)
ax.legend(fontsize=11)
ax.tick_params(labelsize=12)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "fig5_z_peak_intensity.png", dpi=300,
            facecolor='black', bbox_inches='tight')
plt.close(fig)

# ── Figure 6: Phe (1006) intensity vs Z ──────────────────────────────────────
print("  Figure 6: Phe peak intensity vs Z...")
fig, ax = setup_dark_figure(figsize=(10, 6))

vals = np.array([[z_data[p][z][phe_idx] for z in z_values] for p in z_points])
mean, std = vals.mean(axis=0), vals.std(axis=0)

ax.plot(z_arr, mean, color='#ff6b6b', linewidth=2, marker='o', markersize=5)
ax.fill_between(z_arr, mean - std, mean + std, color='#ff6b6b', alpha=0.2)

best_z_idx = np.argmax(mean)
best_z = z_arr[best_z_idx]
ax.axvline(best_z, color='#ffdd57', linestyle='--', alpha=0.7, linewidth=1)
ax.annotate(f'Best Z = {best_z}', xy=(best_z, mean[best_z_idx]),
            xytext=(best_z + 2, mean[best_z_idx] * 1.05),
            color='#ffdd57', fontsize=12, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#ffdd57'))

ax.set_xlabel('Z Position', fontsize=14)
ax.set_ylabel('Normalized Intensity', fontsize=14)
ax.set_title(f'Phenylalanine (1006 cm$^{{-1}}$) Intensity vs Z\n'
             f'(mean ± std, n={n_z_pts} points)', fontsize=15)
ax.tick_params(labelsize=12)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "fig6_z_phe.png", dpi=300,
            facecolor='black', bbox_inches='tight')
plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PART 3: Spatially-Aware Per-Point Analysis
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- Part 3: Spatially-Aware Per-Point Analysis ---")

xs = np.array([point_coords[p][0] for p in all_points])
ys = np.array([point_coords[p][1] for p in all_points])

# ── Figure 7: Per-point Phe intensity vs acquisition, colored by distance ─────
print("  Figure 7: Per-point Phe temporal curves...")
fig, ax = setup_dark_figure(figsize=(12, 7))

for p in all_points:
    color = cmap_spatial(dist_norm(point_dist[p]))
    ax.plot(acq_indices, first30_data[p][:, phe_idx], color=color,
            linewidth=0.8, alpha=0.7)

add_distance_colorbar(fig, ax)
ax.set_xlabel('Acquisition Number', fontsize=14)
ax.set_ylabel('Normalized Intensity', fontsize=14)
ax.set_title('Per-Point Phe (1006 cm$^{-1}$) Intensity vs Acquisition\n'
             'Colored by distance from centroid', fontsize=15)
ax.tick_params(labelsize=12)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "fig7_temporal_phe_per_point.png", dpi=300,
            facecolor='black', bbox_inches='tight')
plt.close(fig)

# ── Figure 8: Per-point Phe intensity vs Z, colored by distance ──────────────
print("  Figure 8: Per-point Phe Z curves...")
fig, ax = setup_dark_figure(figsize=(12, 7))

for p in z_points:
    if p not in point_dist:
        continue
    vals = np.array([z_data[p][z][phe_idx] for z in z_values])
    color = cmap_spatial(dist_norm(point_dist[p]))
    ax.plot(z_arr, vals, color=color, linewidth=0.8, alpha=0.7,
            marker='o', markersize=2)

add_distance_colorbar(fig, ax)
ax.set_xlabel('Z Position', fontsize=14)
ax.set_ylabel('Normalized Intensity', fontsize=14)
ax.set_title('Per-Point Phe (1006 cm$^{-1}$) Intensity vs Z\n'
             'Colored by distance from centroid', fontsize=15)
ax.tick_params(labelsize=12)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "fig8_z_phe_per_point.png", dpi=300,
            facecolor='black', bbox_inches='tight')
plt.close(fig)

# ── Figure 9: Spatial maps (2x2) ─────────────────────────────────────────────
print("  Figure 9: Spatial maps of key metrics...")
fig, axes = setup_dark_figure(nrows=2, ncols=2, figsize=(14, 12))

# (a) Temporal CV of Phe peak
temporal_cv = []
for p in all_points:
    v = first30_data[p][:, phe_idx]
    temporal_cv.append(np.std(v) / np.mean(v) if np.mean(v) > 0 else 0)
temporal_cv = np.array(temporal_cv)

sc = axes[0, 0].scatter(xs, ys, c=temporal_cv, cmap='magma', s=60,
                         edgecolors='white', linewidth=0.3)
fig.colorbar(sc, ax=axes[0, 0], shrink=0.8).set_label('CV', fontsize=11)
axes[0, 0].set_title('Temporal Stability — Phe CV', fontsize=13)
axes[0, 0].set_xlabel('X (µm)', fontsize=11)
axes[0, 0].set_ylabel('Y (µm)', fontsize=11)
axes[0, 0].set_aspect('equal')

# (b) Mean Phe intensity
mean_phe = np.array([first30_data[p][:, phe_idx].mean() for p in all_points])
sc = axes[0, 1].scatter(xs, ys, c=mean_phe, cmap='viridis', s=60,
                         edgecolors='white', linewidth=0.3)
fig.colorbar(sc, ax=axes[0, 1], shrink=0.8).set_label('Norm. Intensity', fontsize=11)
axes[0, 1].set_title('Mean Phe (1006) Intensity', fontsize=13)
axes[0, 1].set_xlabel('X (µm)', fontsize=11)
axes[0, 1].set_ylabel('Y (µm)', fontsize=11)
axes[0, 1].set_aspect('equal')

# (c) Optimal Z per point (by Phe peak)
optimal_z = []
for p in all_points:
    if p in z_data:
        phe_vals = [z_data[p][z][phe_idx] for z in z_values]
        optimal_z.append(z_values[np.argmax(phe_vals)])
    else:
        optimal_z.append(np.nan)
optimal_z = np.array(optimal_z, dtype=float)

sc = axes[1, 0].scatter(xs, ys, c=optimal_z, cmap='plasma', s=60,
                         edgecolors='white', linewidth=0.3)
fig.colorbar(sc, ax=axes[1, 0], shrink=0.8).set_label('Optimal Z', fontsize=11)
axes[1, 0].set_title('Optimal Z (max Phe intensity)', fontsize=13)
axes[1, 0].set_xlabel('X (µm)', fontsize=11)
axes[1, 0].set_ylabel('Y (µm)', fontsize=11)
axes[1, 0].set_aspect('equal')

# (d) Max Phe intensity at optimal Z
max_phe_z = []
for p in all_points:
    if p in z_data:
        max_phe_z.append(max(z_data[p][z][phe_idx] for z in z_values))
    else:
        max_phe_z.append(np.nan)
max_phe_z = np.array(max_phe_z, dtype=float)

sc = axes[1, 1].scatter(xs, ys, c=max_phe_z, cmap='inferno', s=60,
                         edgecolors='white', linewidth=0.3)
fig.colorbar(sc, ax=axes[1, 1], shrink=0.8).set_label('Norm. Intensity', fontsize=11)
axes[1, 1].set_title('Max Phe Intensity (at optimal Z)', fontsize=13)
axes[1, 1].set_xlabel('X (µm)', fontsize=11)
axes[1, 1].set_ylabel('Y (µm)', fontsize=11)
axes[1, 1].set_aspect('equal')

fig.suptitle('Spatial Maps — Phenylalanine (1006 cm$^{-1}$)', fontsize=16, y=1.01)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "fig9_spatial_maps.png", dpi=300,
            facecolor='black', bbox_inches='tight')
plt.close(fig)

# ── Figure 10: Per-point temporal traces at all peaks ─────────────────────────
print("  Figure 10: Per-point temporal traces at all peaks...")
fig, axes = setup_dark_figure(nrows=2, ncols=2, figsize=(16, 12))

for idx, label, ax in zip(char_indices, SHIFT_LABELS, axes.flat):
    for p in all_points:
        color = cmap_spatial(dist_norm(point_dist[p]))
        ax.plot(acq_indices, first30_data[p][:, idx], color=color,
                linewidth=0.6, alpha=0.6)
    ax.set_xlabel('Acquisition Number', fontsize=11)
    ax.set_ylabel('Norm. Intensity', fontsize=11)
    ax.set_title(label, fontsize=13)
    ax.tick_params(labelsize=10)

sm = plt.cm.ScalarMappable(cmap=cmap_spatial, norm=dist_norm)
sm.set_array([])
fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02,
             label='Distance from centroid (µm)')
fig.suptitle('Per-Point Temporal Traces at Biological Peaks', fontsize=16, y=1.01)
fig.savefig(OUTPUT_DIR / "fig10_temporal_per_shift.png", dpi=300,
            facecolor='black', bbox_inches='tight')
plt.close(fig)

# ── Figure 11: Per-point Z traces at all peaks ───────────────────────────────
print("  Figure 11: Per-point Z traces at all peaks...")
fig, axes = setup_dark_figure(nrows=2, ncols=2, figsize=(16, 12))

for idx, label, ax in zip(char_indices, SHIFT_LABELS, axes.flat):
    for p in z_points:
        if p not in point_dist:
            continue
        vals = np.array([z_data[p][z][idx] for z in z_values])
        color = cmap_spatial(dist_norm(point_dist[p]))
        ax.plot(z_arr, vals, color=color, linewidth=0.6, alpha=0.6,
                marker='o', markersize=2)
    ax.set_xlabel('Z Position', fontsize=11)
    ax.set_ylabel('Norm. Intensity', fontsize=11)
    ax.set_title(label, fontsize=13)
    ax.tick_params(labelsize=10)

sm = plt.cm.ScalarMappable(cmap=cmap_spatial, norm=dist_norm)
sm.set_array([])
fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02,
             label='Distance from centroid (µm)')
fig.suptitle('Per-Point Z-Focus Traces at Biological Peaks', fontsize=16, y=1.01)
fig.savefig(OUTPUT_DIR / "fig11_z_per_shift.png", dpi=300,
            facecolor='black', bbox_inches='tight')
plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PART 4: Dimensionality Reduction (PCA, t-SNE, UMAP) — Z-focus spectra
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- Part 4: Dimensionality Reduction ---")

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

# Build matrix: one row per (point, z) combination
print("  Building feature matrix from z-focus spectra...")
spectra_matrix = []
z_labels = []
dist_labels = []
point_labels = []

for p in z_points:
    if p not in point_dist:
        continue
    for z in z_values:
        spectra_matrix.append(z_data[p][z])
        z_labels.append(z)
        dist_labels.append(point_dist[p])
        point_labels.append(p)

spectra_matrix = np.array(spectra_matrix)
z_labels = np.array(z_labels)
dist_labels = np.array(dist_labels)
point_labels = np.array(point_labels)

print(f"  Matrix shape: {spectra_matrix.shape} ({len(z_points)} points × {len(z_values)} Z values)")

# Run dimensionality reduction
print("  Running PCA...")
pca = PCA(n_components=2)
pca_result = pca.fit_transform(spectra_matrix)
print(f"  PCA explained variance: {pca.explained_variance_ratio_[0]:.1%}, {pca.explained_variance_ratio_[1]:.1%}")

print("  Running t-SNE...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
tsne_result = tsne.fit_transform(spectra_matrix)

print("  Running UMAP...")
reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
umap_result = reducer.fit_transform(spectra_matrix)

# ── Figure 12: PCA / t-SNE / UMAP colored by Z (1×3) ─────────────────────────
print("  Figure 12: Dimensionality reduction colored by Z...")
fig, axes = setup_dark_figure(nrows=1, ncols=3, figsize=(20, 6))
cmap_z_dr = plt.cm.turbo
z_norm = plt.Normalize(vmin=z_arr.min(), vmax=z_arr.max())

results = [pca_result, tsne_result, umap_result]
titles = [
    f'PCA ({pca.explained_variance_ratio_[0]:.0%} + {pca.explained_variance_ratio_[1]:.0%})',
    't-SNE',
    'UMAP',
]
xlabels = ['PC1', 't-SNE 1', 'UMAP 1']
ylabels = ['PC2', 't-SNE 2', 'UMAP 2']

for ax, result, title, xl, yl in zip(axes, results, titles, xlabels, ylabels):
    sc = ax.scatter(result[:, 0], result[:, 1], c=z_labels, cmap=cmap_z_dr,
                    norm=z_norm, s=12, alpha=0.8, edgecolors='none')
    ax.set_xlabel(xl, fontsize=12)
    ax.set_ylabel(yl, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.tick_params(labelsize=10)

sm = plt.cm.ScalarMappable(cmap=cmap_z_dr, norm=z_norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes.tolist(), shrink=0.8, pad=0.02)
cbar.set_label('Z Position', fontsize=12)

fig.suptitle('Preprocessed Z-Focus Spectra — Dimensionality Reduction (colored by Z)',
             fontsize=16, y=1.02)
fig.savefig(OUTPUT_DIR / "fig12_dimred_by_z.png", dpi=300,
            facecolor='black', bbox_inches='tight')
plt.close(fig)

# ── Figure 13: Same 3 methods colored by distance from centroid ───────────────
print("  Figure 13: Dimensionality reduction colored by distance...")
fig, axes = setup_dark_figure(nrows=1, ncols=3, figsize=(20, 6))

for ax, result, title, xl, yl in zip(axes, results, titles, xlabels, ylabels):
    sc = ax.scatter(result[:, 0], result[:, 1], c=dist_labels, cmap=cmap_spatial,
                    norm=dist_norm, s=12, alpha=0.8, edgecolors='none')
    ax.set_xlabel(xl, fontsize=12)
    ax.set_ylabel(yl, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.tick_params(labelsize=10)

sm = plt.cm.ScalarMappable(cmap=cmap_spatial, norm=dist_norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes.tolist(), shrink=0.8, pad=0.02)
cbar.set_label('Distance from centroid (µm)', fontsize=12)

fig.suptitle('Preprocessed Z-Focus Spectra — Dimensionality Reduction (colored by distance)',
             fontsize=16, y=1.02)
fig.savefig(OUTPUT_DIR / "fig13_dimred_by_distance.png", dpi=300,
            facecolor='black', bbox_inches='tight')
plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# PART 5: Per-Point Raw vs Preprocessed (all 60 points)
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- Part 5: Per-Point Raw vs Preprocessed (all points) ---")

TEMPORAL_DIR = OUTPUT_DIR / "temporal_per_point"
Z_DIR = OUTPUT_DIR / "z_per_point"
TEMPORAL_DIR.mkdir(exist_ok=True)
Z_DIR.mkdir(exist_ok=True)

for pi, p in enumerate(all_points):
    # ── Temporal: raw vs preprocessed ─────────────────────────────────────────
    fig, (ax_raw, ax_proc) = setup_dark_figure(nrows=2, ncols=1, figsize=(12, 10))

    for i in range(n_acq):
        ax_raw.plot(raw_raman_shifts, first30_data_raw[p][i], color=colors_acq[i],
                    linewidth=0.5, alpha=0.8)
    ax_raw.set_xlabel('Raman Shift (cm$^{-1}$)', fontsize=12)
    ax_raw.set_ylabel('Intensity (counts)', fontsize=12)
    ax_raw.set_title('Raw', fontsize=13)
    ax_raw.tick_params(labelsize=10)

    for i in range(n_acq):
        ax_proc.plot(proc_raman_shifts, first30_data[p][i], color=colors_acq[i],
                     linewidth=0.5, alpha=0.8)
    ax_proc.set_xlabel('Raman Shift (cm$^{-1}$)', fontsize=12)
    ax_proc.set_ylabel('Normalized Intensity', fontsize=12)
    ax_proc.set_title('Preprocessed', fontsize=13)
    ax_proc.tick_params(labelsize=10)

    sm = plt.cm.ScalarMappable(cmap=cmap_acq, norm=plt.Normalize(vmin=1, vmax=n_acq))
    sm.set_array([])
    fig.colorbar(sm, ax=[ax_raw, ax_proc], shrink=0.6, pad=0.02,
                 label='Acquisition Number')

    coords = point_coords[p]
    ring = point_ring[p]
    dist = point_dist[p]
    fig.suptitle(f'Temporal — Point {p}  |  Ring {ring}  |  '
                 f'({coords[0]:.0f}, {coords[1]:.0f}) µm  |  '
                 f'd={dist:.0f} µm', fontsize=14, y=1.01)

    fig.savefig(TEMPORAL_DIR / f"point_{p:02d}_temporal.png", dpi=200,
                facecolor='black', bbox_inches='tight')
    plt.close(fig)

    # ── Z-focus: raw vs preprocessed ─────────────────────────────────────────
    if p in z_data:
        fig, (ax_raw, ax_proc) = setup_dark_figure(nrows=2, ncols=1, figsize=(12, 10))

        for i, z in enumerate(z_values):
            ax_raw.plot(raw_raman_shifts, z_data_raw[p][z], color=colors_z[i],
                        linewidth=0.7, alpha=0.85)
        ax_raw.set_xlabel('Raman Shift (cm$^{-1}$)', fontsize=12)
        ax_raw.set_ylabel('Intensity (counts)', fontsize=12)
        ax_raw.set_title('Raw', fontsize=13)
        ax_raw.tick_params(labelsize=10)

        for i, z in enumerate(z_values):
            ax_proc.plot(proc_raman_shifts, z_data[p][z], color=colors_z[i],
                         linewidth=0.7, alpha=0.85)
        ax_proc.set_xlabel('Raman Shift (cm$^{-1}$)', fontsize=12)
        ax_proc.set_ylabel('Normalized Intensity', fontsize=12)
        ax_proc.set_title('Preprocessed', fontsize=13)
        ax_proc.tick_params(labelsize=10)

        sm = plt.cm.ScalarMappable(cmap=cmap_z,
                                    norm=plt.Normalize(vmin=z_arr.min(), vmax=z_arr.max()))
        sm.set_array([])
        fig.colorbar(sm, ax=[ax_raw, ax_proc], shrink=0.6, pad=0.02,
                     label='Z Position')

        fig.suptitle(f'Z-Focus — Point {p}  |  Ring {ring}  |  '
                     f'({coords[0]:.0f}, {coords[1]:.0f}) µm  |  '
                     f'd={dist:.0f} µm', fontsize=14, y=1.01)

        fig.savefig(Z_DIR / f"point_{p:02d}_z.png", dpi=200,
                    facecolor='black', bbox_inches='tight')
        plt.close(fig)

    if (pi + 1) % 10 == 0:
        print(f"  {pi + 1}/{len(all_points)} points...")

print(f"  Saved {len(all_points)} temporal figures to {TEMPORAL_DIR}/")
print(f"  Saved {len(all_points)} z-focus figures to {Z_DIR}/")

# ══════════════════════════════════════════════════════════════════════════════
# PART 6: Autofocus Validation — Three Metrics Compared
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- Part 6: Autofocus Validation (3 metrics) ---")

from scipy import stats

METADATA_DIR = DATA_DIR.parent / "metadata"

# ── Focus metric functions ────────────────────────────────────────────────────

def calculate_ratio_score(wavelength, intensity):
    """Quartz/laser ratio on raw spectrum. Lower = better sample focus."""
    mask_laser = (wavelength >= 790) & (wavelength <= 795)
    mask_quartz = (wavelength >= 808) & (wavelength <= 815)
    laser_sum = np.sum(intensity[mask_laser])
    quartz_sum = np.sum(intensity[mask_quartz])
    if laser_sum == 0:
        return np.nan
    return quartz_sum / laser_sum


def calculate_phe_intensity(proc_spectrum, proc_rs):
    """Phe peak intensity from preprocessed spectrum. Higher = better."""
    phe_i = find_nearest_idx(proc_rs, 1006)
    return proc_spectrum[phe_i]


def calculate_snr(proc_spectrum, proc_rs):
    """SNR: Phe peak height / noise std in a quiet region. Higher = better."""
    # Signal: Phe peak at ~1006
    phe_i = find_nearest_idx(proc_rs, 1006)
    signal = proc_spectrum[phe_i]
    # Noise: std of a relatively flat region (~1750-1780 cm⁻¹)
    noise_mask = (proc_rs >= 1750) & (proc_rs <= 1780)
    if np.sum(noise_mask) < 3:
        # fallback: use 1100-1130 cm⁻¹
        noise_mask = (proc_rs >= 1100) & (proc_rs <= 1130)
    noise_std = np.std(proc_spectrum[noise_mask])
    if noise_std == 0:
        return np.nan
    return signal / noise_std


# ── Parse nanodrive position from first30 metadata ───────────────────────────
print("  Reading autofocus positions from first30 metadata...")
autofocus_z = {}  # point_id -> nanodrive position (µm)

for f in first30_files:
    m = FNAME_RE.search(f.name)
    if not m:
        continue
    point_id = int(m.group(1))
    meta_name = f.name.replace('.txt', '_metadata.txt')
    meta_path = METADATA_DIR / meta_name
    if meta_path.exists():
        with open(meta_path, 'r') as mf:
            for line in mf:
                if line.startswith('Nanodrive position'):
                    autofocus_z[point_id] = float(line.split(':')[1].strip())
                    break

print(f"  Got autofocus Z for {len(autofocus_z)} points")
print(f"  Autofocus Z range: {min(autofocus_z.values()):.1f} - {max(autofocus_z.values()):.1f} µm")

# ── Compute all three metrics per (point, z) ─────────────────────────────────
print("  Computing focus metrics across Z sweep...")

metric_names = ['Ratio Score', 'Phe Intensity', 'SNR']
metric_short = ['ratio', 'phe', 'snr']
# For each metric: point_id -> [score_at_z0, score_at_z1, ...]
all_metric_scores = {m: {} for m in metric_short}
# For each metric: point_id -> best_z
best_z_by_metric = {m: {} for m in metric_short}

for p in z_points:
    ratio_arr = []
    phe_arr = []
    snr_arr = []

    for z in z_values:
        # Ratio: from raw spectrum
        ratio_arr.append(calculate_ratio_score(sample_wl, z_data_raw[p][z]))
        # Phe & SNR: from preprocessed WITHOUT normalization (preserves absolute intensity)
        phe_arr.append(calculate_phe_intensity(z_data_unnorm[p][z], proc_raman_shifts))
        snr_arr.append(calculate_snr(z_data_unnorm[p][z], proc_raman_shifts))

    ratio_arr = np.array(ratio_arr)
    phe_arr = np.array(phe_arr)
    snr_arr = np.array(snr_arr)

    all_metric_scores['ratio'][p] = ratio_arr
    all_metric_scores['phe'][p] = phe_arr
    all_metric_scores['snr'][p] = snr_arr

    # Ratio: minimize (less quartz = better focus)
    if not np.all(np.isnan(ratio_arr)):
        best_z_by_metric['ratio'][p] = z_values[np.nanargmin(ratio_arr)]
    else:
        best_z_by_metric['ratio'][p] = np.nan

    # Phe: maximize
    if not np.all(np.isnan(phe_arr)):
        best_z_by_metric['phe'][p] = z_values[np.nanargmax(phe_arr)]
    else:
        best_z_by_metric['phe'][p] = np.nan

    # SNR: maximize
    if not np.all(np.isnan(snr_arr)):
        best_z_by_metric['snr'][p] = z_values[np.nanargmax(snr_arr)]
    else:
        best_z_by_metric['snr'][p] = np.nan

# ── Figure 14: All three metrics vs Z (per point, 1×3) ───────────────────────
print("  Figure 14: All metrics vs Z (per point)...")
fig, axes = setup_dark_figure(nrows=1, ncols=3, figsize=(22, 7))
ylabels_metric = ['Quartz / Laser (lower = better)',
                   'Normalized Intensity (higher = better)',
                   'Signal / Noise (higher = better)']

for ax, m, name, yl in zip(axes, metric_short, metric_names, ylabels_metric):
    for p in z_points:
        if p not in point_dist:
            continue
        color = cmap_spatial(dist_norm(point_dist[p]))
        ax.plot(z_arr, all_metric_scores[m][p], color=color, linewidth=0.8,
                alpha=0.7, marker='o', markersize=2)
    ax.set_xlabel('Z Position (µm)', fontsize=12)
    ax.set_ylabel(yl, fontsize=11)
    ax.set_title(name, fontsize=14)
    ax.tick_params(labelsize=10)

sm = plt.cm.ScalarMappable(cmap=cmap_spatial, norm=dist_norm)
sm.set_array([])
fig.colorbar(sm, ax=axes.tolist(), shrink=0.6, pad=0.02,
             label='Distance from centroid (µm)')
fig.suptitle('Focus Metrics vs Z Position (per point)', fontsize=16, y=1.02)
fig.savefig(OUTPUT_DIR / "fig14_all_metrics_vs_z.png", dpi=300,
            facecolor='black', bbox_inches='tight')
plt.close(fig)

# ── Figure 15: Autofocus Z vs optimal Z — three metrics side by side ─────────
print("  Figure 15: Autofocus Z vs optimal Z (3 metrics)...")

points_valid = [p for p in all_points if p in autofocus_z and p in z_data]
af_vals = np.array([autofocus_z[p] for p in points_valid])
dists_v = np.array([point_dist[p] for p in points_valid])

fig, axes = setup_dark_figure(nrows=1, ncols=3, figsize=(22, 7))
z_range = [z_arr.min() - 2, z_arr.max() + 2]

for ax, m, name in zip(axes, metric_short, metric_names):
    opt_vals = np.array([best_z_by_metric[m].get(p, np.nan) for p in points_valid])
    valid = ~np.isnan(opt_vals)

    sc = ax.scatter(af_vals[valid], opt_vals[valid], c=dists_v[valid],
                    cmap=cmap_spatial, norm=dist_norm, s=50,
                    edgecolors='white', linewidth=0.3, zorder=5)

    # Identity line
    ax.plot(z_range, z_range, '--', color='#ffdd57', linewidth=1.5, alpha=0.7,
            label='y = x')

    # Linear fit
    if np.sum(valid) > 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            af_vals[valid], opt_vals[valid])
        fit_x = np.array(z_range)
        fit_y = slope * fit_x + intercept
        ax.plot(fit_x, fit_y, '-', color='#00ff88', linewidth=1.5, alpha=0.8,
                label=f'fit: slope={slope:.2f}, R²={r_value**2:.3f}')
        ax.set_title(f'{name}\nslope = {slope:.3f} ± {std_err:.3f}  |  '
                     f'intercept = {intercept:.1f} µm  |  R² = {r_value**2:.3f}',
                     fontsize=12)
        ax.legend(fontsize=9, loc='upper left')
    else:
        ax.set_title(name, fontsize=13)

    ax.set_xlabel('Autofocus Z (µm)', fontsize=12)
    ax.set_ylabel(f'Optimal Z — {name} (µm)', fontsize=12)
    ax.set_xlim(z_range)
    ax.set_ylim(z_range)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=10)

sm = plt.cm.ScalarMappable(cmap=cmap_spatial, norm=dist_norm)
sm.set_array([])
fig.colorbar(sm, ax=axes.tolist(), shrink=0.6, pad=0.02,
             label='Distance from centroid (µm)')
fig.suptitle('Autofocus Z (short exposure) vs Optimal Z (long exposure) — 3 Metrics',
             fontsize=16, y=1.02)
fig.savefig(OUTPUT_DIR / "fig15_autofocus_3metrics.png", dpi=300,
            facecolor='black', bbox_inches='tight')
plt.close(fig)

# ── Figure 16: Residuals (optimal - autofocus) for each metric ───────────────
print("  Figure 16: Residual analysis (3 metrics)...")
fig, axes = setup_dark_figure(nrows=1, ncols=3, figsize=(22, 7))
colors_metric = ['#00d4ff', '#ff6b6b', '#7dff7d']

for ax, m, name, clr in zip(axes, metric_short, metric_names, colors_metric):
    opt_vals = np.array([best_z_by_metric[m].get(p, np.nan) for p in points_valid])
    valid = ~np.isnan(opt_vals)
    residuals = opt_vals[valid] - af_vals[valid]

    sc = ax.scatter(dists_v[valid], residuals, c=af_vals[valid], cmap=plt.cm.turbo,
                    s=50, edgecolors='white', linewidth=0.3)
    ax.axhline(0, color='#ffdd57', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Distance from centroid (µm)', fontsize=12)
    ax.set_ylabel(f'ΔZ (µm)', fontsize=12)
    ax.set_title(f'{name}\noffset: {np.mean(residuals):.1f} ± {np.std(residuals):.1f} µm',
                 fontsize=13)
    ax.tick_params(labelsize=10)
    fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02).set_label('Autofocus Z (µm)', fontsize=10)

fig.suptitle('Focus Discrepancy vs Spatial Position — 3 Metrics', fontsize=16, y=1.02)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "fig16_residuals_3metrics.png", dpi=300,
            facecolor='black', bbox_inches='tight')
plt.close(fig)

# ── Figure 17: Spatial maps — autofocus + 3 optimal Z + 3 differences (2×4) ──
print("  Figure 17: Spatial maps of focus positions (3 metrics)...")
fig, axes = setup_dark_figure(nrows=2, ncols=4, figsize=(28, 12))
xs_v = np.array([point_coords[p][0] for p in points_valid])
ys_v = np.array([point_coords[p][1] for p in points_valid])

# Global Z range for consistent colorbar
all_opt = []
for m in metric_short:
    all_opt.extend([best_z_by_metric[m].get(p, np.nan) for p in points_valid])
all_opt = np.array(all_opt)
z_vmin = np.nanmin([af_vals.min(), np.nanmin(all_opt)])
z_vmax = np.nanmax([af_vals.max(), np.nanmax(all_opt)])

# Row 1: Autofocus Z + 3 optimal Z maps
sc = axes[0, 0].scatter(xs_v, ys_v, c=af_vals, cmap='turbo', s=50,
                          edgecolors='white', linewidth=0.3, vmin=z_vmin, vmax=z_vmax)
fig.colorbar(sc, ax=axes[0, 0], shrink=0.8).set_label('Z (µm)', fontsize=10)
axes[0, 0].set_title('Autofocus Z\n(short exposure)', fontsize=12)
axes[0, 0].set_aspect('equal')

for j, (m, name) in enumerate(zip(metric_short, metric_names)):
    opt = np.array([best_z_by_metric[m].get(p, np.nan) for p in points_valid])
    valid = ~np.isnan(opt)
    sc = axes[0, j + 1].scatter(xs_v[valid], ys_v[valid], c=opt[valid], cmap='turbo',
                                 s=50, edgecolors='white', linewidth=0.3,
                                 vmin=z_vmin, vmax=z_vmax)
    fig.colorbar(sc, ax=axes[0, j + 1], shrink=0.8).set_label('Z (µm)', fontsize=10)
    axes[0, j + 1].set_title(f'Optimal Z\n({name})', fontsize=12)
    axes[0, j + 1].set_aspect('equal')

# Row 2: empty first cell + 3 difference maps
axes[1, 0].axis('off')

for j, (m, name) in enumerate(zip(metric_short, metric_names)):
    opt = np.array([best_z_by_metric[m].get(p, np.nan) for p in points_valid])
    valid = ~np.isnan(opt)
    diff = opt[valid] - af_vals[valid]
    abs_max = max(np.abs(diff).max(), 1)
    sc = axes[1, j + 1].scatter(xs_v[valid], ys_v[valid], c=diff, cmap='coolwarm',
                                 s=50, edgecolors='white', linewidth=0.3,
                                 vmin=-abs_max, vmax=abs_max)
    fig.colorbar(sc, ax=axes[1, j + 1], shrink=0.8).set_label('ΔZ (µm)', fontsize=10)
    axes[1, j + 1].set_title(f'Difference\n({name} − autofocus)', fontsize=12)
    axes[1, j + 1].set_aspect('equal')

for ax in axes.flat:
    if ax.get_visible() and len(ax.collections) > 0:
        ax.set_xlabel('X (µm)', fontsize=10)
        ax.set_ylabel('Y (µm)', fontsize=10)

fig.suptitle('Spatial Maps: Autofocus vs Optimal Z (3 Metrics)', fontsize=16, y=1.01)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "fig17_focus_spatial_3metrics.png", dpi=300,
            facecolor='black', bbox_inches='tight')
plt.close(fig)

# ── Print summary table ──────────────────────────────────────────────────────
print(f"\n  {'Metric':<20} {'Slope':>8} {'± SE':>8} {'Intercept':>12} {'R²':>8} {'Offset':>10} {'Std':>8}")
print(f"  {'─' * 74}")
for m, name in zip(metric_short, metric_names):
    opt = np.array([best_z_by_metric[m].get(p, np.nan) for p in points_valid])
    valid = ~np.isnan(opt)
    residuals = opt[valid] - af_vals[valid]
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        af_vals[valid], opt[valid])
    print(f"  {name:<20} {slope:>8.3f} {std_err:>8.3f} {intercept:>+12.1f} "
          f"{r_value**2:>8.3f} {np.mean(residuals):>+10.1f} {np.std(residuals):>8.1f}")

# ── Figure 18: Ratio score vs Z for point 50 ─────────────────────────────────
print("  Figure 18: Ratio score vs Z (point 50)...")
p50 = 50
fig, ax = setup_dark_figure(figsize=(10, 7))

scores_50 = all_metric_scores['ratio'][p50]
ax.plot(z_arr, scores_50, color='#00d4ff', linewidth=2, marker='o', markersize=6,
        zorder=5, label='Ratio score')

# Autofocus Z (dashed vertical)
af_z50 = autofocus_z[p50]
ax.axvline(af_z50, color='#ffdd57', linestyle='--', linewidth=1.5, alpha=0.9,
           label=f'Autofocus Z = {af_z50:.0f} µm')

# Lowest ratio score (dashed horizontal + vertical)
best_idx = np.nanargmin(scores_50)
best_z50 = z_arr[best_idx]
best_score = scores_50[best_idx]
ax.axvline(best_z50, color='#ff6b6b', linestyle='--', linewidth=1.5, alpha=0.9,
           label=f'Min ratio Z = {best_z50:.0f} µm')
ax.axhline(best_score, color='#ff6b6b', linestyle=':', linewidth=1, alpha=0.6)

ax.set_xlabel('Z Position (µm)', fontsize=14)
ax.set_ylabel('Ratio Score (quartz / laser)', fontsize=14)
coords_50 = point_coords[p50]
ax.set_title(f'Ratio Score vs Z — Point {p50}\n'
             f'Ring {point_ring[p50]}  |  ({coords_50[0]:.0f}, {coords_50[1]:.0f}) µm  |  '
             f'd = {point_dist[p50]:.0f} µm', fontsize=15)
ax.legend(fontsize=12, loc='best')
ax.tick_params(labelsize=12)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "fig18_ratio_point50.png", dpi=300,
            facecolor='black', bbox_inches='tight')
plt.close(fig)

# ── Figure 19: Optimal Z (ratio) vs Autofocus Z, colored by point number ─────
print("  Figure 19: Optimal Z vs Autofocus Z (colored by point number)...")
fig, ax = setup_dark_figure(figsize=(10, 10))

opt_ratio_vals = np.array([best_z_by_metric['ratio'].get(p, np.nan) for p in points_valid])
valid_mask = ~np.isnan(opt_ratio_vals)
pv = np.array(points_valid)

point_norm = plt.Normalize(vmin=pv[valid_mask].min(), vmax=pv[valid_mask].max())

sc = ax.scatter(af_vals[valid_mask], opt_ratio_vals[valid_mask],
                c=pv[valid_mask], cmap='turbo', norm=point_norm,
                s=200, edgecolors='white', linewidth=0.6, zorder=5)

# Identity line
z_range_19 = [z_arr.min() - 2, z_arr.max() + 2]
ax.plot(z_range_19, z_range_19, '--', color='#ffdd57', linewidth=1.5, alpha=0.7,
        label='y = x')

# Linear fit
slope, intercept, r_value, _, std_err = stats.linregress(
    af_vals[valid_mask], opt_ratio_vals[valid_mask])
fit_x = np.array(z_range_19)
fit_y = slope * fit_x + intercept
ax.plot(fit_x, fit_y, '-', color='#00ff88', linewidth=1.5, alpha=0.8,
        label=f'fit: slope={slope:.3f}, R²={r_value**2:.3f}')

# Annotate each point with its number
for i in range(len(pv)):
    if valid_mask[i]:
        ax.annotate(str(pv[i]), (af_vals[i], opt_ratio_vals[i]),
                    fontsize=12, color='white', alpha=0.8, fontweight='bold',
                    textcoords='offset points', xytext=(6, 6))

cbar = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label('Point Number', fontsize=18)
cbar.ax.tick_params(labelsize=14)

ax.set_xlabel('Autofocus Z — short exposure (µm)', fontsize=20)
ax.set_ylabel('Optimal Z — ratio score, long exposure (µm)', fontsize=20)
ax.set_title(f'Optimal Z (Ratio Score) vs Autofocus Z\n'
             f'slope = {slope:.3f} ± {std_err:.3f}  |  '
             f'intercept = {intercept:.1f} µm  |  R² = {r_value**2:.3f}',
             fontsize=20)
ax.legend(fontsize=16, loc='upper left')
ax.set_xlim(z_range_19)
ax.set_ylim(z_range_19)
ax.set_aspect('equal')
ax.tick_params(labelsize=16)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "fig19_optimal_vs_autofocus.png", dpi=300,
            facecolor='black', bbox_inches='tight')
plt.close(fig)

# ── Figure 20: Ratio vs Z + Optimal vs Autofocus side by side (all points) ───
print("  Figure 20: Ratio vs Z + Optimal vs Autofocus (all points)...")
fig, (ax_left, ax_right) = setup_dark_figure(nrows=1, ncols=2, figsize=(22, 10))

# Shared colormap by point number
all_point_nums = np.array(z_points)
pt_norm = plt.Normalize(vmin=all_point_nums.min(), vmax=all_point_nums.max())

# Left: Ratio score vs Z for all points
for p in z_points:
    color = plt.cm.turbo(pt_norm(p))
    ax_left.plot(z_arr, all_metric_scores['ratio'][p], color=color, linewidth=1.5,
                 alpha=0.7, marker='o', markersize=4)

ax_left.set_xlabel('Z Position (µm)', fontsize=20)
ax_left.set_ylabel('Ratio Score (quartz / laser)', fontsize=20)
ax_left.set_title('Focus Ratio Score vs Z Position', fontsize=20)
ax_left.tick_params(labelsize=16)

# Right: Optimal Z vs Autofocus Z
opt_ratio_all = np.array([best_z_by_metric['ratio'].get(p, np.nan) for p in points_valid])
valid_all = ~np.isnan(opt_ratio_all)
pv_arr = np.array(points_valid)

sc = ax_right.scatter(af_vals[valid_all], opt_ratio_all[valid_all],
                      c=pv_arr[valid_all], cmap='turbo', norm=pt_norm,
                      s=200, edgecolors='white', linewidth=0.6, zorder=5)

ax_right.plot(z_range_19, z_range_19, '--', color='#ffdd57', linewidth=1.5,
              alpha=0.7, label='y = x')
fit_x = np.array(z_range_19)
fit_y = slope * fit_x + intercept
ax_right.plot(fit_x, fit_y, '-', color='#00ff88', linewidth=1.5, alpha=0.8,
              label=f'fit: slope={slope:.3f}, R²={r_value**2:.3f}')

for i in range(len(pv_arr)):
    if valid_all[i]:
        ax_right.annotate(str(pv_arr[i]), (af_vals[i], opt_ratio_all[i]),
                          fontsize=12, color='white', alpha=0.8, fontweight='bold',
                          textcoords='offset points', xytext=(6, 6))

ax_right.set_xlabel('Autofocus Z (µm)', fontsize=20)
ax_right.set_ylabel('Optimal Z — ratio score (µm)', fontsize=20)
ax_right.set_title(f'Optimal Z vs Autofocus Z\n'
                   f'slope={slope:.3f}  |  R²={r_value**2:.3f}', fontsize=20)
ax_right.legend(fontsize=16, loc='upper left')
ax_right.set_xlim(z_range_19)
ax_right.set_ylim(z_range_19)
ax_right.set_aspect('equal')
ax_right.tick_params(labelsize=16)

# Shared colorbar
sm = plt.cm.ScalarMappable(cmap=plt.cm.turbo, norm=pt_norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=[ax_left, ax_right], shrink=0.6, pad=0.02)
cbar.set_label('Point Number', fontsize=18)
cbar.ax.tick_params(labelsize=14)

fig.suptitle('Autofocus Validation — All Points', fontsize=22, y=1.01)
fig.savefig(OUTPUT_DIR / "fig20_ratio_and_optimal_all.png", dpi=300,
            facecolor='black', bbox_inches='tight')
plt.close(fig)

# ── Figure 21: Same but for point 42 only ────────────────────────────────────
print("  Figure 21: Ratio vs Z + Optimal vs Autofocus (point 42)...")
p42 = 42
fig, (ax_left, ax_right) = setup_dark_figure(nrows=1, ncols=2, figsize=(22, 10))
color_42 = plt.cm.turbo(pt_norm(p42))

# Left: Ratio score vs Z for point 42
scores_42 = all_metric_scores['ratio'][p42]
ax_left.plot(z_arr, scores_42, color=color_42, linewidth=2.5, marker='o',
             markersize=10, zorder=5)

# Dashed lines for autofocus Z and min ratio Z
af_z42 = autofocus_z[p42]
best_idx_42 = np.nanargmin(scores_42)
best_z42 = z_arr[best_idx_42]
best_score_42 = scores_42[best_idx_42]

ax_left.axvline(af_z42, color='#ffdd57', linestyle='--', linewidth=2, alpha=0.9,
                label=f'Autofocus Z = {af_z42:.0f} µm')
ax_left.axvline(best_z42, color='#ff6b6b', linestyle='--', linewidth=2, alpha=0.9,
                label=f'Min ratio Z = {best_z42:.0f} µm')
ax_left.axhline(best_score_42, color='#ff6b6b', linestyle=':', linewidth=1.5, alpha=0.6)

ax_left.set_xlabel('Z Position (µm)', fontsize=20)
ax_left.set_ylabel('Ratio Score (quartz / laser)', fontsize=20)
coords_42 = point_coords[p42]
ax_left.set_title(f'Ratio Score vs Z — Point {p42}\n'
                  f'Ring {point_ring[p42]}  |  ({coords_42[0]:.0f}, {coords_42[1]:.0f}) µm  |  '
                  f'd = {point_dist[p42]:.0f} µm', fontsize=20)
ax_left.legend(fontsize=14, loc='best')
ax_left.tick_params(labelsize=16)

# Right: Highlight point 42 on the optimal vs autofocus scatter (all points grey)
ax_right.scatter(af_vals[valid_all], opt_ratio_all[valid_all],
                 c='grey', s=80, alpha=0.3, edgecolors='white', linewidth=0.3)
ax_right.scatter(af_z42, best_z42, c=[color_42], s=300,
                 edgecolors='white', linewidth=1, zorder=10)
ax_right.annotate(f'Point {p42}', (af_z42, best_z42),
                  fontsize=16, color='white', fontweight='bold',
                  textcoords='offset points', xytext=(10, 10),
                  arrowprops=dict(arrowstyle='->', color='white', linewidth=1.5))

ax_right.plot(z_range_19, z_range_19, '--', color='#ffdd57', linewidth=1.5,
              alpha=0.7, label='y = x')
ax_right.plot(fit_x, fit_y, '-', color='#00ff88', linewidth=1.5, alpha=0.8,
              label=f'fit: slope={slope:.3f}')

ax_right.set_xlabel('Autofocus Z (µm)', fontsize=20)
ax_right.set_ylabel('Optimal Z — ratio score (µm)', fontsize=20)
ax_right.set_title(f'Point {p42} Highlighted\n'
                   f'Autofocus={af_z42:.0f}  |  Optimal={best_z42:.0f}  |  '
                   f'Δ={best_z42 - af_z42:.0f} µm', fontsize=20)
ax_right.legend(fontsize=16, loc='upper left')
ax_right.set_xlim(z_range_19)
ax_right.set_ylim(z_range_19)
ax_right.set_aspect('equal')
ax_right.tick_params(labelsize=16)

fig.suptitle(f'Autofocus Validation — Point {p42}', fontsize=22, y=1.01)
fig.savefig(OUTPUT_DIR / "fig21_ratio_and_optimal_point42.png", dpi=300,
            facecolor='black', bbox_inches='tight')
plt.close(fig)

print(f"\nAll figures saved to {OUTPUT_DIR}/")
print("Done!")
