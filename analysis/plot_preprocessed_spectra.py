"""Plot preprocessed spectra with configurable coloring."""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import medfilt, savgol_filter
import pybaselines

# Paths
json_path = Path("/Users/yifeigu/Documents/Carney_Lab/RamanPy/data/6_patients.json")
experiment_path = Path("/Users/yifeigu/Documents/Carney_Lab/Data/RamanRobot/2025_12_10")
spectra_folder = experiment_path / "spectra"
output_path = Path("/Users/yifeigu/Documents/Carney_Lab/RamanPy/data/preprocessed_spectra_plot.png")

# Laser wavelength for Raman shift calculation
LASER_WAVELENGTH_NM = 785.0

# Target Raman shift for coloring (cm^-1) - used when COLOR_MODE = 'intensity'
TARGET_RAMAN_SHIFT = 1009.8

# Color mode: 'grey', 'intensity', or 'random'
COLOR_MODE = 'random'

# ============ PREPROCESSING OPTIONS (same defaults as spectrum viewer) ============
PREPROCESSING_ENABLED = True  # Master switch for all preprocessing

# Cropping
CROPPING = True
CROP_START_CM = 662.697
CROP_END_CM = 1784.104

# Cosmic ray removal (modified z-scores)
REMOVE_COSMIC_RAYS = True
COSMIC_THRESHOLD = 10

# Baseline correction (airPLS)
BASELINE_CORRECTION = True
BASELINE_LAM = 100
BASELINE_DIFF_ORDER = 1
BASELINE_MAX_ITER = 15
BASELINE_TOL = 0.005

# Smoothing (Savitzky-Golay)
SMOOTHING = True
SMOOTH_WINDOW = 5
SMOOTH_POLYORDER = 3

# Normalization
NORMALIZATION = True
NORMALIZATION_TYPE = 'by_max'  # 'by_max' or 'by_area'
# =================================================================================

def wavelength_to_raman_shift(wavelength_nm, laser_nm=785.0):
    """Convert scattered wavelength (nm) to Raman shift (cm^-1)."""
    return (1.0 / laser_nm - 1.0 / wavelength_nm) * 1e7

def remove_cosmic_rays(intensity, threshold=10):
    """Remove cosmic ray spikes using median filtering and thresholding."""
    median_filtered = medfilt(intensity, kernel_size=5)
    residual = intensity - median_filtered
    std_dev = np.std(residual)
    cosmic_ray_mask = np.abs(residual) > (threshold * std_dev)
    result = intensity.copy()
    result[cosmic_ray_mask] = median_filtered[cosmic_ray_mask]
    return result

def airpls_baseline(intensity, raman_shifts, lam=1000, diff_order=1, max_iter=15, tol=0.005):
    """Remove baseline using airPLS algorithm."""
    baseline_fitter = pybaselines.Baseline(x_data=raman_shifts)
    baseline, _ = baseline_fitter.airpls(intensity, lam, diff_order, max_iter, tol)
    return intensity - baseline

def normalize_spectrum(intensity, raman_shifts, norm_type='by_max'):
    """Normalize spectrum."""
    if norm_type == 'by_area':
        return intensity / np.trapz(intensity, raman_shifts)
    elif norm_type == 'by_max':
        return (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity))
    return intensity

def preprocess_spectrum(intensity, raman_shifts):
    """Apply preprocessing steps to a spectrum."""
    if not PREPROCESSING_ENABLED:
        return intensity, raman_shifts

    result = intensity.copy()
    rs = raman_shifts.copy()

    # 1. Remove cosmic rays (before cropping)
    if REMOVE_COSMIC_RAYS:
        result = remove_cosmic_rays(result, COSMIC_THRESHOLD)

    # 2. Cropping
    if CROPPING:
        mask = (rs >= CROP_START_CM) & (rs <= CROP_END_CM)
        if np.any(mask):
            rs = rs[mask]
            result = result[mask]

    # 3. Baseline correction
    if BASELINE_CORRECTION:
        result = airpls_baseline(result, rs, BASELINE_LAM, BASELINE_DIFF_ORDER,
                                  BASELINE_MAX_ITER, BASELINE_TOL)

    # 4. Smoothing
    if SMOOTHING:
        result = savgol_filter(result, SMOOTH_WINDOW, SMOOTH_POLYORDER)

    # 5. Normalization
    if NORMALIZATION:
        result = normalize_spectrum(result, rs, NORMALIZATION_TYPE)

    return result, rs

# Load metadata
print("Loading metadata...")
with open(json_path, 'r') as f:
    metadata_list = json.load(f)

print(f"Found {len(metadata_list)} spectra to plot")

if PREPROCESSING_ENABLED:
    print(f"Preprocessing: cropping={CROPPING}, cosmic_rays={REMOVE_COSMIC_RAYS}, "
          f"baseline={BASELINE_CORRECTION}, smoothing={SMOOTHING}, normalization={NORMALIZATION}")

# Load spectra
spectra_data = []
intensities_at_target = []
all_raman_shifts = None

print("Loading spectra...")
for i, meta in enumerate(metadata_list):
    spectrum_file = spectra_folder / meta['spectrum_file']

    if not spectrum_file.exists():
        print(f"Warning: File not found: {spectrum_file}")
        continue

    # Load spectrum
    data = np.loadtxt(spectrum_file, delimiter=',')
    n_points = 1024
    n_reps = len(data) // n_points

    # Convert wavelength to Raman shift
    wavelengths_nm = data[:n_points, 0]
    raman_shifts = wavelength_to_raman_shift(wavelengths_nm, LASER_WAVELENGTH_NM)

    # Take median of intensities across repetitions
    all_intensities = data[:, 1].reshape(n_reps, n_points)
    intensities = np.median(all_intensities, axis=0)

    # Apply preprocessing
    intensities, raman_shifts = preprocess_spectrum(intensities, raman_shifts)

    if all_raman_shifts is None:
        all_raman_shifts = raman_shifts

    # Find intensity at target Raman shift (for intensity coloring)
    idx = np.argmin(np.abs(raman_shifts - TARGET_RAMAN_SHIFT))
    intensity_at_target = intensities[idx]

    spectra_data.append((raman_shifts, intensities))
    intensities_at_target.append(intensity_at_target)

    if (i + 1) % 100 == 0:
        print(f"  Loaded {i + 1}/{len(metadata_list)} spectra...")

print(f"Loaded {len(spectra_data)} spectra successfully")
print(f"Raman shift range: {all_raman_shifts.min():.1f} - {all_raman_shifts.max():.1f} cm^-1")

# Convert to array for normalization
intensities_at_target = np.array(intensities_at_target)

# Normalize intensities to [0, 1] for colormap
vmin, vmax = intensities_at_target.min(), intensities_at_target.max()
normalized_intensities = (intensities_at_target - vmin) / (vmax - vmin)

print(f"Intensity range at {TARGET_RAMAN_SHIFT} cm^-1: {vmin:.4f} - {vmax:.4f}")

# Create figure with black background
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

# Set black background
fig.patch.set_facecolor('black')
ax.set_facecolor('black')

# Plot each spectrum based on color mode
print(f"Plotting spectra (color mode: {COLOR_MODE})...")

if COLOR_MODE == 'grey':
    for wavenumbers, intensities in spectra_data:
        ax.plot(wavenumbers, intensities, color='grey', linewidth=0.3, alpha=0.5)
    title = f'Preprocessed Spectra (n={len(spectra_data)})'

elif COLOR_MODE == 'intensity':
    cmap = plt.cm.viridis
    # Sort spectra by intensity so brightest are plotted last (on top)
    sort_indices = np.argsort(intensities_at_target)
    for idx in sort_indices:
        wavenumbers, intensities = spectra_data[idx]
        color = cmap(normalized_intensities[idx])
        ax.plot(wavenumbers, intensities, color=color, linewidth=0.3, alpha=0.7)
    title = f'Preprocessed Spectra (n={len(spectra_data)})\nColored by intensity at {TARGET_RAMAN_SHIFT} cm$^{{-1}}$'

elif COLOR_MODE == 'random':
    np.random.seed(42)  # For reproducibility
    colors = plt.cm.turbo(np.linspace(0, 1, 20))
    for i, (wavenumbers, intensities) in enumerate(spectra_data):
        color = colors[i % 20]
        ax.plot(wavenumbers, intensities, color=color, linewidth=0.3, alpha=0.5)
    title = f'Preprocessed Spectra (n={len(spectra_data)})'

# # Set y-axis limits based on normalization
if NORMALIZATION and PREPROCESSING_ENABLED:
    ax.set_ylim(-0.15, 1.1)

# Style the plot
ax.set_xlabel('Raman Shift (cm$^{-1}$)', color='white', fontsize=18)
ax.set_ylabel('Intensity (a.u.)', color='white', fontsize=18)
ax.set_title(title, color='white', fontsize=20)

# White spines and ticks
ax.spines['bottom'].set_color('white')
ax.spines['top'].set_color('white')
ax.spines['left'].set_color('white')
ax.spines['right'].set_color('white')
ax.tick_params(axis='both', colors='white', labelsize=14)

# Add colorbar for intensity mode
if COLOR_MODE == 'intensity':
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(f'Intensity at {TARGET_RAMAN_SHIFT} cm$^{{-1}}$', color='white', fontsize=16)
    cbar.ax.yaxis.set_tick_params(color='white', labelsize=12)
    cbar.outline.set_edgecolor('white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white', fontsize=12)

plt.tight_layout()

# Save
print(f"Saving to {output_path}...")
plt.savefig(output_path, dpi=300, facecolor='black', edgecolor='none', bbox_inches='tight')
plt.close()

print("Done!")
