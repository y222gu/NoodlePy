"""Create animated GIF with hyperspectral map and raw spectra side by side."""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
from scipy.signal import medfilt, savgol_filter
import pybaselines

# Paths
json_path = Path("/Users/yifeigu/Documents/Carney_Lab/RamanPy/data/spectra to plot.json")
experiment_path = Path("/Users/yifeigu/Documents/Carney_Lab/RamanPy/data/2026_01_29_3")
spectra_folder = experiment_path / "spectra"
output_path = Path("/Users/yifeigu/Documents/Carney_Lab/RamanPy/data/hyperspectral_animation.gif")

# Laser wavelength for Raman shift calculation
LASER_WAVELENGTH_NM = 785.0

# Raman shift range for animation (cm^-1)
RAMAN_SHIFT_START = 400
RAMAN_SHIFT_END = 3200
RAMAN_SHIFT_STEP = 20  # Step size for animation frames

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
BASELINE_CORRECTION = False
BASELINE_LAM = 1000
BASELINE_DIFF_ORDER = 1
BASELINE_MAX_ITER = 15
BASELINE_TOL = 0.005

# Smoothing (Savitzky-Golay)
SMOOTHING = False
SMOOTH_WINDOW = 9
SMOOTH_POLYORDER = 2

# Normalization
NORMALIZATION = False
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

print(f"Found {len(metadata_list)} spectra")

# Load all spectra data
x_positions = []
y_positions = []
all_raman_shifts = None
all_spectra = []

print("Loading spectra...")
if PREPROCESSING_ENABLED:
    print(f"Preprocessing enabled: cropping={CROPPING}, cosmic_rays={REMOVE_COSMIC_RAYS}, "
          f"baseline={BASELINE_CORRECTION}, smoothing={SMOOTHING}, normalization={NORMALIZATION}")

for i, meta in enumerate(metadata_list):
    spectrum_file = spectra_folder / meta['spectrum_file']

    if not spectrum_file.exists():
        continue

    # Get x, y position from metadata
    x_positions.append(meta['x'])
    y_positions.append(meta['y'])

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

    all_spectra.append(intensities)

    if (i + 1) % 200 == 0:
        print(f"  Loaded {i + 1}/{len(metadata_list)} spectra...")

print(f"Loaded {len(x_positions)} spectra successfully")

# Convert to arrays
x_positions = np.array(x_positions)
y_positions = np.array(y_positions)
all_spectra = np.array(all_spectra)  # Shape: (n_spectra, n_points)

print(f"Spectra shape: {all_spectra.shape}")
print(f"Raman shift range: {all_raman_shifts.min():.1f} - {all_raman_shifts.max():.1f} cm^-1")

# Generate frames for animation (forward and backward for smooth loop)
# Adjust range based on actual spectrum range after preprocessing
anim_start = max(RAMAN_SHIFT_START, all_raman_shifts.min())
anim_end = min(RAMAN_SHIFT_END, all_raman_shifts.max())
raman_shift_forward = np.arange(anim_start, anim_end + 1, RAMAN_SHIFT_STEP)
raman_shift_backward = raman_shift_forward[::-1][1:-1]  # Reverse, excluding endpoints to avoid duplicates
raman_shift_frames = np.concatenate([raman_shift_forward, raman_shift_backward])
n_frames = len(raman_shift_frames)
print(f"Animation range: {anim_start:.1f} - {anim_end:.1f} cm^-1")
print(f"Creating animation with {n_frames} frames (forward + backward)...")

# Create figure with black background
plt.style.use('dark_background')
fig, (ax_spectra, ax_map) = plt.subplots(1, 2, figsize=(16, 6), dpi=100)

fig.patch.set_facecolor('black')
ax_spectra.set_facecolor('black')
ax_map.set_facecolor('black')

# Pre-compute intensity range for consistent colormap
all_intensities_flat = all_spectra.flatten()
vmin_global = np.percentile(all_intensities_flat, 5)
vmax_global = np.percentile(all_intensities_flat, 95)

# Plot spectra once (static background)
# Sort by mean intensity for better visualization
mean_intensities = all_spectra.mean(axis=1)
sort_indices = np.argsort(mean_intensities)

# Use viridis colormap for spectra based on mean intensity
cmap_spectra = plt.cm.viridis
norm_spectra = plt.Normalize(mean_intensities.min(), mean_intensities.max())

for idx in sort_indices:
    color = cmap_spectra(norm_spectra(mean_intensities[idx]))
    ax_spectra.plot(all_raman_shifts, all_spectra[idx], color=color, linewidth=0.3, alpha=0.5)

ax_spectra.set_xlim(all_raman_shifts.min(), all_raman_shifts.max())

# Set y-axis limits based on normalization
if NORMALIZATION and PREPROCESSING_ENABLED:
    ax_spectra.set_ylim(-0.1, 1.1)
else:
    ax_spectra.set_ylim(750, 2300)

ax_spectra.set_xlabel('Raman Shift (cm$^{-1}$)', color='white', fontsize=14)
ax_spectra.set_ylabel('Intensity (a.u.)', color='white', fontsize=14)
ax_spectra.tick_params(axis='both', colors='white', labelsize=10)
for spine in ax_spectra.spines.values():
    spine.set_color('white')

# Initial vertical line on spectra plot (more visible)
vline = ax_spectra.axvline(x=raman_shift_frames[0], color='red', linestyle='-', linewidth=3, alpha=0.9)

# Title for spectra
title_suffix = " (Preprocessed)" if PREPROCESSING_ENABLED else " (Raw)"
spectra_title = ax_spectra.set_title(f'Spectra (n={len(all_spectra)}){title_suffix}', color='white', fontsize=16)

# Initial scatter plot for map
initial_rs = raman_shift_frames[0]
rs_idx = np.argmin(np.abs(all_raman_shifts - initial_rs))
intensities_at_rs = all_spectra[:, rs_idx]

scatter = ax_map.scatter(
    x_positions, y_positions,
    c=intensities_at_rs,
    cmap='viridis',
    s=100,
    alpha=1,
    vmin=vmin_global,
    vmax=vmax_global
)

ax_map.set_xlabel('X Position (μm)', color='white', fontsize=14)
ax_map.set_ylabel('Y Position (μm)', color='white', fontsize=14)
ax_map.set_aspect('equal')
ax_map.tick_params(axis='both', colors='white', labelsize=10)
for spine in ax_map.spines.values():
    spine.set_color('white')

# Title for map (will be updated)
map_title = ax_map.set_title(f'Hyperspectral Map at {initial_rs:.1f} cm$^{{-1}}$', color='white', fontsize=16)

# Add colorbar for map                                                                                                                                                                                    
cbar = plt.colorbar(scatter, ax=ax_map, shrink=0.8, pad=0.02)                                                                                                                                             
cbar.set_label('Intensity', color='white', fontsize=12)                                                                                                                                                   
cbar.ax.yaxis.set_tick_params(color='white', labelsize=9)                                                                                                                                                 
cbar.outline.set_edgecolor('white')                                                                                                                                                                       
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')                                                                                                                                            
    
plt.tight_layout(pad=0.5, w_pad=1.0)

def update(frame):
    """Update function for animation."""
    rs = raman_shift_frames[frame]

    # Update vertical line position
    vline.set_xdata([rs, rs])

    # Find nearest Raman shift index
    rs_idx = np.argmin(np.abs(all_raman_shifts - rs))
    intensities_at_rs = all_spectra[:, rs_idx]

    # Update scatter plot colors
    scatter.set_array(intensities_at_rs)

    # Update map title
    map_title.set_text(f'Hyperspectral Map at {rs:.1f} cm$^{{-1}}$')

    if (frame + 1) % 20 == 0:
        print(f"  Frame {frame + 1}/{n_frames}")

    return vline, scatter, map_title

# Create animation
print("Creating animation...")
anim = FuncAnimation(fig, update, frames=n_frames, interval=100, blit=False)

# Save as GIF
print(f"Saving to {output_path}...")
writer = PillowWriter(fps=50)
anim.save(output_path, writer=writer, dpi=100)
plt.close()

print("Done!")
