"""Plot selected spectra with configurable coloring."""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
json_path = Path("/Users/yifeigu/Documents/Carney_Lab/RamanPy/data/spectra to plot.json")
experiment_path = Path("/Users/yifeigu/Documents/Carney_Lab/RamanPy/data/2026_01_29_3")
spectra_folder = experiment_path / "spectra"
output_path = Path("/Users/yifeigu/Documents/Carney_Lab/RamanPy/data/selected_spectra_plot.png")

# Laser wavelength for Raman shift calculation
LASER_WAVELENGTH_NM = 785.0

# Target Raman shift for coloring (cm^-1)
TARGET_RAMAN_SHIFT = 1009.8

# Color mode: 'grey', 'intensity', or 'random'
COLOR_MODE = 'random'

def wavelength_to_raman_shift(wavelength_nm, laser_nm=785.0):
    """Convert scattered wavelength (nm) to Raman shift (cm^-1)."""
    return (1.0 / laser_nm - 1.0 / wavelength_nm) * 1e7

# Load metadata
print("Loading metadata...")
with open(json_path, 'r') as f:
    metadata_list = json.load(f)

print(f"Found {len(metadata_list)} spectra to plot")

# Load spectra and find intensities at target wavenumber
spectra_data = []
intensities_at_target = []

print("Loading spectra...")
for i, meta in enumerate(metadata_list):
    spectrum_file = spectra_folder / meta['spectrum_file']

    if not spectrum_file.exists():
        print(f"Warning: File not found: {spectrum_file}")
        continue

    # Load spectrum (comma-separated: wavelength_nm, intensity)
    # Files contain 3 repetitions concatenated (3 × 1024 = 3072 points)
    # Take median across repetitions for each wavelength
    data = np.loadtxt(spectrum_file, delimiter=',')
    n_points = 1024
    n_reps = len(data) // n_points

    # Use wavelengths from first repetition and convert to Raman shift
    wavelengths_nm = data[:n_points, 0]
    raman_shifts = wavelength_to_raman_shift(wavelengths_nm, LASER_WAVELENGTH_NM)

    # Reshape intensities and take median across repetitions
    all_intensities = data[:, 1].reshape(n_reps, n_points)
    intensities = np.median(all_intensities, axis=0)

    # Find intensity at target Raman shift
    idx = np.argmin(np.abs(raman_shifts - TARGET_RAMAN_SHIFT))
    intensity_at_target = intensities[idx]

    spectra_data.append((raman_shifts, intensities))
    intensities_at_target.append(intensity_at_target)

    if (i + 1) % 100 == 0:
        print(f"  Loaded {i + 1}/{len(metadata_list)} spectra...")

print(f"Loaded {len(spectra_data)} spectra successfully")

# Convert to array for normalization
intensities_at_target = np.array(intensities_at_target)

# Normalize intensities to [0, 1] for colormap
vmin, vmax = intensities_at_target.min(), intensities_at_target.max()
normalized_intensities = (intensities_at_target - vmin) / (vmax - vmin)

print(f"Intensity range at {TARGET_RAMAN_SHIFT} cm^-1: {vmin:.1f} - {vmax:.1f}")

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
    title = f'Raw Spectra (n={len(spectra_data)})'

elif COLOR_MODE == 'intensity':
    cmap = plt.cm.viridis
    # Sort spectra by intensity so brightest are plotted last (on top)
    sort_indices = np.argsort(intensities_at_target)
    for idx in sort_indices:
        wavenumbers, intensities = spectra_data[idx]
        color = cmap(normalized_intensities[idx])
        ax.plot(wavenumbers, intensities, color=color, linewidth=0.3, alpha=0.7)
    title = f'Raw Spectra (n={len(spectra_data)})\nColored by intensity at {TARGET_RAMAN_SHIFT} cm$^{{-1}}$'

elif COLOR_MODE == 'random':
    np.random.seed(42)  # For reproducibility
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    for i, (wavenumbers, intensities) in enumerate(spectra_data):
        color = colors[i % 20]
        ax.plot(wavenumbers, intensities, color=color, linewidth=0.3, alpha=0.5)
    title = f'Raw Spectra (n={len(spectra_data)})'

# Set y-axis limit
ax.set_ylim(750, 2300)

# Style the plot
ax.set_xlabel('Raman Shift (cm$^{-1}$)', color='white', fontsize=24)
ax.set_ylabel('Intensity (a.u.)', color='white', fontsize=24)
ax.set_title(title, color='white', fontsize=24)

# White spines and ticks
ax.spines['bottom'].set_color('white')
ax.spines['top'].set_color('white')
ax.spines['left'].set_color('white')
ax.spines['right'].set_color('white')
ax.tick_params(axis='both', colors='white', labelsize=14)

# # Add colorbar for intensity mode
# if COLOR_MODE == 'intensity':
#     sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=vmin, vmax=vmax))
#     sm.set_array([])
#     cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
#     cbar.set_label(f'Intensity at {TARGET_RAMAN_SHIFT} cm$^{{-1}}$', color='white', fontsize=16)
#     cbar.ax.yaxis.set_tick_params(color='white', labelsize=12)
#     cbar.outline.set_edgecolor('white')
#     plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white', fontsize=12)

# plt.tight_layout()

# Save
print(f"Saving to {output_path}...")
plt.savefig(output_path, dpi=300, facecolor='black', edgecolor='none', bbox_inches='tight')
plt.close()

print("Done!")
