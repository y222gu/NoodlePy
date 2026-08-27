"""Plot hyperspectral map with configurable coloring - separate plot per sample."""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# Paths
json_path = Path("/Users/yifeigu/Documents/Carney_Lab/RamanPy/data/all.json")
data_path = Path("/Users/yifeigu/Documents/Carney_Lab/RamanPy/data/2026_01_29_3")
output_dir = Path("/Users/yifeigu/Documents/Carney_Lab/RamanPy/data/hyperspectral_maps")
output_dir.mkdir(parents=True, exist_ok=True)

# Laser wavelength for Raman shift calculation
LASER_WAVELENGTH_NM = 785.0

# Target Raman shift for the map (cm^-1)
TARGET_RAMAN_SHIFT = 1009.8

# Color mode: 'grey', 'intensity', or 'random'
COLOR_MODE = 'grey'

# Figure size
FIG_SIZE = (10, 8)

def wavelength_to_raman_shift(wavelength_nm, laser_nm=785.0):
    """Convert scattered wavelength (nm) to Raman shift (cm^-1)."""
    return (1.0 / laser_nm - 1.0 / wavelength_nm) * 1e7

# Load metadata
print("Loading metadata...")
with open(json_path, 'r') as f:
    metadata_list = json.load(f)

print(f"Found {len(metadata_list)} spectra")

# Group spectra by sample/droplet
samples = defaultdict(lambda: {
    'x': [], 'y': [], 'intensities': [], 'info': None
})

print("Loading spectra...")
loaded_count = 0
for i, meta in enumerate(metadata_list):
    # Get experiment folder from metadata
    experiment_folder = meta.get('experiment_folder', meta.get('date', ''))

    # Try multiple path patterns
    spectrum_file = None
    for spectra_folder in [
        data_path / experiment_folder / "spectra",  # Nested: data_path/experiment_folder/spectra
        data_path / "spectra",                       # Direct: data_path/spectra
    ]:
        candidate = spectra_folder / meta['spectrum_file']
        if candidate.exists():
            spectrum_file = candidate
            break

    if spectrum_file is None:
        continue

    # Get sample identifier
    sample_id = meta.get('droplet_id', meta.get('sample', 0))

    # Store sample info
    if samples[sample_id]['info'] is None:
        samples[sample_id]['info'] = {
            'patient': meta.get('patient', 'unknown'),
            'sample': meta.get('sample', sample_id),
            'staging': meta.get('staging', 'unknown'),
            'experiment_folder': experiment_folder
        }

    # Get x, y position from metadata
    samples[sample_id]['x'].append(meta['x'])
    samples[sample_id]['y'].append(meta['y'])

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

    # Find intensity at target Raman shift
    idx = np.argmin(np.abs(raman_shifts - TARGET_RAMAN_SHIFT))
    samples[sample_id]['intensities'].append(intensities[idx])

    loaded_count += 1
    if (i + 1) % 200 == 0:
        print(f"  Loaded {i + 1}/{len(metadata_list)} spectra...")

print(f"Loaded {loaded_count} spectra from {len(samples)} samples")

# Plot each sample separately
print(f"\nPlotting hyperspectral maps (color mode: {COLOR_MODE})...")
plt.style.use('dark_background')

for sample_id in sorted(samples.keys()):
    sample_data = samples[sample_id]
    info = sample_data['info']

    x_positions = np.array(sample_data['x'])
    y_positions = np.array(sample_data['y'])
    intensities_at_target = np.array(sample_data['intensities'])

    if len(x_positions) == 0:
        continue

    # Create figure
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=300)
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # Plot based on color mode
    if COLOR_MODE == 'grey':
        ax.scatter(x_positions, y_positions, c='grey', s=100, alpha=1)

    elif COLOR_MODE == 'intensity':
        scatter = ax.scatter(
            x_positions, y_positions,
            c=intensities_at_target,
            cmap='viridis',
            s=100,
            alpha=1
        )

    elif COLOR_MODE == 'random':
        np.random.seed(sample_id)
        colors = np.random.rand(len(x_positions), 3)
        ax.scatter(x_positions, y_positions, c=colors, s=100, alpha=1)

    # Style the plot
    ax.set_xlabel('X Position (μm)', color='white', fontsize=14)
    ax.set_ylabel('Y Position (μm)', color='white', fontsize=14)

    title = f"Patient {info['patient']}, Sample {info['sample']}\n({info['staging']}, n={len(x_positions)})"
    ax.set_title(title, color='white', fontsize=14)

    ax.set_aspect('equal')
    ax.tick_params(axis='both', colors='white', labelsize=10)

    for spine in ax.spines.values():
        spine.set_color('white')

    plt.tight_layout()

    # Save
    output_path = output_dir / f"sample_{sample_id}_patient_{info['patient']}_sample_{info['sample']}.png"
    print(f"  Saving {output_path.name}...")
    plt.savefig(output_path, dpi=300, facecolor='black', edgecolor='none', bbox_inches='tight')
    plt.close()

print(f"\nDone! Saved {len(samples)} maps to: {output_dir}")
