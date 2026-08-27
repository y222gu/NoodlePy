"""Plot separate hyperspectral maps for each sample/droplet."""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# Paths
json_path = Path("/Users/yifeigu/Documents/Carney_Lab/RamanPy/data/spectra to plot_10_droplet.json")
experiment_path = Path("/Users/yifeigu/Documents/Carney_Lab/RamanPy/data")
output_dir = Path("/Users/yifeigu/Documents/Carney_Lab/RamanPy/data/sample_maps")

# Laser wavelength for Raman shift calculation
LASER_WAVELENGTH_NM = 785.0

# Target Raman shift for the map (cm^-1)
TARGET_RAMAN_SHIFT = 1009.8

# Color mode: 'grey', 'intensity', or 'random'
COLOR_MODE = 'intensity'

def wavelength_to_raman_shift(wavelength_nm, laser_nm=785.0):
    """Convert scattered wavelength (nm) to Raman shift (cm^-1)."""
    return (1.0 / laser_nm - 1.0 / wavelength_nm) * 1e7

# Create output directory
output_dir.mkdir(parents=True, exist_ok=True)

# Load metadata
print("Loading metadata...")
with open(json_path, 'r') as f:
    metadata_list = json.load(f)

print(f"Found {len(metadata_list)} spectra")

# Group spectra by droplet_id
droplets = defaultdict(list)
for meta in metadata_list:
    droplet_id = meta.get('droplet_id', 0)
    droplets[droplet_id].append(meta)

print(f"Found {len(droplets)} droplets/samples")

# Process each droplet
for droplet_id in sorted(droplets.keys()):
    droplet_spectra = droplets[droplet_id]

    # Get sample info from first spectrum
    first_meta = droplet_spectra[0]
    patient = first_meta.get('patient', 'unknown')
    sample = first_meta.get('sample', 'unknown')
    staging = first_meta.get('staging', 'unknown')
    experiment_folder = first_meta.get('experiment_folder', first_meta.get('date', 'unknown'))

    print(f"\nProcessing Droplet {droplet_id}: Patient {patient}, Sample {sample}, {staging}")

    # Determine spectra folder
    spectra_folder = experiment_path / experiment_folder / "spectra"
    if not spectra_folder.exists():
        # Try alternative path
        spectra_folder = experiment_path / experiment_folder
        if not spectra_folder.exists():
            print(f"  Warning: Spectra folder not found, skipping...")
            continue

    # Load spectra for this droplet
    x_positions = []
    y_positions = []
    intensities_at_target = []

    for meta in droplet_spectra:
        spectrum_file = spectra_folder / meta['spectrum_file']

        if not spectrum_file.exists():
            continue

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

        # Find intensity at target Raman shift
        idx = np.argmin(np.abs(raman_shifts - TARGET_RAMAN_SHIFT))
        intensities_at_target.append(intensities[idx])

    if len(x_positions) == 0:
        print(f"  No spectra loaded, skipping...")
        continue

    print(f"  Loaded {len(x_positions)} spectra")

    # Convert to arrays
    x_positions = np.array(x_positions)
    y_positions = np.array(y_positions)
    intensities_at_target = np.array(intensities_at_target)

    # Create figure with black background
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # Plot based on color mode
    if COLOR_MODE == 'grey':
        ax.scatter(x_positions, y_positions, c='grey', s=100, alpha=1)
        title = f'Droplet {droplet_id}: Patient {patient}, Sample {sample}\n({staging}, n={len(x_positions)})'

    elif COLOR_MODE == 'intensity':
        scatter = ax.scatter(
            x_positions, y_positions,
            c=intensities_at_target,
            cmap='viridis',
            s=100,
            alpha=1
        )
        title = f'Droplet {droplet_id}: Patient {patient}, Sample {sample}\n({staging}, n={len(x_positions)}, at {TARGET_RAMAN_SHIFT} cm$^{{-1}}$)'

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label(f'Intensity at {TARGET_RAMAN_SHIFT} cm$^{{-1}}$', color='white', fontsize=14)
        cbar.ax.yaxis.set_tick_params(color='white', labelsize=12)
        cbar.outline.set_edgecolor('white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    elif COLOR_MODE == 'random':
        np.random.seed(droplet_id)
        colors = np.random.rand(len(x_positions), 3)
        ax.scatter(x_positions, y_positions, c=colors, s=100, alpha=1)
        title = f'Droplet {droplet_id}: Patient {patient}, Sample {sample}\n({staging}, n={len(x_positions)})'

    # Style the plot
    ax.set_xlabel('X Position (μm)', color='white', fontsize=16)
    ax.set_ylabel('Y Position (μm)', color='white', fontsize=16)
    ax.set_title(title, color='white', fontsize=18)

    ax.set_aspect('equal')
    ax.tick_params(axis='both', colors='white', labelsize=14)

    for spine in ax.spines.values():
        spine.set_color('white')

    plt.tight_layout()

    # Save
    output_path = output_dir / f"droplet_{droplet_id}_patient_{patient}_sample_{sample}.png"
    print(f"  Saving to {output_path}...")
    plt.savefig(output_path, dpi=300, facecolor='black', edgecolor='none', bbox_inches='tight')
    plt.close()

print("\nDone!")
print(f"All plots saved to: {output_dir}")
