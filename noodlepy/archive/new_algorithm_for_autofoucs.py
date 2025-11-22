import os
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set the folder path
folder_path = r'/Users/yifeigu/Library/CloudStorage/Box-Box/Carney Lab Shared/Data/Raman_Robot/3_bad_repeat_with_more_steps'

# Get all txt files from the folder
files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

# Extract position number and sort files
def extract_position(filename):
    """Extract the position number from the filename"""
    match = re.search(r'position_(\d+\.?\d*)', filename)
    if match:
        return float(match.group(1))
    return float('inf')

# Sort files by position number
sorted_files = sorted(files, key=extract_position)

# Create figure with subplots (one column, multiple rows)
num_files = len(sorted_files)
fig, axes = plt.subplots(num_files, 1, figsize=(10, 2 * num_files), sharex=True, sharey=True)

# Handle case where there's only one file (axes won't be array)
if num_files == 1:
    axes = [axes]

# Plot each spectrum
for idx, filename in enumerate(sorted_files):
    file_path = os.path.join(folder_path, filename)
    
    # Extract position number for the title
    position = extract_position(filename)
    
    # Read the data
    wavelengths = []
    intensities = []
    
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                try:
                    wavelengths.append(float(parts[0]))
                    intensities.append(float(parts[1]))
                except ValueError:
                    continue
    
    # Smooth the spectrum using a moving average with window size 5
    intensities_array = np.array(intensities)
    window_size = 1
    smoothed_intensities = np.convolve(intensities_array, np.ones(window_size)/window_size, mode='valid')
    smoothed_wavelengths = wavelengths[window_size//2:window_size//2+len(smoothed_intensities)]
    
    # Plot
    ax = axes[idx]
    ax.plot(smoothed_wavelengths, smoothed_intensities, 'b-', linewidth=1.5)
    ax.text(0.02, 0.95, f'Position: {position}', transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_yticks([])
    if idx < num_files - 1:
        ax.set_xticks([])

# Only add labels to the bottom and left axes
axes[-1].set_xlabel('Wavelength')
fig.text(0.04, 0.5, 'Intensity', va='center', rotation='vertical')
plt.savefig('spectrum_plot.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Plotted {num_files} spectra in ascending order of position")
print(f"Files plotted in order:")
for filename in sorted_files:
    position = extract_position(filename)
    print(f"  Position {position}: {filename}")

# Extract entropy and plot entropy vs position
def extract_entropy(filename):
    """Extract the entropy value from the filename"""
    match = re.search(r'entropy_(\d+\.?\d*)', filename)
    if match:
        return float(match.group(1))
    return None

positions = []
entropies = []

for filename in sorted_files:
    position = extract_position(filename)
    entropy = extract_entropy(filename)
    if entropy is not None:
        positions.append(position)
        entropies.append(entropy)

# Create entropy vs position plot
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(positions, entropies, 'bo-', linewidth=2, markersize=8)
ax2.set_xlabel('Position', fontsize=12)
ax2.set_ylabel('Entropy', fontsize=12)
ax2.set_title('Entropy vs Position', fontsize=14)
ax2.grid(True, alpha=0.3)

plt.savefig('entropy_vs_position.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nEntropy vs Position:")
for pos, ent in zip(positions, entropies):
    print(f"  Position {pos}: Entropy {ent}")

# Load reference spectrum (focused spectrum)
reference_filename = '/Users/yifeigu/Library/CloudStorage/Box-Box/Carney Lab Shared/Data/Raman_Robot/2025_11_18_copy/sample1_OD408_cancer/Captured_spectrum_4.txt'
reference_path = os.path.join(folder_path, reference_filename)

ref_wavelengths = []
ref_intensities = []

with open(reference_path, 'r') as f:
    for line in f:
        parts = line.strip().split(',')
        if len(parts) >= 2:
            try:
                ref_wavelengths.append(float(parts[0]))
                ref_intensities.append(float(parts[1]))
            except ValueError:
                continue

# Smooth reference spectrum
ref_intensities_array = np.array(ref_intensities)
window_size = 3
ref_smoothed_intensities = np.convolve(ref_intensities_array, np.ones(window_size)/window_size, mode='valid')

# Normalize reference spectrum
ref_smoothed_intensities = ref_smoothed_intensities / np.max(np.abs(ref_smoothed_intensities))

# Calculate focus scores based on convolution and correlation with reference
focus_scores_convolution = []
focus_scores_correlation = []
focus_scores_sharpness = []

# Collect smoothed spectra and compute focus scores, then plot all smoothed spectra together
all_smoothed_wavelengths = []
all_smoothed_intensities = []

# Reset positions to align with focus scores (this won't affect earlier plots already created)
positions = []

for filename in sorted_files:
    file_path = os.path.join(folder_path, filename)
    position = extract_position(filename)
    positions.append(position)
    
    # Read the data
    wavelengths = []
    intensities = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                try:
                    wavelengths.append(float(parts[0]))
                    intensities.append(float(parts[1]))
                except ValueError:
                    continue

    if len(intensities) == 0:
        # skip empty files
        focus_scores_convolution.append(0.0)
        focus_scores_correlation.append(0.0)
        focus_scores_sharpness.append(0.0)
        all_smoothed_wavelengths.append(np.array([]))
        all_smoothed_intensities.append(np.array([]))
        continue

    # Smooth spectrum (use same window as reference smoothing for consistency)
    window_size = 3
    intensities_array = np.array(intensities)
    smoothed_intensities = np.convolve(intensities_array, np.ones(window_size) / window_size, mode='valid')
    smoothed_wavelengths = np.array(wavelengths[window_size//2 : window_size//2 + len(smoothed_intensities)])

    # Normalize spectrum (avoid division by zero)
    max_abs = np.max(np.abs(smoothed_intensities))
    if max_abs == 0:
        smoothed_intensities_norm = smoothed_intensities
    else:
        smoothed_intensities_norm = smoothed_intensities / max_abs

    # Store for combined plotting
    all_smoothed_wavelengths.append(smoothed_wavelengths)
    all_smoothed_intensities.append(smoothed_intensities_norm)

    # Focus Score 1: Convolution with reference spectrum (alignment/match)
    min_len = min(len(ref_smoothed_intensities), len(smoothed_intensities_norm))
    if min_len > 0:
        conv_score = np.max(np.correlate(smoothed_intensities_norm[:min_len],
                                          ref_smoothed_intensities[:min_len], mode='valid'))
    else:
        conv_score = 0.0
    focus_scores_convolution.append(conv_score)

    # Focus Score 2: Cross-correlation with reference spectrum
    if min_len > 1:
        corr_score = np.corrcoef(smoothed_intensities_norm[:min_len],
                                 ref_smoothed_intensities[:min_len])[0, 1]
        if np.isnan(corr_score):
            corr_score = 0.0
    else:
        corr_score = 0.0
    focus_scores_correlation.append(corr_score)

    # Focus Score 3: Sharpness/contrast (variance of gradient)
    if len(smoothed_intensities_norm) > 1:
        intensity_gradient = np.gradient(smoothed_intensities_norm)
        sharpness = np.var(intensity_gradient)
    else:
        sharpness = 0.0
    focus_scores_sharpness.append(sharpness)

# Plot all smoothed spectra on one axes
fig_all, ax_all = plt.subplots(figsize=(10, 6))
cmap = plt.get_cmap('viridis')
N = len(all_smoothed_intensities)
for i, (wl, inten, pos) in enumerate(zip(all_smoothed_wavelengths, all_smoothed_intensities, positions)):
    if len(wl) == 0:
        continue
    color = cmap(i / max(1, N-1))
    ax_all.plot(wl, inten, color=color, alpha=0.7, linewidth=1)
    # small text label at left of each trace
    ax_all.text(wl[0], inten[0], f'{pos:.2f}', fontsize=8, verticalalignment='bottom', color=color)

ax_all.set_xlabel('Wavelength')
ax_all.set_ylabel('Normalized Intensity')
ax_all.set_title('All Smoothed Spectra (normalized)')
ax_all.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig('all_smoothed_spectra.png', dpi=150, bbox_inches='tight')
plt.show()

# Plot focus scores
fig3, axes3 = plt.subplots(3, 1, figsize=(10, 10))

# Convolution score
axes3[0].plot(positions, focus_scores_convolution, 'ro-', linewidth=2, markersize=8)
axes3[0].set_ylabel('Convolution Score', fontsize=11)
axes3[0].set_title('Focus Score Metrics vs Position', fontsize=12)
axes3[0].grid(True, alpha=0.3)
axes3[0].axvline(x=6.28, color='g', linestyle='--', label='Reference (6.28)', alpha=0.7)
axes3[0].legend()

# Correlation score
axes3[1].plot(positions, focus_scores_correlation, 'go-', linewidth=2, markersize=8)
axes3[1].set_ylabel('Correlation Score', fontsize=11)
axes3[1].grid(True, alpha=0.3)
axes3[1].axvline(x=6.28, color='g', linestyle='--', label='Reference (6.28)', alpha=0.7)
axes3[1].legend()

# Sharpness score
axes3[2].plot(positions, focus_scores_sharpness, 'bo-', linewidth=2, markersize=8)
axes3[2].set_xlabel('Position', fontsize=11)
axes3[2].set_ylabel('Sharpness Score', fontsize=11)
axes3[2].grid(True, alpha=0.3)
axes3[2].axvline(x=6.28, color='g', linestyle='--', label='Reference (6.28)', alpha=0.7)
axes3[2].legend()

plt.tight_layout()
plt.savefig('focus_scores.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n\nFocus Score Analysis:")
print("Position | Convolution | Correlation | Sharpness")
print("-" * 55)
for pos, conv, corr, sharp in zip(positions, focus_scores_convolution, focus_scores_correlation, focus_scores_sharpness):
    print(f"{pos:8.2f} | {conv:11.6f} | {corr:11.6f} | {sharp:9.6f}")

print("\n\nSuggestions for Focus Score Algorithm:")
print("1. Convolution Score: Measures how well the spectrum matches the reference focused spectrum")
print("2. Correlation Score: Pearson correlation with reference - ranges from -1 to 1 (1 = perfect match)")
print("3. Sharpness Score: Measures contrast/edges in the spectrum - higher = sharper/better focused")
print("\nYou could combine these into a single focus metric:")
print("  Combined Score = w1*Correlation + w2*Sharpness + w3*Convolution")
print("  where w1, w2, w3 are weights you can tune based on which metric is most important for your application.")