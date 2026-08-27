"""Plot preprocessed spectra and hyperspectral map colored by K-means cluster."""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import medfilt, savgol_filter
from scipy.cluster.vq import kmeans2, whiten
from scipy.spatial.distance import cdist
import pybaselines

# Paths - Dataset 1
json_path1 = Path("/Users/yifeigu/Documents/Carney_Lab/RamanPy/data/spectra to plot.json")
experiment_path1 = Path("/Users/yifeigu/Documents/Carney_Lab/RamanPy/data/2026_01_29_3")

# Paths - Dataset 2
json_path2 = Path("/Users/yifeigu/Documents/Carney_Lab/RamanPy/data/spectra to plot.json")
experiment_path2 = Path("/Users/yifeigu/Documents/Carney_Lab/RamanPy/data/2026_01_29_3")
 # Will use experiment_folder from metadata

# Output paths
output_spectra_path = Path("/Users/yifeigu/Documents/Carney_Lab/RamanPy/data/clustered_spectra_plot.png")
output_map_dir = Path("/Users/yifeigu/Documents/Carney_Lab/RamanPy/data/clustered_maps")
output_map_dir.mkdir(parents=True, exist_ok=True)

# Laser wavelength for Raman shift calculation
LASER_WAVELENGTH_NM = 785.0

# K-means clustering parameters
OPTIMIZE_CLUSTERS = True  # If True, find optimal number of clusters automatically
MIN_CLUSTERS = 2          # Minimum number of clusters to try
MAX_CLUSTERS = 10         # Maximum number of clusters to try
N_CLUSTERS = 5            # Used only if OPTIMIZE_CLUSTERS = False
RANDOM_STATE = 42
SAVE_OPTIMIZATION_PLOT = True  # Save elbow/silhouette plot

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

def silhouette_score(data, labels):
    """Calculate silhouette score for clustering quality."""
    n_samples = len(labels)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    if n_clusters <= 1 or n_clusters >= n_samples:
        return -1

    # Calculate pairwise distances
    distances = cdist(data, data, metric='euclidean')

    silhouette_vals = []
    for i in range(n_samples):
        # a(i) = average distance to points in same cluster
        same_cluster = labels == labels[i]
        same_cluster[i] = False  # Exclude self
        if np.sum(same_cluster) > 0:
            a_i = np.mean(distances[i, same_cluster])
        else:
            a_i = 0

        # b(i) = minimum average distance to points in other clusters
        b_i = np.inf
        for label in unique_labels:
            if label != labels[i]:
                other_cluster = labels == label
                if np.sum(other_cluster) > 0:
                    avg_dist = np.mean(distances[i, other_cluster])
                    b_i = min(b_i, avg_dist)

        if b_i == np.inf:
            b_i = 0

        # Silhouette coefficient
        s_i = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0
        silhouette_vals.append(s_i)

    return np.mean(silhouette_vals)

def calculate_inertia(data, centroids, labels):
    """Calculate within-cluster sum of squares (inertia)."""
    inertia = 0
    for i, centroid in enumerate(centroids):
        cluster_points = data[labels == i]
        if len(cluster_points) > 0:
            inertia += np.sum((cluster_points - centroid) ** 2)
    return inertia

def find_optimal_clusters(data, min_k, max_k, random_state=42):
    """Find optimal number of clusters using silhouette score."""
    np.random.seed(random_state)

    results = []
    print(f"Testing {min_k} to {max_k} clusters...")

    for k in range(min_k, max_k + 1):
        try:
            centroids, labels = kmeans2(data, k, iter=20, minit='++')
            labels = labels.astype(int)

            # Calculate metrics
            inertia = calculate_inertia(data, centroids, labels)
            sil_score = silhouette_score(data, labels)

            results.append({
                'k': k,
                'inertia': inertia,
                'silhouette': sil_score,
                'centroids': centroids,
                'labels': labels
            })
            print(f"  k={k}: silhouette={sil_score:.4f}, inertia={inertia:.2f}")
        except Exception as e:
            print(f"  k={k}: failed ({e})")

    # Find optimal k based on silhouette score
    best_result = max(results, key=lambda x: x['silhouette'])
    optimal_k = best_result['k']

    print(f"\nOptimal number of clusters: {optimal_k} (silhouette={best_result['silhouette']:.4f})")

    return optimal_k, results, best_result

# Load metadata from both datasets
print("Loading metadata...")

with open(json_path1, 'r') as f:
    metadata_list1 = json.load(f)
print(f"Dataset 1: Found {len(metadata_list1)} spectra")

with open(json_path2, 'r') as f:
    metadata_list2 = json.load(f)
print(f"Dataset 2: Found {len(metadata_list2)} spectra")

if PREPROCESSING_ENABLED:
    print(f"Preprocessing: cropping={CROPPING}, cosmic_rays={REMOVE_COSMIC_RAYS}, "
          f"baseline={BASELINE_CORRECTION}, smoothing={SMOOTHING}, normalization={NORMALIZATION}")

# Load spectra and positions from both datasets
# Dataset 1
x_positions1 = []
y_positions1 = []
sample_ids1 = []  # Track sample/droplet for each spectrum
sample_info1 = {}  # Store sample metadata
spectra1 = []
all_raman_shifts = None

print("\nLoading Dataset 1 spectra...")
spectra_folder1 = experiment_path1 / "spectra"
for i, meta in enumerate(metadata_list1):
    spectrum_file = spectra_folder1 / meta['spectrum_file']

    if not spectrum_file.exists():
        continue

    x_positions1.append(meta['x'])
    y_positions1.append(meta['y'])

    # Get sample identifier (use droplet_id if available, else sample number)
    sample_id = meta.get('droplet_id', meta.get('sample', 0))
    sample_ids1.append(sample_id)

    # Store sample info
    if sample_id not in sample_info1:
        sample_info1[sample_id] = {
            'patient': meta.get('patient', 'unknown'),
            'sample': meta.get('sample', sample_id),
            'staging': meta.get('staging', 'unknown')
        }

    data = np.loadtxt(spectrum_file, delimiter=',')
    n_points = 1024
    n_reps = len(data) // n_points

    wavelengths_nm = data[:n_points, 0]
    raman_shifts = wavelength_to_raman_shift(wavelengths_nm, LASER_WAVELENGTH_NM)

    all_intensities = data[:, 1].reshape(n_reps, n_points)
    intensities = np.median(all_intensities, axis=0)

    intensities, raman_shifts = preprocess_spectrum(intensities, raman_shifts)

    if all_raman_shifts is None:
        all_raman_shifts = raman_shifts

    spectra1.append(intensities)

    if (i + 1) % 200 == 0:
        print(f"  Loaded {i + 1}/{len(metadata_list1)} spectra...")

print(f"  Loaded {len(spectra1)} spectra from Dataset 1 ({len(sample_info1)} samples)")

# Dataset 2
x_positions2 = []
y_positions2 = []
sample_ids2 = []  # Track sample/droplet for each spectrum
sample_info2 = {}  # Store sample metadata
spectra2 = []

print("\nLoading Dataset 2 spectra...")
for i, meta in enumerate(metadata_list2):
    # Get experiment folder from metadata
    experiment_folder = meta.get('experiment_folder', meta.get('date', ''))
    spectra_folder2 = experiment_path2 / experiment_folder / "spectra"

    spectrum_file = spectra_folder2 / meta['spectrum_file']

    if not spectrum_file.exists():
        continue

    x_positions2.append(meta['x'])
    y_positions2.append(meta['y'])

    # Get sample identifier (use droplet_id if available, else sample number)
    sample_id = meta.get('droplet_id', meta.get('sample', 0))
    sample_ids2.append(sample_id)

    # Store sample info
    if sample_id not in sample_info2:
        sample_info2[sample_id] = {
            'patient': meta.get('patient', 'unknown'),
            'sample': meta.get('sample', sample_id),
            'staging': meta.get('staging', 'unknown')
        }

    data = np.loadtxt(spectrum_file, delimiter=',')
    n_points = 1024
    n_reps = len(data) // n_points

    wavelengths_nm = data[:n_points, 0]
    raman_shifts = wavelength_to_raman_shift(wavelengths_nm, LASER_WAVELENGTH_NM)

    all_intensities = data[:, 1].reshape(n_reps, n_points)
    intensities = np.median(all_intensities, axis=0)

    intensities, raman_shifts = preprocess_spectrum(intensities, raman_shifts)

    spectra2.append(intensities)

    if (i + 1) % 200 == 0:
        print(f"  Loaded {i + 1}/{len(metadata_list2)} spectra...")

print(f"  Loaded {len(spectra2)} spectra from Dataset 2 ({len(sample_info2)} samples)")

# Convert to arrays
x_positions1 = np.array(x_positions1)
y_positions1 = np.array(y_positions1)
sample_ids1 = np.array(sample_ids1)
spectra1 = np.array(spectra1)

x_positions2 = np.array(x_positions2)
y_positions2 = np.array(y_positions2)
sample_ids2 = np.array(sample_ids2)
spectra2 = np.array(spectra2)

# Combine all spectra for clustering
all_spectra = np.vstack([spectra1, spectra2])
dataset_labels = np.array([1] * len(spectra1) + [2] * len(spectra2))  # Track which dataset

print(f"\nCombined spectra shape: {all_spectra.shape}")
print(f"  Dataset 1: {len(spectra1)} spectra, {len(sample_info1)} samples")
print(f"  Dataset 2: {len(spectra2)} spectra, {len(sample_info2)} samples")
print(f"Raman shift range: {all_raman_shifts.min():.1f} - {all_raman_shifts.max():.1f} cm^-1")

# Whiten data for better clustering
whitened = whiten(all_spectra)

# Run K-means clustering
if OPTIMIZE_CLUSTERS:
    print(f"\nOptimizing number of clusters ({MIN_CLUSTERS}-{MAX_CLUSTERS})...")
    optimal_k, optimization_results, best_result = find_optimal_clusters(
        whitened, MIN_CLUSTERS, MAX_CLUSTERS, RANDOM_STATE
    )
    n_clusters = optimal_k
    cluster_labels = best_result['labels']
    centroids = best_result['centroids']

    # Save optimization plot
    if SAVE_OPTIMIZATION_PLOT:
        output_opt_path = Path("/Users/yifeigu/Documents/Carney_Lab/RamanPy/data/cluster_optimization.png")
        print(f"\nSaving optimization plot to {output_opt_path}...")

        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
        fig.patch.set_facecolor('black')

        ks = [r['k'] for r in optimization_results]
        inertias = [r['inertia'] for r in optimization_results]
        silhouettes = [r['silhouette'] for r in optimization_results]

        # Elbow plot
        ax1.plot(ks, inertias, 'o-', color='cyan', linewidth=2, markersize=8)
        ax1.axvline(x=optimal_k, color='red', linestyle='--', linewidth=2, label=f'Optimal k={optimal_k}')
        ax1.set_xlabel('Number of Clusters (k)', color='white', fontsize=14)
        ax1.set_ylabel('Inertia (WCSS)', color='white', fontsize=14)
        ax1.set_title('Elbow Method', color='white', fontsize=16)
        ax1.tick_params(axis='both', colors='white', labelsize=12)
        ax1.legend(facecolor='black', edgecolor='white', labelcolor='white')
        for spine in ax1.spines.values():
            spine.set_color('white')
        ax1.set_facecolor('black')

        # Silhouette plot
        ax2.plot(ks, silhouettes, 'o-', color='lime', linewidth=2, markersize=8)
        ax2.axvline(x=optimal_k, color='red', linestyle='--', linewidth=2, label=f'Optimal k={optimal_k}')
        ax2.set_xlabel('Number of Clusters (k)', color='white', fontsize=14)
        ax2.set_ylabel('Silhouette Score', color='white', fontsize=14)
        ax2.set_title('Silhouette Score (higher is better)', color='white', fontsize=16)
        ax2.tick_params(axis='both', colors='white', labelsize=12)
        ax2.legend(facecolor='black', edgecolor='white', labelcolor='white')
        for spine in ax2.spines.values():
            spine.set_color('white')
        ax2.set_facecolor('black')

        plt.tight_layout()
        plt.savefig(output_opt_path, dpi=300, facecolor='black', edgecolor='none', bbox_inches='tight')
        plt.close()
else:
    print(f"\nRunning K-means clustering with {N_CLUSTERS} clusters...")
    np.random.seed(RANDOM_STATE)
    n_clusters = N_CLUSTERS
    centroids, cluster_labels = kmeans2(whitened, n_clusters, iter=20, minit='++')
    cluster_labels = cluster_labels.astype(int)

# Print cluster distribution
unique, counts = np.unique(cluster_labels, return_counts=True)
print("\nCluster distribution:")
for cluster, count in zip(unique, counts):
    print(f"  Cluster {cluster}: {count} spectra ({100*count/len(cluster_labels):.1f}%)")

# Define colors for clusters
cluster_colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

# ============ PLOT 1: Spectra colored by cluster ============
print("\nPlotting clustered spectra...")
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

fig.patch.set_facecolor('black')
ax.set_facecolor('black')

# Plot each spectrum colored by cluster
for cluster in range(n_clusters):
    cluster_mask = cluster_labels == cluster
    cluster_spectra = all_spectra[cluster_mask]
    color = cluster_colors[cluster]

    for spectrum in cluster_spectra:
        ax.plot(all_raman_shifts, spectrum, color=color, linewidth=0.3, alpha=0.5)

# Set y-axis limits based on normalization
if NORMALIZATION and PREPROCESSING_ENABLED:
    ax.set_ylim(-0.15, 1.1)

# Style the plot
ax.set_xlabel('Raman Shift (cm$^{-1}$)', color='white', fontsize=18)
ax.set_ylabel('Intensity (a.u.)', color='white', fontsize=18)
ax.set_title(f'Preprocessed Spectra by K-means Cluster (k={n_clusters}, n={len(all_spectra)})',
             color='white', fontsize=20)

# White spines and ticks
for spine in ax.spines.values():
    spine.set_color('white')
ax.tick_params(axis='both', colors='white', labelsize=14)

# Add legend
legend_handles = [plt.Line2D([0], [0], color=cluster_colors[i], linewidth=2,
                              label=f'Cluster {i} (n={counts[i]})')
                  for i in range(n_clusters)]
ax.legend(handles=legend_handles, loc='upper right', fontsize=12,
          facecolor='black', edgecolor='white', labelcolor='white')

plt.tight_layout()

print(f"Saving spectra plot to {output_spectra_path}...")
plt.savefig(output_spectra_path, dpi=300, facecolor='black', edgecolor='none', bbox_inches='tight')
plt.close()

# Split cluster labels by dataset
cluster_labels1 = cluster_labels[:len(spectra1)]
cluster_labels2 = cluster_labels[len(spectra1):]

# ============ PLOT 2: Hyperspectral maps for Dataset 1 (one per sample) ============
print("\nPlotting clustered hyperspectral maps for Dataset 1...")

for sample_id in sorted(sample_info1.keys()):
    info = sample_info1[sample_id]
    sample_mask = sample_ids1 == sample_id

    if not np.any(sample_mask):
        continue

    x_sample = x_positions1[sample_mask]
    y_sample = y_positions1[sample_mask]
    labels_sample = cluster_labels1[sample_mask]

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # Count clusters in this sample
    unique_s, counts_s = np.unique(labels_sample, return_counts=True)
    counts_s_dict = dict(zip(unique_s, counts_s))

    # Plot each cluster with its color
    for cluster in range(n_clusters):
        cluster_mask = labels_sample == cluster
        if np.any(cluster_mask):
            count = counts_s_dict.get(cluster, 0)
            ax.scatter(
                x_sample[cluster_mask],
                y_sample[cluster_mask],
                c=[cluster_colors[cluster]],
                s=100,
                alpha=1,
                label=f'Cluster {cluster} (n={count})'
            )

    # Style the plot
    ax.set_xlabel('X Position (μm)', color='white', fontsize=18)
    ax.set_ylabel('Y Position (μm)', color='white', fontsize=18)
    title = f"Dataset 1 - Patient {info['patient']}, Sample {info['sample']}\n({info['staging']}, n={len(x_sample)})"
    ax.set_title(title, color='white', fontsize=18)

    ax.set_aspect('equal')
    ax.tick_params(axis='both', colors='white', labelsize=14)

    for spine in ax.spines.values():
        spine.set_color('white')

    ax.legend(loc='upper right', fontsize=10, facecolor='black', edgecolor='white', labelcolor='white')

    plt.tight_layout()

    output_path = output_map_dir / f"dataset1_sample_{sample_id}_patient_{info['patient']}.png"
    print(f"  Saving {output_path.name}...")
    plt.savefig(output_path, dpi=300, facecolor='black', edgecolor='none', bbox_inches='tight')
    plt.close()

print(f"  Saved {len(sample_info1)} maps for Dataset 1")

# ============ PLOT 3: Hyperspectral maps for Dataset 2 (one per sample) ============
print("\nPlotting clustered hyperspectral maps for Dataset 2...")

for sample_id in sorted(sample_info2.keys()):
    info = sample_info2[sample_id]
    sample_mask = sample_ids2 == sample_id

    if not np.any(sample_mask):
        continue

    x_sample = x_positions2[sample_mask]
    y_sample = y_positions2[sample_mask]
    labels_sample = cluster_labels2[sample_mask]

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # Count clusters in this sample
    unique_s, counts_s = np.unique(labels_sample, return_counts=True)
    counts_s_dict = dict(zip(unique_s, counts_s))

    # Plot each cluster with its color
    for cluster in range(n_clusters):
        cluster_mask = labels_sample == cluster
        if np.any(cluster_mask):
            count = counts_s_dict.get(cluster, 0)
            ax.scatter(
                x_sample[cluster_mask],
                y_sample[cluster_mask],
                c=[cluster_colors[cluster]],
                s=100,
                alpha=1,
                label=f'Cluster {cluster} (n={count})'
            )

    # Style the plot
    ax.set_xlabel('X Position (μm)', color='white', fontsize=18)
    ax.set_ylabel('Y Position (μm)', color='white', fontsize=18)
    title = f"Dataset 2 - Patient {info['patient']}, Sample {info['sample']}\n({info['staging']}, n={len(x_sample)})"
    ax.set_title(title, color='white', fontsize=18)

    ax.set_aspect('equal')
    ax.tick_params(axis='both', colors='white', labelsize=14)

    for spine in ax.spines.values():
        spine.set_color('white')

    ax.legend(loc='upper right', fontsize=10, facecolor='black', edgecolor='white', labelcolor='white')

    plt.tight_layout()

    output_path = output_map_dir / f"dataset2_sample_{sample_id}_patient_{info['patient']}.png"
    print(f"  Saving {output_path.name}...")
    plt.savefig(output_path, dpi=300, facecolor='black', edgecolor='none', bbox_inches='tight')
    plt.close()

print(f"  Saved {len(sample_info2)} maps for Dataset 2")

print("\nDone!")
print(f"All maps saved to: {output_map_dir}")
