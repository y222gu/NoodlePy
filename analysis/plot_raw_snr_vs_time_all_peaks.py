"""Plot SNR of raw spectra vs time for all 10 literature blood plasma peaks."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import medfilt
import pandas as pd

# ---- Config ----
LASER_WAVELENGTH_NM = 785.0
N_POINTS = 1024
N_REPS = 300
TOTAL_SPECTRA = 600

data_dir = Path("/Users/yifeigu/Documents/Carney_Lab/RamanPy/data/2026_03_05_1/capture")
spec_files = sorted((data_dir / "spectra").glob("*.txt"))

# Literature peaks
PEAKS = [
    (755,  'Trp / Hb'),
    (850,  'Tyr'),
    (1003, 'Phe'),
    (1079, 'Phospholipids'),
    (1155, 'Carotenoid C-C'),
    (1265, 'Amide III'),
    (1340, 'Trp / a-helix'),
    (1446, 'CH2 scissoring'),
    (1519, 'Carotenoid C=C'),
    (1656, 'Amide I'),
]

# Font sizes
TITLE_SIZE = 28
LABEL_SIZE = 24
TICK_SIZE = 20
LEGEND_SIZE = 16

# ---- Helper ----
def wavelength_to_raman_shift(wavelength_nm, laser_nm=785.0):
    return (1.0 / laser_nm - 1.0 / wavelength_nm) * 1e7

def smooth(arr, w=20):
    return pd.Series(arr).rolling(w, center=True, min_periods=1).mean().values

# ---- Load spectra ----
print("Loading spectra...")
all_raw_list = []
raman_shifts = None

for fpath in spec_files:
    data = np.loadtxt(fpath, delimiter=',')
    wavelengths = data[:N_POINTS, 0]
    if raman_shifts is None:
        raman_shifts = wavelength_to_raman_shift(wavelengths, LASER_WAVELENGTH_NM)
    intensities = data[:, 1].reshape(N_REPS, N_POINTS)
    all_raw_list.append(intensities)

combined_raw = np.vstack(all_raw_list)
print(f"Combined: {combined_raw.shape}")

# ---- Compute noise floor per spectrum (fingerprint region) ----
print("Computing noise floor...")
fp_mask = (raman_shifts >= 700) & (raman_shifts <= 1800)
noise_floor = np.array([
    np.std(combined_raw[i, fp_mask] - medfilt(combined_raw[i, fp_mask], kernel_size=5))
    for i in range(TOTAL_SPECTRA)
])

# ---- Compute SNR per peak per spectrum ----
print("Computing per-peak SNR...")
time_s = np.arange(TOTAL_SPECTRA)
peak_window = 20  # points around peak for local baseline

peak_snr = {}
peak_snr_smooth = {}

for peak_cm, label in PEAKS:
    peak_idx = np.argmin(np.abs(raman_shifts - peak_cm))
    local_region = slice(max(0, peak_idx - peak_window), peak_idx + peak_window)

    snr_arr = np.array([
        (combined_raw[i, peak_idx] - np.median(combined_raw[i, local_region])) / noise_floor[i]
        if noise_floor[i] > 0 else 0
        for i in range(TOTAL_SPECTRA)
    ])
    peak_snr[label] = snr_arr
    peak_snr_smooth[label] = smooth(snr_arr)

# ---- Plot ----
print("Plotting...")
plt.style.use('dark_background')

fig, ax = plt.subplots(figsize=(18, 10), dpi=150)
fig.patch.set_facecolor('black')

# Use same colors as the SNR analysis plot
cmap = plt.cm.tab10
colors_list = [cmap(i) for i in range(10)]

for i, (peak_cm, label) in enumerate(PEAKS):
    color = colors_list[i]
    ax.scatter(time_s, peak_snr[label], s=6, c=[color], alpha=0.35)
    ax.plot(time_s, peak_snr_smooth[label], color=color, linewidth=2.5,
            label=f'{peak_cm} {label}')

ax.set_xlabel('Time (s)', fontsize=LABEL_SIZE)
ax.set_ylabel('SNR', fontsize=LABEL_SIZE)
ax.set_title('SNR of Raw Spectra vs Time for Multiple Peaks', fontsize=TITLE_SIZE)
ax.tick_params(labelsize=TICK_SIZE)
ax.legend(fontsize=LEGEND_SIZE, loc='upper right', ncol=2)

plt.tight_layout()
output_path = data_dir / 'snr_raw_vs_time_all_peaks.png'
plt.savefig(output_path, dpi=150, facecolor='black', bbox_inches='tight')
plt.close()
print(f"Saved to {output_path}")
