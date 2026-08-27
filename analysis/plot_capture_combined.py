"""Combined capture data figure: spectra (left) + background/noise analysis (right)."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from scipy.signal import medfilt, savgol_filter
import pybaselines

# ---- Config ----
LASER_WAVELENGTH_NM = 785.0
N_POINTS = 1024
N_REPS = 300
TOTAL_SPECTRA = 600
CMAP = 'rainbow'

data_dir = Path("/Users/yifeigu/Documents/Carney_Lab/RamanPy/data/2026_03_05_1/capture")
spec_files = sorted((data_dir / "spectra").glob("*.txt"))

# ---- Helper functions ----
def wavelength_to_raman_shift(wavelength_nm, laser_nm=785.0):
    return (1.0 / laser_nm - 1.0 / wavelength_nm) * 1e7

def airpls_baseline(intensity, raman_shifts, lam=100, diff_order=1, max_iter=15, tol=0.005):
    fitter = pybaselines.Baseline(x_data=raman_shifts)
    baseline, _ = fitter.airpls(intensity, lam, diff_order, max_iter, tol)
    return intensity - baseline

def normalize_spectrum(intensity):
    return (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity))

def preprocess_spectrum(intensity, raman_shifts):
    result = intensity.copy()
    crop_mask = (raman_shifts >= 662.697) & (raman_shifts <= 1784.104)
    rs = raman_shifts[crop_mask]
    result = result[crop_mask]
    result = airpls_baseline(result, rs)
    result = savgol_filter(result, 5, 3)
    result = normalize_spectrum(result)
    return result, rs

# ---- Load all spectra ----
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

combined_raw = np.vstack(all_raw_list)  # (600, 1024)
print(f"Combined: {combined_raw.shape}")

# ---- Preprocess all spectra ----
print("Preprocessing...")
processed_list = []
proc_rs = None
for i in range(TOTAL_SPECTRA):
    proc_int, rs = preprocess_spectrum(combined_raw[i], raman_shifts)
    processed_list.append(proc_int)
    if proc_rs is None:
        proc_rs = rs
combined_processed = np.array(processed_list)

# ---- Compute background & noise metrics ----
print("Computing background & noise metrics...")
time_s = np.arange(TOTAL_SPECTRA)  # 1s per spectrum

# Background level: mean in 1900-2400 cm^-1
bg_mask = (raman_shifts >= 1900) & (raman_shifts <= 2400)
background = np.mean(combined_raw[:, bg_mask], axis=1)

# Fingerprint region mean: 700-1800 cm^-1
fp_mask = (raman_shifts >= 700) & (raman_shifts <= 1800)
fp_mean = np.mean(combined_raw[:, fp_mask], axis=1)

# Noise floor: std of residual after median filter in fingerprint region
noise_floor = np.array([
    np.std(combined_raw[i, fp_mask] - medfilt(combined_raw[i, fp_mask], kernel_size=5))
    for i in range(TOTAL_SPECTRA)
])

# SNR: Phe 1003 peak height / noise
phe_idx = np.argmin(np.abs(raman_shifts - 1003))
phe_window = 20
phe_region = slice(max(0, phe_idx - phe_window), phe_idx + phe_window)
snr = np.array([
    (combined_raw[i, phe_idx] - np.median(combined_raw[i, phe_region])) / noise_floor[i]
    if noise_floor[i] > 0 else 0
    for i in range(TOTAL_SPECTRA)
])

# Smoothed versions using pandas rolling mean (no edge artifacts)
import pandas as pd
def smooth(arr, w=20):
    return pd.Series(arr).rolling(w, center=True, min_periods=1).mean().values

bg_smooth = smooth(background)
bg_rate = np.abs(np.gradient(bg_smooth))

# Background excess over stable level
stable_ref = np.mean(bg_smooth[-50:])
bg_excess = (bg_smooth - stable_ref) / stable_ref * 100

bg_s = bg_smooth
fp_s = smooth(fp_mean)
nf_s = smooth(noise_floor)
snr_s = smooth(snr)
rate_s = smooth(bg_rate)

# Processed SNR: Phe 1003 peak height / noise on processed spectra
proc_fp_mask = (proc_rs >= 700) & (proc_rs <= 1800)
proc_phe_idx = np.argmin(np.abs(proc_rs - 1003))
proc_phe_window = 20
proc_phe_region = slice(max(0, proc_phe_idx - proc_phe_window), proc_phe_idx + proc_phe_window)

proc_noise_floor = np.array([
    np.std(combined_processed[i, proc_fp_mask] - medfilt(combined_processed[i, proc_fp_mask], kernel_size=5))
    for i in range(TOTAL_SPECTRA)
])
proc_snr = np.array([
    (combined_processed[i, proc_phe_idx] - np.median(combined_processed[i, proc_phe_region])) / proc_noise_floor[i]
    if proc_noise_floor[i] > 0 else 0
    for i in range(TOTAL_SPECTRA)
])
proc_snr_s = smooth(proc_snr)

# Find cutoff times
cutoff_2pct = next((i for i in range(len(bg_excess)) if bg_excess[i] <= 2.0), TOTAL_SPECTRA)
cutoff_1pct = next((i for i in range(len(bg_excess)) if bg_excess[i] <= 1.0), TOTAL_SPECTRA)

# ---- Font sizes ----
TITLE_SIZE = 28
LABEL_SIZE = 24
TICK_SIZE = 20
LEGEND_SIZE = 19
CBAR_LABEL_SIZE = 22
CBAR_TICK_SIZE = 18
CUTOFF_TEXT_SIZE = 20

# ---- Create combined figure ----
print("Creating combined figure...")
plt.style.use('dark_background')

fig = plt.figure(figsize=(36, 22), dpi=150)
fig.patch.set_facecolor('black')

# Two separate GridSpecs: left for spectra (with colorbar room), right for background/noise
gs_left = GridSpec(3, 1, figure=fig, hspace=0.30,
                   left=0.03, right=0.40, top=0.95, bottom=0.05)
gs_right = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.25,
                    left=0.43, right=0.98, top=0.95, bottom=0.05)

# -- Left column: spectra --
ax_raw = fig.add_subplot(gs_left[0:2, 0])   # Raw spectra (spans 2 rows)
ax_proc = fig.add_subplot(gs_left[2, 0])     # Processed spectra

# Combined raw spectra
colors = plt.cm.rainbow(np.linspace(0, 1, TOTAL_SPECTRA))
for i in range(TOTAL_SPECTRA):
    ax_raw.plot(raman_shifts, combined_raw[i], color=colors[i], linewidth=0.2, alpha=0.5)
ax_raw.set_xlabel('Raman Shift (cm$^{-1}$)', fontsize=LABEL_SIZE)
ax_raw.set_ylabel('Intensity (counts)', fontsize=LABEL_SIZE)
ax_raw.set_title('600 Consecutive Raw Spectra at a Same XYZ', fontsize=TITLE_SIZE)
ax_raw.tick_params(labelsize=TICK_SIZE)

# Colorbar for raw
sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(vmin=1, vmax=TOTAL_SPECTRA))
sm.set_array([])
cbar1 = fig.colorbar(sm, ax=ax_raw, shrink=0.8, pad=0.01)
cbar1.set_label('Sequence in time (s)', fontsize=CBAR_LABEL_SIZE)
cbar1.ax.tick_params(labelsize=CBAR_TICK_SIZE)

# Combined processed spectra
for i in range(TOTAL_SPECTRA):
    ax_proc.plot(proc_rs, combined_processed[i], color=colors[i], linewidth=0.2, alpha=0.5)
ax_proc.set_xlabel('Raman Shift (cm$^{-1}$)', fontsize=LABEL_SIZE)
ax_proc.set_ylabel('Intensity (a.u.)', fontsize=LABEL_SIZE)
ax_proc.set_title('600 Consecutive Processed Spectra at a Same XYZ', fontsize=TITLE_SIZE)
ax_proc.set_ylim(-0.15, 1.1)
ax_proc.tick_params(labelsize=TICK_SIZE)

# Colorbar for processed
cbar2 = fig.colorbar(sm, ax=ax_proc, shrink=0.8, pad=0.01)
cbar2.set_label('Sequence in time (s)', fontsize=CBAR_LABEL_SIZE)
cbar2.ax.tick_params(labelsize=CBAR_TICK_SIZE)

# -- Right column: background & noise (3 rows x 2 cols) --
DOT_COLOR = 'cyan'
DOT_SIZE = 5
DOT_ALPHA = 0.3
LINE_COLOR = 'yellow'
LINE_WIDTH = 2

ax_bg = fig.add_subplot(gs_right[0, 0])
ax_fp = fig.add_subplot(gs_right[0, 1])
ax_nf = fig.add_subplot(gs_right[1, 0])
ax_snr = fig.add_subplot(gs_right[1, 1])
ax_rate = fig.add_subplot(gs_right[2, 0])
ax_excess = fig.add_subplot(gs_right[2, 1])

# Background level
ax_bg.scatter(time_s, background, s=DOT_SIZE, c=DOT_COLOR, alpha=DOT_ALPHA)
ax_bg.plot(time_s, bg_s, color=LINE_COLOR, linewidth=LINE_WIDTH, label='20-pt avg')
ax_bg.set_xlabel('Time (s)', fontsize=LABEL_SIZE)
ax_bg.set_ylabel('Mean Intensity (counts)', fontsize=LABEL_SIZE)
ax_bg.set_title('Background Level (1900-2400 cm$^{-1}$)', fontsize=TITLE_SIZE)
ax_bg.legend(fontsize=LEGEND_SIZE)
ax_bg.tick_params(labelsize=TICK_SIZE)

# Background rate of change (swapped from bottom-left)
ax_fp.scatter(time_s, bg_rate, s=DOT_SIZE, c=DOT_COLOR, alpha=DOT_ALPHA)
ax_fp.plot(time_s, rate_s, color=LINE_COLOR, linewidth=LINE_WIDTH, label='20-pt avg')
ax_fp.set_xlabel('Time (s)', fontsize=LABEL_SIZE)
ax_fp.set_ylabel('|d(bg)/dt| (counts/s)', fontsize=LABEL_SIZE)
ax_fp.set_title('Background Rate of Change', fontsize=TITLE_SIZE)
ax_fp.legend(fontsize=LEGEND_SIZE)
ax_fp.tick_params(labelsize=TICK_SIZE)

# Noise floor
ax_nf.scatter(time_s, noise_floor, s=DOT_SIZE, c=DOT_COLOR, alpha=DOT_ALPHA)
ax_nf.plot(time_s, nf_s, color=LINE_COLOR, linewidth=LINE_WIDTH, label='20-pt avg')
ax_nf.set_xlabel('Time (s)', fontsize=LABEL_SIZE)
ax_nf.set_ylabel('Std Dev (counts)', fontsize=LABEL_SIZE)
ax_nf.set_title('Noise Floor of Raw Spectra', fontsize=TITLE_SIZE)
ax_nf.legend(fontsize=LEGEND_SIZE)
ax_nf.tick_params(labelsize=TICK_SIZE)

# SNR
ax_snr.scatter(time_s, snr, s=DOT_SIZE, c=DOT_COLOR, alpha=DOT_ALPHA)
ax_snr.plot(time_s, snr_s, color=LINE_COLOR, linewidth=LINE_WIDTH, label='20-pt avg')
ax_snr.set_xlabel('Time (s)', fontsize=LABEL_SIZE)
ax_snr.set_ylabel('SNR', fontsize=LABEL_SIZE)
ax_snr.set_title('SNR of Raw Spectra (~1003 cm$^{-1}$)', fontsize=TITLE_SIZE)
ax_snr.legend(fontsize=LEGEND_SIZE)
ax_snr.tick_params(labelsize=TICK_SIZE)

# Background excess over stable level
ax_rate.plot(time_s, bg_excess, color=LINE_COLOR, linewidth=LINE_WIDTH)
ax_rate.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, label='1% excess threshold')
ax_rate.axhline(y=2.0, color='red', linestyle='-.', linewidth=1.5, label='2% excess threshold')
ax_rate.axvline(x=cutoff_2pct, color='green', linestyle=':', linewidth=2)
ax_rate.axvline(x=cutoff_1pct, color='red', linestyle=':', linewidth=2)
ax_rate.text(cutoff_2pct - 5, 3.5, f'{cutoff_2pct}s', color='green', fontsize=CUTOFF_TEXT_SIZE, ha='right')
ax_rate.text(cutoff_1pct + 5, 3.5, f'{cutoff_1pct}s', color='red', fontsize=CUTOFF_TEXT_SIZE, ha='left')
ax_rate.set_xlabel('Time (s)', fontsize=LABEL_SIZE)
ax_rate.set_ylabel('Excess Background (%)', fontsize=LABEL_SIZE)
ax_rate.set_title('Background Excess Over Stable Level', fontsize=TITLE_SIZE)
ax_rate.legend(fontsize=LEGEND_SIZE - 2)
ax_rate.tick_params(labelsize=TICK_SIZE)
ax_rate.set_ylim(-0.5, 6)

# Processed SNR
ax_excess.scatter(time_s, proc_snr, s=DOT_SIZE, c=DOT_COLOR, alpha=DOT_ALPHA)
ax_excess.plot(time_s, proc_snr_s, color=LINE_COLOR, linewidth=LINE_WIDTH, label='20-pt avg')
ax_excess.set_xlabel('Time (s)', fontsize=LABEL_SIZE)
ax_excess.set_ylabel('SNR', fontsize=LABEL_SIZE)
ax_excess.set_title('SNR of Processed Spectra (~1003 cm$^{-1}$)', fontsize=TITLE_SIZE)
ax_excess.legend(fontsize=LEGEND_SIZE)
ax_excess.tick_params(labelsize=TICK_SIZE)

# Save
output_path = data_dir / 'capture_analysis_combined.png'
plt.savefig(output_path, dpi=150, facecolor='black', bbox_inches='tight')
plt.close()
print(f"Saved to {output_path}")
