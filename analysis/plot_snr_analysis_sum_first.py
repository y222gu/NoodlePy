"""SNR Analysis — Average/Sum Raw, Then Process. Last 300 spectra, both directions."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import savgol_filter, medfilt
import pybaselines
from matplotlib.patches import Patch

LASER_WAVELENGTH_NM = 785.0
N_POINTS = 1024
N_REPS = 300

data_dir = Path('/Users/yifeigu/Documents/Carney_Lab/RamanPy/data/2026_03_05_1/capture')

def wavelength_to_raman_shift(wl, laser=785.0):
    return (1.0/laser - 1.0/wl) * 1e7

def airpls_baseline(intensity, rs, lam=100, diff_order=1, max_iter=15, tol=0.005):
    fitter = pybaselines.Baseline(x_data=rs)
    bl, _ = fitter.airpls(intensity, lam, diff_order, max_iter, tol)
    return intensity - bl

def process_spectrum(intensity, rs, normalize=True):
    crop_mask = (rs >= 662.697) & (rs <= 1784.104)
    rs_c = rs[crop_mask]
    s = intensity[crop_mask]
    s = airpls_baseline(s, rs_c)
    s = savgol_filter(s, 5, 3)
    if normalize:
        s = (s - s.min()) / (s.max() - s.min()) if s.max() != s.min() else s
    return s, rs_c

def compute_raw_noise(raw_spectrum, rs, fp_lo=700, fp_hi=1800):
    """Noise floor from raw spectrum (before processing) in fingerprint region."""
    fp_mask = (rs >= fp_lo) & (rs <= fp_hi)
    raw_fp = raw_spectrum[fp_mask]
    med = medfilt(raw_fp, kernel_size=5)
    return np.std(raw_fp - med)

def compute_peak_snr(proc_spectrum, proc_rs, raw_noise, peak_cm, window=15):
    """SNR = processed peak height / raw noise floor."""
    mask = (proc_rs >= peak_cm - window) & (proc_rs <= peak_cm + window)
    if not np.any(mask): return 0
    peak_val = proc_spectrum[mask].max()
    local_bl = np.mean([proc_spectrum[mask][0], proc_spectrum[mask][-1]])
    peak_height = peak_val - local_bl
    return peak_height / raw_noise if raw_noise > 0 else 0

# Load, skip #520
print("Loading spectra...")
d3 = np.loadtxt(data_dir / 'spectra/Captured_spectra_3.txt', delimiter=',')
d4 = np.loadtxt(data_dir / 'spectra/Captured_spectra_4.txt', delimiter=',')
wl = d3[:N_POINTS, 0]
rs_full = wavelength_to_raman_shift(wl)
raw_all = np.vstack([d3[:, 1].reshape(N_REPS, N_POINTS), d4[:, 1].reshape(N_REPS, N_POINTS)])
skip_mask = np.ones(600, dtype=bool)
skip_mask[519] = False
raw_clean = raw_all[skip_mask]  # 599
N_total = 599

# Use last 300 spectra
raw_last300 = raw_clean[-300:]
N_pool = 300
print(f'Using {N_total} spectra (skipped #520), last {N_pool} for analysis')

peaks = {
    755:  ('Trp / Hb',        'Proteins; haemoglobin'),
    850:  ('Tyr',              'Tyrosine ring breathing'),
    1003: ('Phe',              'Phenylalanine ring breathing'),
    1079: ('Phospholipids',    'O-P-O & C-C stretch'),
    1155: ('Carotenoid C-C',   'Carotenoid C-C stretch'),
    1265: ('Amide III',        'Protein backbone N-H + C-N'),
    1340: ('Trp / a-helix',   'Tryptophan; phospholipids'),
    1446: ('CH2 scissoring',   'Phospholipids, proteins'),
    1519: ('Carotenoid C=C',   'Carotenoid C=C stretch'),
    1656: ('Amide I',          'Protein a-helix; phospholipids'),
}

spec_avg_counts = [1, 2, 5, 10, 20, 50, 100, 200, 300]

# ---- Compute SNR for both methods (avg and sum), both directions ----
def compute_snr_front(raw_pool, rs, method, avg_list):
    """Compute SNR distributions from front only."""
    combine_fn = np.mean if method == 'mean' else np.sum
    n_total = len(raw_pool)
    dist = {n: {pc: [] for pc in peaks} for n in avg_list}
    for n_avg in avg_list:
        n_groups = n_total // n_avg
        for g in range(n_groups):
            s_f = g * n_avg
            e_f = (g + 1) * n_avg
            combined = combine_fn(raw_pool[s_f:e_f], axis=0)
            noise = compute_raw_noise(combined, rs)
            proc, rs_c = process_spectrum(combined, rs, normalize=False)
            for pc in peaks:
                dist[n_avg][pc].append(compute_peak_snr(proc, rs_c, noise, pc))
        print(f'  {method} n={n_avg}: {n_groups} groups done')
    return dist

def compute_snr_distributions(raw_pool, rs, method='mean'):
    """Compute SNR distributions for all averaging levels, both directions."""
    combine_fn = np.mean if method == 'mean' else np.sum
    n_total = len(raw_pool)
    dist_end = {n: {pc: [] for pc in peaks} for n in spec_avg_counts}
    dist_front = {n: {pc: [] for pc in peaks} for n in spec_avg_counts}

    for n_avg in spec_avg_counts:
        n_groups = n_total // n_avg
        for g in range(n_groups):
            # From end
            s_e = n_total - (g + 1) * n_avg
            e_e = n_total - g * n_avg
            combined_e = combine_fn(raw_pool[s_e:e_e], axis=0)
            noise_e = compute_raw_noise(combined_e, rs)
            proc_e, rs_c = process_spectrum(combined_e, rs, normalize=False)
            # From front
            s_f = g * n_avg
            e_f = (g + 1) * n_avg
            combined_f = combine_fn(raw_pool[s_f:e_f], axis=0)
            noise_f = compute_raw_noise(combined_f, rs)
            proc_f, _ = process_spectrum(combined_f, rs, normalize=False)
            for pc in peaks:
                dist_end[n_avg][pc].append(compute_peak_snr(proc_e, rs_c, noise_e, pc))
                dist_front[n_avg][pc].append(compute_peak_snr(proc_f, rs_c, noise_f, pc))
        print(f'  {method} n={n_avg}: {n_groups} groups done')
    return dist_end, dist_front

print("Computing SNR distributions (average)...")
avg_dist_end, avg_dist_front = compute_snr_distributions(raw_last300, rs_full, 'mean')
print("Computing SNR distributions (sum)...")
sum_dist_end, sum_dist_front = compute_snr_distributions(raw_last300, rs_full, 'sum')

def get_median_snr(dist, avg_list=None):
    if avg_list is None:
        avg_list = spec_avg_counts
    median = {pc: [] for pc in peaks}
    for n_avg in avg_list:
        for pc in peaks:
            vals = dist[n_avg][pc]
            median[pc].append(np.median(vals) if len(vals) > 0 else 0)
    return median

# Spectra for bottom plot — raw for avg, processed for sum
def make_raw_spectra(raw_pool):
    end_d, front_d = {}, {}
    for n in spec_avg_counts:
        end_d[n] = raw_pool[-n:].mean(axis=0)
        front_d[n] = raw_pool[:n].mean(axis=0)
    return end_d, front_d

def make_proc_spectra(raw_pool, rs):
    end_d, front_d = {}, {}
    rs_c = None
    for n in spec_avg_counts:
        summed_e = raw_pool[-n:].sum(axis=0)
        proc_e, rs_c = process_spectrum(summed_e, rs)
        end_d[n] = proc_e
        summed_f = raw_pool[:n].sum(axis=0)
        proc_f, _ = process_spectrum(summed_f, rs)
        front_d[n] = proc_f
    return end_d, front_d, rs_c

raw_avg_end, raw_avg_front = make_raw_spectra(raw_last300)
proc_sum_end, proc_sum_front, rs_crop = make_proc_spectra(raw_last300, rs_full)

# ========== PLOT ==========
TICK_SIZE = 22
LABEL_SIZE = 26
TITLE_SIZE = 28
SUPTITLE_SIZE = 30
LEGEND_SIZE = 18
ANNOT_SIZE = 15

n_peaks = len(peaks)
peak_list = list(peaks.keys())
x_base = np.arange(n_peaks)
show_avgs = [1, 10, 50, 200]
display_labels = {1:'1s', 2:'2s', 5:'5s', 10:'10s', 20:'20s', 50:'50s', 100:'100s', 200:'200s', 300:'300s'}
time_s = np.array(spec_avg_counts)
cmap_lines = [plt.cm.tab10(i) for i in range(n_peaks)]
n_levels = len(spec_avg_counts)
cmap_spec = plt.cm.rainbow(np.linspace(0.0, 1.0, n_levels))

# Build color map: spec_avg_counts index -> color, so boxplot uses same rainbow colors
spec_color_map = {n: cmap_spec[i] for i, n in enumerate(spec_avg_counts)}
colors_box = [spec_color_map[n] for n in show_avgs]

plt.style.use('dark_background')

# All 4 plots: (method, direction, snr_dist, spec_dict, suffix, is_processed)
plot_configs = [
    ('Average', 'From End',   avg_dist_end,   raw_avg_end,    'avg_first_from_end',   False),
    ('Average', 'From Front', avg_dist_front, raw_avg_front,  'avg_first_from_front', False),
    ('Sum',     'From End',   sum_dist_end,   proc_sum_end,   'sum_first_from_end',   True),
    ('Sum',     'From Front', sum_dist_front, proc_sum_front, 'sum_first_from_front', True),
]

for method, direction, snr_dist, spec_dict, suffix, is_processed in plot_configs:
    median_snr = get_median_snr(snr_dist)
    method_label = method
    print(f"Plotting {method} {direction}...")
    fig = plt.figure(figsize=(30, 24), dpi=150)
    fig.patch.set_facecolor('black')
    fig.suptitle(f'SNR Analysis: Last 300 Spectra, {method} Raw ({direction}) Then Process',
                 fontsize=SUPTITLE_SIZE, y=1.01)

    # --- TOP-LEFT: Boxplot ---
    ax1 = fig.add_subplot(2, 2, 1)
    width = 0.18
    for j, (n_avg, c) in enumerate(zip(show_avgs, colors_box)):
        data_bp = [snr_dist[n_avg][pc] for pc in peak_list]
        positions = x_base + (j - 1.5) * width
        ax1.boxplot(data_bp, positions=positions, widths=width*0.85, patch_artist=True,
                    boxprops=dict(facecolor=c, alpha=0.4, edgecolor=c),
                    medianprops=dict(color='white', linewidth=1.5),
                    whiskerprops=dict(color=c, alpha=0.6),
                    capprops=dict(color=c, alpha=0.6),
                    flierprops=dict(marker='.', markersize=2, color=c, alpha=0.4),
                    showfliers=True)
    legend_patches = [Patch(facecolor=c, alpha=0.5, label=f'n={n} ({n}s)') for n, c in zip(show_avgs, colors_box)]
    ax1.legend(handles=legend_patches, fontsize=LEGEND_SIZE, loc='upper left')
    ax1.set_xticks(x_base)
    ax1.set_xticklabels([f'{pc} {peaks[pc][0]}' for pc in peak_list], fontsize=16, rotation=45, ha='right')
    ax1.set_ylabel('SNR', fontsize=LABEL_SIZE)
    ax1.set_title(f'Per-Peak SNR at Different {method}ing Levels ({direction})', fontsize=TITLE_SIZE)
    ax1.tick_params(axis='y', labelsize=TICK_SIZE)
    ax1.set_facecolor('black')

    # --- TOP-RIGHT: SNR vs count ---
    ax2 = fig.add_subplot(2, 2, 2)
    for i, (pc, c) in enumerate(zip(peaks.keys(), cmap_lines)):
        label = f'{pc} {peaks[pc][0]}'
        ax2.plot(time_s, median_snr[pc], 'o-', color=c, label=label, linewidth=2.5, markersize=8)
    ax2.set_ylim(-1, 25)
    ax2.set_xlabel(f'Number of Spectra {method}d', fontsize=LABEL_SIZE)
    ax2.set_ylabel('SNR (median)', fontsize=LABEL_SIZE)
    ax2.set_title(f'SNR vs {method}ing ({direction})', fontsize=TITLE_SIZE)
    ax2.legend(fontsize=14, loc='upper left', ncol=2)
    ax2.tick_params(axis='both', labelsize=TICK_SIZE)
    ax2.set_facecolor('black')

    # --- BOTTOM: Overlaid spectra ---
    ax3 = fig.add_subplot(2, 1, 2)
    x_rs = rs_crop if is_processed else rs_full
    for idx, (n_avg, c) in enumerate(zip(spec_avg_counts, cmap_spec)):
        ax3.plot(x_rs, spec_dict[n_avg], color=c, linewidth=1.5, label=display_labels[n_avg])
    ax3.legend(fontsize=LEGEND_SIZE, loc='upper right' if not is_processed else 'upper left',
               title=f'{method}ing over', title_fontsize=LEGEND_SIZE + 2, ncol=2)

    ax3.set_xlabel('Raman Shift (cm$^{-1}$)', fontsize=LABEL_SIZE)
    if is_processed:
        ax3.set_ylabel('Normalized Intensity', fontsize=LABEL_SIZE)
        ax3.set_title(f'Overlaid Processed Spectra: Sum Raw ({direction}) Then Process', fontsize=TITLE_SIZE)
        ax3.set_ylim(-0.05, 1.15)
    else:
        ax3.set_ylabel('Intensity (counts)', fontsize=LABEL_SIZE)
        ax3.set_title(f'Overlaid Raw Averaged Spectra ({direction})', fontsize=TITLE_SIZE)
    ax3.tick_params(axis='both', labelsize=TICK_SIZE)
    ax3.set_facecolor('black')

    plt.tight_layout()
    output_path = data_dir / f'snr_analysis_{suffix}.png'
    plt.savefig(output_path, dpi=150, facecolor='black', bbox_inches='tight')
    plt.close()
    print(f'Saved {output_path}')

# Print summary for last 300
for method, label, dist_e, dist_f in [
    ('Average', 'avg', avg_dist_end, avg_dist_front),
    ('Sum', 'sum', sum_dist_end, sum_dist_front),
]:
    for direction, snr_dist in [('From End', dist_e), ('From Front', dist_f)]:
        print(f'\nSNR summary — {method}, last 300 spectra ({direction}):')
        print(f'{"Peak":>6} {"Assignment":<20} {"SNR(1)":>8} {"SNR(10)":>9} {"SNR(50)":>9} {"SNR(200)":>10} {"SNR(300)":>10}')
        print('-' * 72)
        for pc, (lbl, _) in peaks.items():
            vals = [np.median(snr_dist[n][pc]) if len(snr_dist[n][pc]) > 0 else 0 for n in [1, 10, 50, 200, 300]]
            print(f'{pc:>6} {lbl:<20} {vals[0]:>8.1f} {vals[1]:>9.1f} {vals[2]:>9.1f} {vals[3]:>10.1f} {vals[4]:>10.1f}')

# ========== ALL 599 SPECTRA, FROM FRONT ==========
print("\n===== All 599 spectra, from front =====")
all_avg_counts = [1, 2, 5, 10, 20, 50, 100, 200, 300, 599]

print("Computing SNR distributions (average, all 599 from front)...")
all_avg_dist = compute_snr_front(raw_clean, rs_full, 'mean', all_avg_counts)
print("Computing SNR distributions (sum, all 599 from front)...")
all_sum_dist = compute_snr_front(raw_clean, rs_full, 'sum', all_avg_counts)

# Spectra for bottom plot
all_raw_avg_front = {}
all_proc_sum_front = {}
for n in all_avg_counts:
    all_raw_avg_front[n] = raw_clean[:n].mean(axis=0)
    summed = raw_clean[:n].sum(axis=0)
    proc, rs_c_all = process_spectrum(summed, rs_full)
    all_proc_sum_front[n] = proc
rs_crop_all = rs_c_all

# Plot settings for all-599
all_display_labels = {n: f'{n}s' for n in all_avg_counts}
all_time_s = np.array(all_avg_counts)
all_n_levels = len(all_avg_counts)
all_cmap_spec = plt.cm.rainbow(np.linspace(0.0, 1.0, all_n_levels))
all_spec_color_map = {n: all_cmap_spec[i] for i, n in enumerate(all_avg_counts)}
all_show_avgs = [1, 10, 50, 200]
all_colors_box = [all_spec_color_map[n] for n in all_show_avgs]

all_plot_configs = [
    ('Average', all_avg_dist, all_raw_avg_front, 'avg_first_from_front_all599', False),
    ('Sum',     all_sum_dist, all_proc_sum_front, 'sum_first_from_front_all599', True),
]

for method, snr_dist, spec_dict, suffix, is_processed in all_plot_configs:
    median_snr = get_median_snr(snr_dist, all_avg_counts)
    print(f"Plotting {method} From Front (all 599)...")
    fig = plt.figure(figsize=(30, 24), dpi=150)
    fig.patch.set_facecolor('black')
    fig.suptitle(f'SNR Analysis: All 599 Spectra, {method} Raw (From Front) Then Process',
                 fontsize=SUPTITLE_SIZE, y=1.01)

    # --- TOP-LEFT: Boxplot ---
    ax1 = fig.add_subplot(2, 2, 1)
    width = 0.18
    for j, (n_avg, c) in enumerate(zip(all_show_avgs, all_colors_box)):
        data_bp = [snr_dist[n_avg][pc] for pc in peak_list]
        positions = x_base + (j - 1.5) * width
        ax1.boxplot(data_bp, positions=positions, widths=width*0.85, patch_artist=True,
                    boxprops=dict(facecolor=c, alpha=0.4, edgecolor=c),
                    medianprops=dict(color='white', linewidth=1.5),
                    whiskerprops=dict(color=c, alpha=0.6),
                    capprops=dict(color=c, alpha=0.6),
                    flierprops=dict(marker='.', markersize=2, color=c, alpha=0.4),
                    showfliers=True)
    legend_patches = [Patch(facecolor=c, alpha=0.5, label=f'n={n} ({n}s)') for n, c in zip(all_show_avgs, all_colors_box)]
    ax1.legend(handles=legend_patches, fontsize=LEGEND_SIZE, loc='upper left')
    ax1.set_xticks(x_base)
    ax1.set_xticklabels([f'{pc} {peaks[pc][0]}' for pc in peak_list], fontsize=16, rotation=45, ha='right')
    ax1.set_ylabel('SNR', fontsize=LABEL_SIZE)
    ax1.set_title(f'Per-Peak SNR at Different {method}ing Levels (From Front)', fontsize=TITLE_SIZE)
    ax1.tick_params(axis='y', labelsize=TICK_SIZE)
    ax1.set_facecolor('black')

    # --- TOP-RIGHT: SNR vs count ---
    ax2 = fig.add_subplot(2, 2, 2)
    for i, (pc, c) in enumerate(zip(peaks.keys(), cmap_lines)):
        label = f'{pc} {peaks[pc][0]}'
        ax2.plot(all_time_s, median_snr[pc], 'o-', color=c, label=label, linewidth=2.5, markersize=8)
    ax2.set_ylim(-1, 25)
    ax2.set_xlabel(f'Number of Spectra {method}d', fontsize=LABEL_SIZE)
    ax2.set_ylabel('SNR (median)', fontsize=LABEL_SIZE)
    ax2.set_title(f'SNR vs {method}ing (From Front)', fontsize=TITLE_SIZE)
    ax2.legend(fontsize=14, loc='upper left', ncol=2)
    ax2.tick_params(axis='both', labelsize=TICK_SIZE)
    ax2.set_facecolor('black')

    # --- BOTTOM: Overlaid spectra ---
    ax3 = fig.add_subplot(2, 1, 2)
    x_rs = rs_crop_all if is_processed else rs_full
    for idx, (n_avg, c) in enumerate(zip(all_avg_counts, all_cmap_spec)):
        ax3.plot(x_rs, spec_dict[n_avg], color=c, linewidth=1.5, label=all_display_labels[n_avg])
    ax3.legend(fontsize=LEGEND_SIZE, loc='upper right' if not is_processed else 'upper left',
               title=f'{method}ing over', title_fontsize=LEGEND_SIZE + 2, ncol=2)
    ax3.set_xlabel('Raman Shift (cm$^{-1}$)', fontsize=LABEL_SIZE)
    if is_processed:
        ax3.set_ylabel('Normalized Intensity', fontsize=LABEL_SIZE)
        ax3.set_title(f'Overlaid Processed Spectra: Sum Raw (From Front) Then Process', fontsize=TITLE_SIZE)
        ax3.set_ylim(-0.05, 1.15)
    else:
        ax3.set_ylabel('Intensity (counts)', fontsize=LABEL_SIZE)
        ax3.set_title(f'Overlaid Raw Averaged Spectra (From Front)', fontsize=TITLE_SIZE)
    ax3.tick_params(axis='both', labelsize=TICK_SIZE)
    ax3.set_facecolor('black')

    plt.tight_layout()
    output_path = data_dir / f'snr_analysis_{suffix}.png'
    plt.savefig(output_path, dpi=150, facecolor='black', bbox_inches='tight')
    plt.close()
    print(f'Saved {output_path}')

# Print summary for all 599
for method, snr_dist in [('Average', all_avg_dist), ('Sum', all_sum_dist)]:
    print(f'\nSNR summary — {method}, all 599 spectra (From Front):')
    print(f'{"Peak":>6} {"Assignment":<20} {"SNR(1)":>8} {"SNR(10)":>9} {"SNR(50)":>9} {"SNR(200)":>10} {"SNR(599)":>10}')
    print('-' * 72)
    for pc, (lbl, _) in peaks.items():
        vals = [np.median(snr_dist[n][pc]) if len(snr_dist[n][pc]) > 0 else 0 for n in [1, 10, 50, 200, 599]]
        print(f'{pc:>6} {lbl:<20} {vals[0]:>8.1f} {vals[1]:>9.1f} {vals[2]:>9.1f} {vals[3]:>10.1f} {vals[4]:>10.1f}')
