import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV
import random
import os
import pandas as pd

from noodlepy.utils.bec_hnc_dataset import Bec_HNC_Dataset
from noodlepy.utils.spectrumpreprocessor import SpectrumPreprocessor

# reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# data
data_folder = os.path.join(os.getcwd(), "noodlepy", "data", "bec_hnc_temp")
metadata_file = os.path.join(os.getcwd(), "noodlepy", "data", "bec_hnc_patient_annotations.xlsx")


# preprocessor
preprocessor = SpectrumPreprocessor(
    cropping=False,
    baseline_correction=True,
    remove_cosmic_rays=True,
    normalization=True,
    smoothing=True
)

# define your three r_filter settings
r_filters = {
    "no_filter": list(range(4, 47)),
    "3-7": [4,5,6,7, 47,46,45,44],
    "8-12": [8,9,10,11,12, 43,42,41,40, 39],
    "13-25": [13,14,15,16,17,18,19,20,21,22,23,24,25, 26,27,28,29,30,31,32,33,34,35,36,37,38],

}
    # "edges_only": [3,4,5,6,7,8,9,10, 11, 12, 13, 14, 15, 47,46,45,44,43,42,41, 40, 39, 38, 37, 36],
    # "inner_only": list(range(16,35))

bin_sizes = [1, 5, 10, 20, 50, 60, 70, 74, 80, 100]

# CV objects
loo = LeaveOneOut()
lasso_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

def loocv_metrics(X, y):
    """Run LOOCV LASSO, return (accuracy, sensitivity, specificity)."""
    y_true, y_pred, coeffs = [], [], []
    for train_idx, test_idx in loo.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        model = LogisticRegressionCV(
            Cs=10,
            penalty='l1',
            solver='liblinear',
            cv=lasso_cv,
            random_state=SEED,
            n_jobs=1,
            max_iter=2000
        )
        model.fit(X_tr, y_tr)
        coeffs.append(model.coef_[0])  # Coefficients for the binary classification (class 0 vs class 1)
        y_pred.append(model.predict(X_te)[0])
        y_true.append(y_te[0])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # basic counts
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    accuracy    = (TP + TN) / len(y_true)
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    coefficients = np.mean(np.abs(coeffs), axis=0)

    return accuracy, sensitivity, specificity, coefficients

# store results
results = {}
feature_selection = {}
for name, r_filter in r_filters.items():
    # load dataset with this filter
    dataset = Bec_HNC_Dataset(
        data_folder,
        metadata_file,
        r_filter,
        preprocessor,
        augmentor = None
    )

    # aggregate per‐patient spectra
    patient_data = {}
    for i in range(len(dataset)):
        spec1, wavenumber, meta = dataset[i]
        pid = meta['patient_id']
        if pid not in patient_data:
            patient_data[pid] = {"spectra": [], "label": meta['staging']}
        patient_data[pid]["spectra"].append(spec1.squeeze(0).numpy())

    # build X, y
    patient_ids = sorted(patient_data)
    X_full = np.array([np.mean(patient_data[pid]["spectra"], axis=0) for pid in patient_ids])
    y = np.array([patient_data[pid]["label"] for pid in patient_ids])

    accs, senss, specs = [], [], []
    coeffs = {}

    for bin_size in bin_sizes:
        num_bins = X_full.shape[1] // bin_size
        remainder = X_full.shape[1] % bin_size

        X_binned = np.vstack([
            X_full[:, i*bin_size:(i+1)*bin_size].mean(axis=1)
            for i in range(num_bins)
        ])

        # Add an additional bin for the remainder columns, if any
        if remainder > 0:
            X_binned = np.vstack([
                X_binned,
                X_full[:, num_bins*bin_size:].mean(axis=1)
            ])

        X_binned = X_binned.T
        X_scaled = StandardScaler().fit_transform(X_binned)

        a, s, t, c = loocv_metrics(X_scaled, y)
        accs.append(a)
        senss.append(s)
        specs.append(t)
        coeffs[bin_size] = { 
            'coefficients': c,
            'wavenumbers': np.array(
            [np.median(wavenumber[i*bin_size:(i+1)*bin_size]) for i in range(num_bins)] +
            ([np.median(wavenumber[num_bins*bin_size:])] if remainder > 0 else [])
            )
        }
        
        print(f'wavenumbers: {coeffs[bin_size]["wavenumbers"]}')
        print(f'coefficients: {coeffs[bin_size]["coefficients"]}')
        print(f"[{name}] bin={bin_size:3d} → acc={a:.3f}, sens={s:.3f}, spec={t:.3f}")

    results[name] = {
        'accuracy':    accs,
        'sensitivity': senss,
        'specificity': specs
    }

    feature_selection[name] = coeffs

# now plot all three metrics
fig, axs = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

metric_names = ['accuracy', 'sensitivity', 'specificity']
y_labels     = ['Accuracy', 'Sensitivity (TPR)', 'Specificity (TNR)']

for ax, metric, ylabel in zip(axs, metric_names, y_labels):
    for name, res in results.items():
        ax.plot(bin_sizes, res[metric], marker='o', label=name)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(title='r_filter')

axs[-1].set_xlabel('Bin size (wavenumbers per bin)')
fig.suptitle('LOOCV LASSO Performance vs. Bin Size\n(Accuracy, Sensitivity, Specificity)')
plt.tight_layout(rect=[0, 0, 1, 0.96])

# plot coefficients for all different bin size for each filter
for i, (name, coeffs) in enumerate(feature_selection.items()):
    fig, axs = plt.subplots(1, 1, figsize=(8, 4))
    for bin_size, coeff in coeffs.items():
        axs.plot(coeff['wavenumbers'], np.abs(coeff['coefficients']), label=f'bin={bin_size}')
    axs.set_title(name)
    axs.grid(alpha=0.3)
    axs.legend(title='Bin size')
    axs.set_xlabel('Wavenumber (nm)')
    axs.set_ylabel('Mean Absolute Coefficient Value')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# compare the feature selection results for bin size of 2 for all filters
fig, axs = plt.subplots(len(r_filters), 1, figsize=(8, 12), sharex=True)
for ax, (name, coeffs) in zip(axs, feature_selection.items()):
    bin_size = 5
    coeff = coeffs[bin_size]
    ax.plot(coeff['wavenumbers'], np.abs(coeff['coefficients']), label=f'bin={bin_size}')
    bin_size = 1
    coeff = coeffs[bin_size]
    ax.plot(coeff['wavenumbers'], np.abs(coeff['coefficients']), label=f'bin={bin_size}')
    ax.set_title(name)
    ax.grid(alpha=0.3)
    ax.legend(title='Bin size')
    ax.set_xlabel('Wavenumber (nm)')
    ax.set_ylabel('Mean Absolute Coefficient Value')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

