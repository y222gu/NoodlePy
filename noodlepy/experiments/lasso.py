import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
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

# data paths (set your external test set here)
data_folder       = os.path.join(os.getcwd(), "noodlepy", "data", "bec_hnc_train")
metadata_file     = os.path.join(os.getcwd(), "noodlepy", "data", "bec_hnc_patient_annotations.xlsx")
test_data_folder  = os.path.join(os.getcwd(), "noodlepy", "data", "bec_hnc_test")   
test_meta_file    = os.path.join(os.getcwd(), "noodlepy", "data", "bec_hnc_patient_annotations.xlsx")  

# preprocessor
preprocessor = SpectrumPreprocessor(
    cropping=False,
    baseline_correction=True,
    remove_cosmic_rays=True,
    normalization=True,
    smoothing=True
)

# your three r_filter settings
r_filters = {
    "no_filter": list(range(4, 47)),
    "3-7": [4,5,6,7, 47,46,45,44],
    "8-12": [8,9,10,11,12, 43,42,41,40,39],
    "13-25": list(range(13,26)) + list(range(26,38)),
}

# bin sizes and C values
bin_sizes = [1, 5, 10, 20, 50]
C_values  = [0.1] # 0.001, 0.01, 0.1, 1, 10, 100

# CV objects
loo       = LeaveOneOut()
inner_cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# helper to aggregate per-patient

def aggregate_dataset(dataset):
    ########### # aggregate spectra per patient and average them
    # patient_data = {}
    # for i in range(len(dataset)):
    #     spec, wavenumber, meta = dataset[i]
    #     pid = meta['patient_id']
    #     if pid not in patient_data:
    #         patient_data[pid] = {"spectra": [], "label": meta['staging']}
    #     patient_data[pid]['spectra'].append(spec.squeeze(0).numpy())

    # pids     = sorted(patient_data)
    # X_full   = np.array([np.mean(patient_data[pid]['spectra'], axis=0) for pid in pids])
    # y        = np.array([patient_data[pid]['label'] for pid in pids])
    # wavenumber = wavenumber # same for all


    # without averaging
    spectra_list = []
    labels_list = []
    for i in range(len(dataset)):
        spec1, wavenumber, meta = dataset[i]
        spectra_list.append(spec1.squeeze(0).numpy())
        labels_list.append(meta['staging'])

    # build X, y
    X_full = np.array(spectra_list)
    y = np.array(labels_list)

    return X_full, y, wavenumber

# nested LOOCV with grid search for C

def loocv_nested(X, y, Cs, inner_cv):
    y_true, y_pred, coeffs, best_Cs = [], [], [], []
    for tr, te in loo.split(X):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        print('starting inner CV...')
        # inner grid search
        grid = GridSearchCV(
            LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000),
            param_grid={'C': Cs},
            cv=inner_cv,
            n_jobs=1
        )
        grid.fit(X_tr, y_tr)
        best_Cs.append(grid.best_params_['C'])

        best_model = grid.best_estimator_
        coeffs.append(best_model.coef_[0])
        y_pred.append(best_model.predict(X_te)[0])
        y_true.append(y_te[0])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    TP = np.sum((y_true==1)&(y_pred==1))
    TN = np.sum((y_true==0)&(y_pred==0))
    FP = np.sum((y_true==0)&(y_pred==1))
    FN = np.sum((y_true==1)&(y_pred==0))

    acc  = (TP+TN)/len(y_true)
    sens = TP/(TP+FN) if (TP+FN)>0 else 0.0
    spec = TN/(TN+FP) if (TN+FP)>0 else 0.0
    mean_coeffs = np.mean(np.abs(coeffs), axis=0)

    return acc, sens, spec, mean_coeffs, best_model

# initialize result containers
results       = {}
test_results  = {}
feature_selection = {}
C_selection   = {}

for name, r_filter in r_filters.items():
    # load train & test datasets
    train_ds = Bec_HNC_Dataset(data_folder, metadata_file, r_filter, preprocessor, augmentor=None)
    test_ds  = Bec_HNC_Dataset(test_data_folder, test_meta_file, r_filter, preprocessor, augmentor=None)

    # aggregate
    X_train_full, y_train, wvnr = aggregate_dataset(train_ds)
    X_test_full,  y_test,  _   = aggregate_dataset(test_ds)

    results[name]      = {'accuracy': [], 'sensitivity': [], 'specificity': []}
    test_results[name] = {'accuracy': [], 'sensitivity': [], 'specificity': []}
    feature_selection[name] = {}
    C_selection[name] = []

    for bin_size in bin_sizes:
        # binning
        n_bins    = X_train_full.shape[1] // bin_size
        rem       = X_train_full.shape[1] % bin_size

        print(f"[{name}] binning with bin size {bin_size}...")

        def bin_data(X_full):
            Xb = np.vstack([X_full[:, i*bin_size:(i+1)*bin_size].mean(axis=1) for i in range(n_bins)])
            if rem>0:
                Xb = np.vstack([Xb, X_full[:, n_bins*bin_size:].mean(axis=1)])
            return Xb.T

        train_binned = bin_data(X_train_full)
        test_binned  = bin_data(X_test_full)
        # shared wavenumbers
        wv_bins = np.array([np.median(wvnr[i*bin_size:(i+1)*bin_size]) for i in range(n_bins)] +
                           ([np.median(wvnr[n_bins*bin_size:])] if rem>0 else []))

        # scaling
        print(f"[{name}] scaling...")
        scaler = StandardScaler().fit(train_binned)
        X_tr_sc = scaler.transform(train_binned)
        X_te_sc = scaler.transform(test_binned)
        print(f"[{name}] scaling done.")

        # nested LOOCV
        print(f"[{name}] LOOCV...")
        acc, sens, spec, coeffs, best_model = loocv_nested(X_tr_sc, y_train, C_values, inner_cv)
        results[name]['accuracy'].append(acc)
        results[name]['sensitivity'].append(sens)
        results[name]['specificity'].append(spec)
        feature_selection[name][bin_size] = {'coefficients': coeffs, 'wavenumbers': wv_bins}
        print(f"[{name}] bin={bin_size:3d} → LOOCV acc={acc:.3f}, sens={sens:.3f}, spec={spec:.3f}")

        # final fit on full data with best model, evaluate on external test
        y_pred_test = best_model.predict(X_te_sc)
        TP_t = np.sum((y_test==1)&(y_pred_test==1))
        TN_t = np.sum((y_test==0)&(y_pred_test==0))
        FP_t = np.sum((y_test==0)&(y_pred_test==1))
        FN_t = np.sum((y_test==1)&(y_pred_test==0))
        acc_t  = (TP_t+TN_t)/len(y_test)
        sens_t = TP_t/(TP_t+FN_t) if (TP_t+FN_t)>0 else 0.0
        spec_t = TN_t/(TN_t+FP_t) if (TN_t+FP_t)>0 else 0.0

        test_results[name]['accuracy'].append(acc_t)
        test_results[name]['sensitivity'].append(sens_t)
        test_results[name]['specificity'].append(spec_t)

        print(f"[{name}] bin={bin_size:3d} → LOOCV acc={acc:.3f}, sens={sens:.3f}, spec={spec:.3f}")
        print(f"[{name}] bin={bin_size:3d} → TEST   acc={acc_t:.3f}, sens={sens_t:.3f}, spec={spec_t:.3f}\n")

# plot LOOCV vs Test accuracy for different filters when changing bin size
plt.figure(figsize=(12, 6))
colors = plt.cm.tab10(np.arange(len(r_filters)))  # Generate distinct colors for each filter
for idx, (name, color) in enumerate(zip(r_filters.keys(), colors)):
    plt.plot(bin_sizes, results[name]['accuracy'], label=f"{name} (LOOCV)", marker='o', color=color)
    plt.plot(bin_sizes, test_results[name]['accuracy'], label=f"{name} (TEST)", linestyle='--', marker='x', color=color)
plt.xlabel('Bin Size')
plt.ylabel('Accuracy')
plt.title('LOOCV vs Test Accuracy for Different Filters')
plt.xticks(bin_sizes)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# visualize coefficients for bin size 1 for each filter
plt.figure(figsize=(12, 6))
for name, coeffs in feature_selection.items():
    bin_size = 1
    if bin_size in coeffs:
        plt.plot(coeffs[bin_size]['wavenumbers'], coeffs[bin_size]['coefficients'], label=name)
plt.xlabel('Wavenumber (nm)')
plt.ylabel('Mean Absolute Coefficient Value')
plt.title('LASSO Coefficients for Bin Size 1')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

print('Done!')