import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
import random
import os
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from noodlepy.utils.bec_hnc_dataset import Bec_HNC_Dataset
from noodlepy.utils.spectrumpreprocessor import SpectrumPreprocessor

def aggregate_dataset(dataset, filter=None):

    # ########## # aggregate spectra per patient and average them
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
        if meta['ring'] in filter:
            spectra_list.append(spec1.squeeze(0).numpy())
            labels_list.append(meta['staging'])

    # build X, y
    X_full = np.array(spectra_list)
    y = np.array(labels_list)

    return X_full, y, wavenumber

# simple 5-fold CV with grid search for C
def kfold_simple(X, y, Cs, cv):
    y_true, y_pred, coeffs, best_Cs = [], [], [], []
    for tr, te in cv.split(X, y):  # Using StratifiedKFold for CV
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        # grid search
        grid = GridSearchCV(
            LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000),
            param_grid={'C': Cs},
            cv=cv,
            n_jobs=1
        )
        grid.fit(X_tr, y_tr)
        best_Cs.append(grid.best_params_['C'])

        best_model = grid.best_estimator_
        coeffs.append(best_model.coef_[0])
        y_pred.extend(best_model.predict(X_te))
        y_true.extend(y_te)

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

    # Calculate the majority class ratios as the baseline accuracy
    baseline_accuracy = np.max(np.bincount(y) / len(y))

    return acc, sens, spec, mean_coeffs, best_model, baseline_accuracy

####################################################################################################

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
    remove_cosmic_rays=False,
    normalization=True,
    smoothing=True
)

# your three r_filter settings
r_filters = {
    "All": list(range(3, 48)),  # all rings],
    "Edge Only": [3,4,5,6,7,8,9,42,43,44,45, 46,47,48],
    "Center Only": [19,20,21,22,23,24,25,26,27,28,29,30,31,32]
}#    

# bin sizes and C values
bin_sizes = [1]#, 5, 10, 20, 50
C_values = [[0.21]]

# CV objects
inner_cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

accuracy_bin_1_train = {name: [] for name in r_filters.keys()}
accuracy_bin_1_test = {name: [] for name in r_filters.keys()}

train_ds = Bec_HNC_Dataset(data_folder, metadata_file, r_filter = None, preprocessor = preprocessor, augmentor=None)
test_ds  = Bec_HNC_Dataset(test_data_folder, test_meta_file, r_filter = None, preprocessor = preprocessor, augmentor=None)

plt.figure(figsize=(4, 4))
ax = plt.gca()
ax.tick_params(axis='both', colors='white')
for spine in ax.spines.values():
    spine.set_edgecolor('white')
for c in C_values:
    # initialize result containers
    results       = {}
    test_results  = {}
    feature_selection = {}
    C_selection   = {}
    colors = plt.cm.tab10(np.arange(len(r_filters))) 
    print(f"Feature density = {c[0]}")
    i = 0
    for name, r_filter in r_filters.items():
        # aggregate
        X_train_full, y_train, wvnr = aggregate_dataset(train_ds, r_filter)
        X_test_full,  y_test,  _   = aggregate_dataset(test_ds, r_filter)
        print(f"Aggregated {len(X_train_full)} training samples and {len(X_test_full)} test samples for filter '{name}'")

        results[name]      = {'accuracy': [], 'sensitivity': [], 'specificity': [], 'baseline_accuracy': []}
        test_results[name] = {'accuracy': [], 'sensitivity': [], 'specificity': [], 'baseline_accuracy': [], 'roc_auc': []}
        feature_selection[name] = {}
        C_selection[name] = []

        for bin_size in bin_sizes:
            # binning
            n_bins    = X_train_full.shape[1] // bin_size
            rem       = X_train_full.shape[1] % bin_size

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
            scaler = StandardScaler().fit(train_binned)
            X_tr_sc = scaler.transform(train_binned)
            X_te_sc = scaler.transform(test_binned)

            # nested LOOCV
            acc, sens, spec, coeffs, best_model, baseline_acc = kfold_simple(X_tr_sc, y_train, c, inner_cv)
            results[name]['accuracy'].append(acc)
            results[name]['sensitivity'].append(sens)
            results[name]['specificity'].append(spec)
            results[name]['baseline_accuracy'].append(baseline_acc)
            feature_selection[name][bin_size] = {'coefficients': coeffs, 'wavenumbers': wv_bins}
            
            # final fit on full data with best model, evaluate on external test
            y_pred_test = best_model.predict(X_te_sc)
            TP_t = np.sum((y_test==1)&(y_pred_test==1))
            TN_t = np.sum((y_test==0)&(y_pred_test==0))
            FP_t = np.sum((y_test==0)&(y_pred_test==1))
            FN_t = np.sum((y_test==1)&(y_pred_test==0))
            acc_t  = (TP_t+TN_t)/len(y_test)
            sens_t = TP_t/(TP_t+FN_t) if (TP_t+FN_t)>0 else 0.0
            spec_t = TN_t/(TN_t+FP_t) if (TN_t+FP_t)>0 else 0.0
            baseline_acc_t = np.max(np.bincount(y_test) / len(y_test))

            if bin_size == 1:
                accuracy_bin_1_train[name].append(acc)
                accuracy_bin_1_test[name].append(acc_t)

            # # calculate the number of spectra were predicted correctly for each patient
            # patient_data = {}
            # for i in range(len(test_ds)):
            #     spec, wavenumber, meta = test_ds[i]
            #     pid = meta['patient_id']
            #     if pid not in patient_data:
            #         patient_data[pid] = {"predicted": [], "label": meta['staging']}
            #     patient_data[pid]['predicted'].append(y_pred_test[i])
            # pids     = sorted(patient_data)
            # y_pred_patient = np.array([np.mean(patient_data[pid]['predicted']) for pid in pids])
            # y_true_patient = np.array([patient_data[pid]['label'] for pid in pids])
            # # 

            # plot the confusion matrix for the test set at c = 0.02 for each filter

            # cm = confusion_matrix(y_test, y_pred_test, labels=[0, 1])
            # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Healthy', 'Cancer'])
            # disp.plot(cmap='Blues')
            # plt.title(f'Confusion Matrix for {name} (C={c[0]})')
            # plt.savefig(f'confusion_matrix_{name}_C{c[0]}.png', dpi=300)
            # plt.close()

            # Compute ROC curve and AUC for the test set
            y_score = best_model.decision_function(X_te_sc)
            fpr, tpr, _ = roc_curve(y_test, y_score)
            roc_auc = auc(fpr, tpr)


            # Plot the ROC curve
            plt.plot(fpr, tpr, lw=2, label=f"{name} (test)", color=colors[i])
            fpr_r, tpr_r, _ = roc_curve(y_train, best_model.decision_function(X_tr_sc))
            offset = i * 0.005  # adjust the multiplier for a bigger/smaller offset
            plt.plot(fpr_r + offset, tpr_r + offset, lw=2, linestyle='--', color=colors[i], label=f"{name} (train)")
            i += 1


            test_results[name]['accuracy'].append(acc_t)
            test_results[name]['sensitivity'].append(sens_t)
            test_results[name]['specificity'].append(spec_t)
            test_results[name]['baseline_accuracy'].append(baseline_acc_t)
            test_results[name]['roc_auc'].append(roc_auc)

            print(f"[{name}] bin={bin_size:3d} → CV acc={acc:.3f}, sens={sens:.3f}, spec={spec:.3f}, baseline_acc={baseline_acc:.3f}")
            print(f"[{name}] bin={bin_size:3d} → TEST acc={acc_t:.3f}, sens={sens_t:.3f}, spec={spec_t:.3f}, baseline_acc={baseline_acc_t:.3f}\n, roc_auc={roc_auc:.3f}")
    
    plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--', label='Random Guessing')
    plt.gcf().patch.set_alpha(0.0)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', color='white')
    plt.ylabel('True Positive Rate', color='white')
    leg = plt.legend(loc="lower right")
    leg.get_frame().set_alpha(0.0)
    plt.setp(leg.get_texts(), color='white')
    plt.savefig(f'qe_roc_curve_all_three.svg', dpi=300, transparent=True)



# feature selection plot with transparent background and white labels/ticks
fig, axes = plt.subplots(1, 1, figsize=(8, 6), sharex=True)
fig.patch.set_alpha(0.0)
colors = plt.cm.tab10(np.arange(len(feature_selection)))
for i, (name, data) in enumerate(feature_selection.items()):
    fig, axes = plt.subplots(1, 1, figsize=(8, 6), sharex=True)
    bin_size = 1
    if bin_size in data:
        coeffs = data[bin_size]['coefficients']
        wv_bins = data[bin_size]['wavenumbers']
        axes.plot(wv_bins, coeffs, label="Coefficients", color=colors[i])
        axes.set_ylabel('Coefficient', color='white', fontsize=16)
        axes.legend(fontsize=16, facecolor='none', edgecolor='white', labelcolor='white')
        axes.grid(color='white', alpha=0.3)
    axes.tick_params(axis='both', colors='white', labelsize=16)
    for spine in axes.spines.values():
        spine.set_color('white')

    plt.suptitle('L1 Penalized Logistic Regression Coefficients', fontsize=16, color='white')
    plt.xlabel('Raman Shift (cm$^{-1}$)', fontsize=16, color='white')
    plt.subplots_adjust(top=0.85, left=0.05, right=0.98, wspace=0.1)
    plt.savefig(f'feature_selection_plot{name}.png', dpi=300, bbox_inches='tight', transparent=True)

# re-organize feature selection for bin 1 saving to csv
# # each column is a filter, each row is a coefficient for a wavenumber
# coeffs_df = pd.DataFrame(index=wv_bins)
# for name, data in feature_selection.items():
#     bin_size = 1
#     if bin_size in data:
#         coeffs = data[bin_size]['coefficients']
#         coeffs_df[name] = coeffs
# coeffs_df.to_csv('feature_selection.csv', index=True)


# plot the accuracy for each filter of bin 1 with different feature density 
# plt.figure(figsize=(6, 6))
# colors = plt.cm.tab10(np.arange(len(r_filters)))  # Generate distinct colors for each filter
# for idx, (name, color) in enumerate(zip(r_filters.keys(), colors)):
#     plt.plot([c[0] for c in C_values], accuracy_bin_1_train[name], label=f"{name} (CV)", marker='o', color=color)
#     plt.plot([c[0] for c in (C_values+ [[1], [10], [50], [100]])], accuracy_bin_1_test[name], label=f"{name} (TEST)", linestyle='--', marker='x', color=color)
#     plt.axhline(y=results[name]['baseline_accuracy'][0], color='k', label=f"{name} (Baseline)")
#     plt.axhline(y=test_results[name]['baseline_accuracy'][0], color='k', linestyle='--', label=f"{name} (Baseline Test)")
# plt.xscale('log')  # Set x-axis to log scale
# plt.xlabel('L1 Penalty Strength (lambda)')
# plt.ylabel('Accuracy')
# plt.title('Train cross-validation & Test Accuracy VS. Feature ')
# plt.xticks([c[0] for c in C_values], [f'{c[0]:.2f}' for c in C_values])
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.grid()
# plt.show()
# plt.savefig('accuracy_vs_lambda_log_long.png', dpi=300, bbox_inches='tight')

print('Done!')