import os
import random
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, recall_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from noodlepy.utils.bec_hnc_dataset import Bec_HNC_Dataset
from noodlepy.utils.spectrumpreprocessor import SpectrumPreprocessor

# ---------------------- CONFIGURATION ----------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

BASE_DIR = os.getcwd()
TRAIN_DATA_FOLDER = os.path.join(BASE_DIR, "noodlepy", "data", "bec_hnc_train")
TRAIN_META_FILE   = os.path.join(BASE_DIR, "noodlepy", "data", "bec_hnc_patient_annotations.xlsx")
TEST_DATA_FOLDER  = os.path.join(BASE_DIR, "noodlepy", "data", "bec_hnc_test")
TEST_META_FILE    = os.path.join(BASE_DIR, "noodlepy", "data", "bec_hnc_patient_annotations.xlsx")

preprocessor = SpectrumPreprocessor(
    cropping=False,
    baseline_correction=True,
    remove_cosmic_rays=False,
    normalization=True,
    smoothing=True
)

i_filters = {
    "All rings": list(range(4, 48)),
    "3-9 rings": [3,4,5,6,7,8,9,48,47,46,45,44,43,42],
    "10-25 rings": list(range(10,41))
}

grid_C = [0.02]
bin_sizes = [1]
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# Custom scorers
def make_scorers():
    from sklearn.metrics import make_scorer
    sensitivity = make_scorer(recall_score, pos_label=1)
    specificity = make_scorer(lambda yt, yp: recall_score(yt, yp, pos_label=0))
    return {'accuracy':'accuracy', 'sensitivity':sensitivity, 'specificity':specificity, 'roc_auc':'roc_auc'}
scoring = make_scorers()

# Helper to aggregate dataset
def aggregate_dataset(dataset, r_filter):
    # X_list, y_list = [], []
    # patient_data = {}
    
    # for spectrum, _, meta in dataset:
    #     if meta['ring'] in r_filter:
    #         patient_id = meta['patient_id']
    #         if patient_id not in patient_data:
    #             patient_data[patient_id] = {'spectra': [], 'staging': meta['staging']}
    #         patient_data[patient_id]['spectra'].append(spectrum.squeeze(0).numpy())
    
    # for patient_id, data in patient_data.items():
    #     averaged_spectrum = np.mean(data['spectra'], axis=0)
    #     X_list.append(averaged_spectrum)
    #     y_list.append(data['staging'])
    X_list, y_list = [], []
    for spectrum, _, meta in dataset:
        if meta['ring'] in r_filter:
            X_list.append(spectrum.squeeze(0).numpy())
            y_list.append(meta['staging'])

    return np.vstack(X_list), np.array(y_list)

# Load data
train_ds = Bec_HNC_Dataset(TRAIN_DATA_FOLDER, TRAIN_META_FILE, r_filter=None, preprocessor=preprocessor, augmentor=None)
test_ds  = Bec_HNC_Dataset(TEST_DATA_FOLDER,  TEST_META_FILE,  r_filter=None, preprocessor=preprocessor, augmentor=None)

# Containers for ROC and baseline points
roc_test = {bs: {} for bs in bin_sizes}
roc_train = {bs: {} for bs in bin_sizes}
baseline_points = {bs: {} for bs in bin_sizes}
results = []

# Main loop
for filter_name, rings in i_filters.items():
    # Aggregate full data
    X_train_full, y_train = aggregate_dataset(train_ds, rings)
    X_test_full,  y_test  = aggregate_dataset(test_ds,  rings)

    # Compute baseline ROC point for majority classifier on test set
    majority_test = np.bincount(y_test).argmax()
    y_pred_base = np.full_like(y_test, fill_value=majority_test)
    tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_test, y_pred_base).ravel()
    baseline_fpr = fp_b / (fp_b + tn_b)
    baseline_tpr = tp_b / (tp_b + fn_b)

    for bin_size in bin_sizes:
        # Bin features
        n_feat = X_train_full.shape[1]
        n_bins = n_feat // bin_size
        rem    = n_feat % bin_size
        def bin_data(X):
            X_b = X.reshape(-1, n_bins, bin_size).mean(axis=2)
            if rem:
                last = X[:, n_bins*bin_size:].mean(axis=1, keepdims=True)
                X_b = np.hstack([X_b, last])
            return X_b
        Xb_train = bin_data(X_train_full)
        Xb_test  = bin_data(X_test_full)

        # Pipeline and Grid Search
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000, random_state=SEED))
        ])
        grid = GridSearchCV(
            estimator=pipe,
            param_grid={'clf__C': grid_C},
            scoring=scoring,
            refit='accuracy',
            cv=inner_cv,
            n_jobs=-1,
            return_train_score=False
        )
        grid.fit(Xb_train, y_train)
        model = grid.best_estimator_

        # Compute ROCs
        y_scores_test  = model.decision_function(Xb_test)
        fpr_t, tpr_t, _ = roc_curve(y_test, y_scores_test)
        roc_test[bin_size][filter_name] = (fpr_t, tpr_t)
        y_scores_train = model.decision_function(Xb_train)
        fpr_r, tpr_r, _ = roc_curve(y_train, y_scores_train)
        roc_train[bin_size][filter_name] = (fpr_r, tpr_r)

        # Store baseline point
        baseline_points[bin_size][filter_name] = (baseline_fpr, baseline_tpr)

        # Record metrics
        tn, fp, fn, tp = confusion_matrix(y_test, model.predict(Xb_test)).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        results.append({
            'filter': filter_name,
            'bin_size': bin_size,
            'best_C': grid.best_params_['clf__C'],
            'baseline_test_acc': np.bincount(y_test).max() / len(y_test),
            'cv_accuracy': grid.best_score_,
            'test_accuracy': (tp+tn)/(tp+tn+fp+fn),
            'test_roc_auc': auc(fpr_t, tpr_t),
            'sensitivity': sensitivity,
            'specificity': specificity
        })

# Plot combined ROC with baseline points
colors = plt.cm.tab10(np.arange(len(i_filters)))
for bin_size in bin_sizes:
    plt.figure()
    for idx, flt in enumerate(i_filters.keys()):
        c = colors[idx]
        # test/train curves
        fpr_t, tpr_t  = roc_test[bin_size][flt]
        fpr_r, tpr_r  = roc_train[bin_size][flt]
        plt.plot(fpr_t, tpr_t, lw=2, label=f"{flt} (test)", color=c)
        plt.plot(fpr_r, tpr_r, lw=2, linestyle='--', label=f"{flt} (train)", color=c)
    # random guessing line adjusted for class imbalance
    p = np.bincount(y_test)[majority_test] / len(y_test)  # proportion of majority class in test set
    plt.plot([0, 1], [0, 1], '--', color='gray', lw=1, label='Random Guessing')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title(f'L1 Penalized Logistic ROC Comparison')
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'ROC Comparison of Models Trained on Different Rings', dpi=300, bbox_inches='tight')
    plt.close()


# Summarize and save metrics
summary_df = pd.DataFrame(results).pivot_table(
    index=['filter', 'bin_size'],
    values=['test_accuracy', 'test_roc_auc', 'sensitivity', 'specificity'],
    aggfunc='first'
)
print(summary_df)


