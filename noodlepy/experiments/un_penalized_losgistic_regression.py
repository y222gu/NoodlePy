import os
import random
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, confusion_matrix, roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt
from noodlepy.utils.bec_hnc_dataset import Bec_HNC_Dataset
from noodlepy.utils.spectrumpreprocessor import SpectrumPreprocessor
from sklearn.metrics import ConfusionMatrixDisplay

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

# Preprocessor
preprocessor = SpectrumPreprocessor(
    cropping=False,
    baseline_correction=True,
    remove_cosmic_rays=False,
    normalization=True,
    smoothing=True
)

# Ring filters
i_filters = {
    "All": list(range(3, 48)),

}    #"Edge Only": [3,4,5,6,7,8,9,48,47,46,45,44,43,42],
    #"Center Only": list(range(10,41))

# Binning and CV
bin_sizes = [1]
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# Helper to aggregate
def aggregate_dataset(dataset, r_filter):
    X_list, y_list = [], []
    for spectrum, _, meta in dataset:
        if meta['ring'] in r_filter:
            X_list.append(spectrum.squeeze(0).numpy())
            y_list.append(meta['staging'])
    return np.vstack(X_list), np.array(y_list)

# Load datasets
train_ds = Bec_HNC_Dataset(TRAIN_DATA_FOLDER, TRAIN_META_FILE, r_filter=None, preprocessor=preprocessor, augmentor=None)
test_ds  = Bec_HNC_Dataset(TEST_DATA_FOLDER,  TEST_META_FILE,  r_filter=None, preprocessor=preprocessor, augmentor=None)

# Containers
roc_test = {bs: {} for bs in bin_sizes}
roc_train = {bs: {} for bs in bin_sizes}
baseline_points = {bs: {} for bs in bin_sizes}
results = []

# Main loop: unpenalized logistic regression
for filter_name, rings in i_filters.items():
    # Prepare data
    X_train_full, y_train = aggregate_dataset(train_ds, rings)
    X_test_full,  y_test  = aggregate_dataset(test_ds,  rings)

    # Baseline (majority classifier) on test set
    maj_class = np.bincount(y_test).argmax()
    y_pred_base = np.full_like(y_test, fill_value=maj_class)
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

        # Unpenalized logistic regression
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000, random_state=SEED))
        ])
        pipeline.fit(Xb_train, y_train)

        # ROC curves
        y_scores_test  = pipeline.decision_function(Xb_test)
        fpr_t, tpr_t, _ = roc_curve(y_test, y_scores_test)
        roc_test[bin_size][filter_name] = (fpr_t, tpr_t)
        y_scores_train = pipeline.decision_function(Xb_train)
        fpr_r, tpr_r, _ = roc_curve(y_train, y_scores_train)
        roc_train[bin_size][filter_name] = (fpr_r, tpr_r)

        # Store baseline
        baseline_points[bin_size][filter_name] = (baseline_fpr, baseline_tpr)

        # Metrics
        y_pred_test = pipeline.predict(Xb_test)
        acc_test = accuracy_score(y_test, y_pred_test)
        roc_auc_test = auc(fpr_t, tpr_t)
        acc_train = accuracy_score(y_train, pipeline.predict(Xb_train))
        
        # Sensitivity and Specificity
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        results.append({
            'filter': filter_name,
            'test_accuracy': acc_test,
            'train_accuracy': acc_train,
            'test_roc_auc': roc_auc_test,
            'sensitivity': sensitivity,
            'specificity': specificity
        })

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()
        cm = confusion_matrix(y_test, y_pred_test)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipeline.classes_)
        disp.plot(cmap='Blues', values_format='d')
        plt.title(f'Confusion Matrix for {filter_name} with Unpenalized Logistic Regression')
        # Set a fixed maximum for the color bar
        vmax = max(cm.max(), 1)  # Ensure at least 1 to avoid issues with empty matrices
        colorbar = plt.colorbar(disp.im_, ax=disp.ax_, fraction=0.046, pad=0.04)
        disp.im_.set_clim(0, vmax)
        plt.savefig(f'Confusion Matrix for {filter_name} with Unpenalized Logistic Regression', dpi=300, bbox_inches='tight')
        plt.close()
        # Print results
        # Plot combined ROC with baseline
        colors = plt.cm.tab10(np.arange(len(i_filters)))
        for bin_size in bin_sizes:
            plt.figure(figsize=(4, 4), facecolor='white')  #
            for idx, flt in enumerate(i_filters.keys()):
                c = colors[idx]
                fpr_t, tpr_t = roc_test[bin_size][flt]
                fpr_r, tpr_r = roc_train[bin_size][flt]
                bfpr, btpr = baseline_points[bin_size][flt]
                plt.plot(fpr_t, tpr_t, lw=2, label="Unpenalized (test)", color=c)
                plt.plot(fpr_r, tpr_r, lw=2, linestyle='--', label="Unpenalized (train)", color=c)
            plt.plot([0, 1], [0, 1], '--', color='gray', lw=2)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            
            ax = plt.gca()
            ax.set_facecolor('none')
            ax.tick_params(axis='x')
            ax.tick_params(axis='y')
            for spine in ax.spines.values():
                spine.set_edgecolor('white')
            
            leg = plt.legend(loc="lower right", facecolor='none', edgecolor='none')
            plt.savefig('unpenalized_roc.svg', dpi=300, bbox_inches='tight', transparent=True)

# pritn results
df = pd.DataFrame(results)
print(df)
