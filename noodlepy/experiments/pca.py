import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict, StratifiedKFold
from scipy.stats import ttest_ind
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from noodlepy.utils.bec_hnc_dataset import Bec_HNC_Dataset
from noodlepy.utils.spectrumpreprocessor import SpectrumPreprocessor
from noodlepy.utils.spectrumaugmentor import SpectrumAugmentor
import os
os.environ['PYTHONHASHSEED'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import scipy.stats as stats
import seaborn as sns
from sklearn.metrics import accuracy_score
import random

# -------------------------------
# PCA for exploratory visualization
# -------------------------------
data_folder = os.path.join(os.getcwd(), "noodlepy", "data", "bec_hnc_temp")
metadata_file = os.path.join(os.getcwd(), "noodlepy", "data", "bec_hnc_patient_annotations.xlsx")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

preprocessor = SpectrumPreprocessor(
    cropping=False,
    baseline_correction=True,
    remove_cosmic_rays=True,
    normalization=True,
    smoothing=True
)

r_filter = None

dataset = Bec_HNC_Dataset(
    data_folder,
    metadata_file,
    r_filter,
    preprocessor,
    augmentor = None
)

# Aggregate spectra per patient
patient_data = {}
for i in range(len(dataset)):
    spectrum_tensor_1, spectrum_tensor_2, metadata = dataset[i]
    pid = metadata['patient_id']
    staging = metadata['staging']  # 0 = healthy, 1 = cancer
    if pid not in patient_data:
        patient_data[pid] = {"spectra": [], "label": staging, "date": metadata['date']}
    patient_data[pid]["spectra"].append(
        spectrum_tensor_1.squeeze(0).numpy()
    )

# Build patient-level arrays
patient_ids = sorted(patient_data.keys())
X = np.array([np.mean(patient_data[pid]["spectra"], axis=0)[5:745] for pid in patient_ids])#
y = np.array([patient_data[pid]["label"] for pid in patient_ids])
dates = pd.to_datetime([patient_data[pid]["date"] for pid in patient_ids])
wavenumbers = np.array(dataset.db[0].wavelength_nm)[5:745]

# Bin X by every 149 wavenumbers
bin_size = 2
num_bins = X.shape[1] // bin_size
X_binned = np.array([np.mean(X[:, i*bin_size:(i+1)*bin_size], axis=1) for i in range(num_bins)]).T
wavenumbers_binned = np.array([np.mean(wavenumbers[i*bin_size:(i+1)*bin_size]) for i in range(num_bins)])

X = X_binned
wavenumbers = wavenumbers_binned

scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)
#########################################################################################################


# # 1) Original PCA
# pca = PCA(n_components=3)
# X_pca = pca.fit_transform(X_scaled)

# # 2) Remove PC1 & PC2
# pcs    = pca.components_[:1]          # shape (2, n_features)
# scores = X_scaled.dot(pcs.T)          # (n_samples, 2)
# X_pc12 = scores.dot(pcs)              # reconstruct PC1+PC2 part
# X_res  = X_scaled - X_pc12            # residual matrix

# # 3) PCA on residuals
# pca2    = PCA(n_components=2)
# X_pca2  = pca2.fit_transform(X_res)

# # 4) Prepare date numbers for continuous colormap
# dates_dt  = pd.to_datetime(dates)             # convert strings→datetime
# dates_num = mdates.date2num(dates_dt)         # float days since epoch

# # 5) Plot 2×2 grid
# fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

# # Top-left: Original PCA by label
# axes[0,0].scatter(X_pca[:,0], X_pca[:,1],
#                   c=['red' if yi else 'blue' for yi in y], alpha=0.7)
# axes[0,0].set_title('Original PCA\n(cancer vs healthy)')
# axes[0,0].set_xlabel('PC1'); axes[0,0].set_ylabel('PC2')
# axes[0,0].legend(['Cancer','Healthy'], loc='best')

# # Top-right: Original PCA by date (continuous)
# sc0 = axes[0,1].scatter(X_pca[:,0], X_pca[:,1],
#                         c=dates_num, cmap='viridis', alpha=0.7)
# axes[0,1].set_title('Original PCA\n(colored by date)')
# axes[0,1].set_xlabel('PC1'); axes[0,1].set_ylabel('PC2')
# cb0 = fig.colorbar(sc0, ax=axes[0,1], pad=0.02, aspect=30)
# cb0.ax.yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
# cb0.set_label('Measurement Date')

# # Bottom-left: Residual PCA by label
# axes[1,0].scatter(X_pca2[:,0], X_pca2[:,1],
#                   c=['red' if yi else 'blue' for yi in y], alpha=0.7)
# axes[1,0].set_title('Residual PCA\n(after removing PC1 & PC2)')
# axes[1,0].set_xlabel('PC1_res'); axes[1,0].set_ylabel('PC2_res')
# axes[1,0].legend(['Cancer','Healthy'], loc='best')

# # Bottom-right: Residual PCA by date (continuous)
# sc1 = axes[1,1].scatter(X_pca2[:,0], X_pca2[:,1],
#                         c=dates_num, cmap='viridis', alpha=0.7)
# axes[1,1].set_title('Residual PCA\n(colored by date)')
# axes[1,1].set_xlabel('PC1_res'); axes[1,1].set_ylabel('PC2_res')
# cb1 = fig.colorbar(sc1, ax=axes[1,1], pad=0.02, aspect=30)
# cb1.ax.yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
# cb1.set_label('Measurement Date')


# -------------------------------
# LASSO Logistic Regression with LOOCV
# -------------------------------
X_res  = X_scaled
loo = LeaveOneOut()
lasso_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

y_true = []
y_pred = []
coefs = []

for train_index, test_index in loo.split(X_res):
    X_train, X_test = X_res[train_index], X_res[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    model = LogisticRegressionCV(
    Cs=10,
    penalty='l1',
    solver='liblinear',         # deterministic solver
    scoring='accuracy',
    cv=lasso_cv,
    random_state=SEED,          # ensures any randomness inside is fixed
    n_jobs=1,                   # single‐threaded for determinism
    max_iter=2000
)
    model.fit(X_train, y_train)
    
    y_pred.append(model.predict(X_test)[0])
    y_true.append(y_test[0])
    coefs.append(model.coef_[0])

accuracy = np.mean(np.array(y_true) == np.array(y_pred))
print(f'LOOCV Accuracy: {accuracy:.2f}')

# Average coefficients across folds
avg_coefs = np.mean(np.abs(coefs), axis=0)

plt.figure(figsize=(12,4))
plt.plot(wavenumbers, avg_coefs)
plt.xlabel('Wavenumber Index')
plt.ylabel('Average Absolute Coefficient')
plt.title('LASSO Logistic Regression Variable Importance')

# # -------------------------------
# # 5) Supervised embedding with PLS-DA
# # -------------------------------
pls_components = 2        # you can tune this as needed
y_pred_pls = []
pls_weights = []

# use the same LOOCV splitter you already defined
for train_index, test_index in loo.split(X_res):
    X_train, X_test = X_res[train_index], X_res[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # 1) Fit PLS to (X_train, y_train)
    pls = PLSRegression(n_components=pls_components)
    # PLSRegression expects Y2d, so reshape
    pls.fit(X_train, y_train.reshape(-1, 1))
    
    # 2) Project both train & test
    X_train_scores = pls.transform(X_train)  # shape (n_train, n_components)
    X_test_scores  = pls.transform(X_test)   # shape (1, n_components)
    
    # 3) Fit a simple logistic on the PLS scores
    clf = LogisticRegression(
        solver='liblinear',
        random_state=SEED,
        max_iter=2000
    )
    clf.fit(X_train_scores, y_train)
    
    # 4) Predict
    y_pred_pls.append(clf.predict(X_test_scores)[0])
    
    # 5) record the PLS X-weights (feature importances) for later
    pls_weights.append(pls.x_weights_)  # shape (n_features, n_components)

# compute and report accuracy
accuracy_pls = np.mean(np.array(y_pred_pls) == y)
print(f'LOOCV PLS-DA Accuracy: {accuracy_pls:.2f}')

# average absolute weights over folds
pls_weights = np.array(pls_weights)               # shape (n_folds, n_features, n_components)
avg_abs_weights = np.mean(np.abs(pls_weights), axis=0)  # (n_features, n_components)

# plot the first component’s variable importance
plt.figure(figsize=(12,4))
plt.plot(wavenumbers, avg_abs_weights[:, 0], label='PLS Comp.1')
plt.xlabel('Wavenumber Index')
plt.ylabel('Average |Weight|')
plt.title('PLS-DA Variable Importance (Component 1)')
plt.legend()
plt.show()