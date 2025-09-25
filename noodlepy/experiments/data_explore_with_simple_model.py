import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from scipy.stats import ttest_ind
from sklearn.cross_decomposition import PLSRegression
import matplotlib.pyplot as plt
from noodlepy.utils.bec_hnc_dataset import Bec_HNC_Dataset
from noodlepy.utils.spectrumpreprocessor import SpectrumPreprocessor
from noodlepy.utils.spectrumaugmentor import SpectrumAugmentor
import os
import scipy.stats as stats
import seaborn as sns

# -------------------------------
# PCA for exploratory visualization
# -------------------------------
data_folder = os.path.join(os.getcwd(), "noodlepy", "data", "bec_hnc_temp")
metadata_file = os.path.join(os.getcwd(), "noodlepy", "data", "bec_hnc_patient_annotations.xlsx")

preprocessor = SpectrumPreprocessor(
    cropping=False,
    baseline_correction=True,
    remove_cosmic_rays=True,
    normalization=True,
    smoothing=True
)

augmentor = SpectrumAugmentor(
    ramdom_augmentations=False,
    augmentation_step_list=None,
    config_path=None
)

r_filter = None

dataset = Bec_HNC_Dataset(
    data_folder,
    metadata_file,
    r_filter,
    preprocessor,
    augmentor
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
X = np.array([np.mean(patient_data[pid]["spectra"], axis=0)[5:745] for pid in patient_ids])
y = np.array([patient_data[pid]["label"] for pid in patient_ids])
dates = pd.to_datetime([patient_data[pid]["date"] for pid in patient_ids])
wavenumbers = np.array(dataset.db[0].wavelength_nm)[5:745]


# Bin X by every 149 wavenumbers
bin_size = 10
num_bins = X.shape[1] // bin_size
X_binned = np.array([np.mean(X[:, i*bin_size:(i+1)*bin_size], axis=1) for i in range(num_bins)]).T
wavenumbers_binned = np.array([np.mean(wavenumbers[i*bin_size:(i+1)*bin_size]) for i in range(num_bins)])

X = X_binned
wavenumbers = wavenumbers_binned

#######################################
# 1) Standardize
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

# 2) Fit PCA with 1 component to get PC1
# Perform PCA with 2 components before any denoising
pca_initial = PCA(n_components=2).fit(X_scaled)
X_pca_initial = pca_initial.transform(X_scaled)

# Plot PC1 vs PC2 before denoising
plt.figure(figsize=(7, 5))
plt.scatter(X_pca_initial[y == 0, 0], X_pca_initial[y == 0, 1], label='Healthy', c='blue')
plt.scatter(X_pca_initial[y == 1, 0], X_pca_initial[y == 1, 1], label='Cancer', c='red')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA before Denoising')
plt.legend()

# Perform PCA with 1 component for further analysis
pca1 = PCA(n_components=1).fit(X_scaled)
scores1 = pca1.transform(X_scaled).ravel()   # shape (N,)
loadings1 = pca1.components_[0]              # shape (n_wavenumbers,)

# 3) Find top-10 wavenumbers driving PC1
top10_idx = np.argsort(np.abs(loadings1))[-10:]
print("Top-10 wavenumbers:", wavenumbers[top10_idx])

# 4) Convert dates to numeric (seconds since first date)
dates_dt  = pd.to_datetime(dates)
date_nums = ((dates_dt - dates_dt.min()) / np.timedelta64(1, 's')).astype(float)


# 5) For each of those top-10, regress out its date trend
X_denoised = X_scaled.copy()
threshold = 0.4

for idx in top10_idx:
    r = np.corrcoef(X_scaled[:, idx], date_nums)[0,1]
    print(f"{wavenumbers[idx]:.1f} cm⁻¹ corr with date: {r:.2f}")
    if abs(r) > threshold:
        a, b    = np.polyfit(date_nums, X_scaled[:, idx], 1)
        trend   = a * date_nums + b
        X_denoised[:, idx] = X_scaled[:, idx] - trend

# 6) Recompute PCA with 2 components on the cleaned data
pca2 = PCA(n_components=2).fit(X_denoised)
X_pca2 = pca2.transform(X_denoised)

# 7) Plot PC1 vs PC2
plt.figure(figsize=(7,5))
plt.scatter(X_pca2[y==0,0], X_pca2[y==0,1], label='Healthy', c='blue')
plt.scatter(X_pca2[y==1,0], X_pca2[y==1,1], label='Cancer', c='red')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA after date-trend removal')
plt.legend()
plt.show()

# X_scaled = X_denoised


# # Standardize features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # -------------------------------
# # 2) PCA for exploratory visualization
# # -------------------------------

# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X_scaled)

# plt.figure(figsize=(8,6))
# plt.scatter(X_pca[y==0, 0], X_pca[y==0, 1], label='Healthy', c='blue')
# plt.scatter(X_pca[y==1, 0], X_pca[y==1, 1], label='Cancer', c='red')
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.title('PCA of Averaged Patient Spectra')
# plt.legend()
# plt.show()

# # -------------------------------
# # 3) Identify dominant noisy wavenumbers via PCA loadings
# # -------------------------------
# loadings = pca.components_[0]
# top10_idx = np.argsort(np.abs(loadings))[-10:]
# top10_wavenumbers = wavenumbers[top10_idx]
# print("Top 10 wavenumbers driving PC1:", top10_wavenumbers)

# for wn, idx in zip(top10_wavenumbers, top10_idx):
#     plt.figure(figsize=(6,4))
#     unique_dates = sorted(dates.unique())
#     date_indices = [unique_dates.index(date) for date in dates if date in unique_dates]
#     for xi, date_idx, lbl in zip(X[:, idx], date_indices, y):
#         plt.scatter(date_idx, xi, c='red' if lbl==1 else 'blue')
#     plt.xticks(ticks=range(len(unique_dates)), labels=[date.strftime('%Y-%m-%d') for date in unique_dates], rotation=45)
#     plt.title(f"Intensity at {wn} cm⁻¹ by Date & Label")
#     plt.xlabel('Measurement Date')
#     plt.ylabel('Raw Intensity')
#     plt.tight_layout()
#     plt.show()

# -------------------------------
# 4) Physically inspect the top 10 wavenumbers
# -------------------------------

# healthy_idx = np.where(y==0)[0][:2]
# cancer_idx = np.where(y==1)[0][:2]
# sel_idx = np.concatenate([healthy_idx, cancer_idx])

# window = 10
# for wn_idx, wn in zip(top10_idx, top10_wavenumbers):
#     # define the slice
#     start = max(0, wn_idx - window)
#     end = min(X.shape[1], wn_idx + window)
#     xs = wavenumbers[start:end]  # use wavenumbers for x-axis
    
#     plt.figure(figsize=(6, 4))
#     for i in sel_idx:
#         lbl = 'Cancer' if y[i] == 1 else 'Healthy'
#         plt.plot(xs, X[i, start:end],
#                  alpha=0.6,
#                  label=f'PID {patient_ids[i]} ({lbl})')
#     # mark the exact peak
#     plt.axvline(wn, color='k', linestyle='--')
#     plt.title(f'Zoom around {wn:.1f} cm⁻¹')
#     plt.xlabel('Wavenumber (cm⁻¹)')
#     plt.ylabel('Intensity')
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.tight_layout()
#     plt.show()

# -------------------------------
# 5) check the correlation of the top 10 wavenumbers with date and label
# -------------------------------

# for wn in top10_wavenumbers:
#     idx = np.where(wavenumbers == wn)[0][0]
#     scores = X[:, idx]
#     date_nums = dates.astype('int64') // 10**9
#     r_date, p_date = stats.pearsonr(scores, date_nums)
#     r_label, p_label = stats.pointbiserialr(scores, y)
#     print(f'{wn} cm⁻¹: r(date)={r_date:.2f}, r(label)={r_label:.2f}')






# -------------------------------
# Univariate t-test across wavenumbers
# -------------------------------

t_stats, p_values = ttest_ind(X_scaled[y==0], X_scaled[y==1])

plt.figure(figsize=(12,4))
plt.plot(wavenumbers, p_values)
plt.xlabel('Wavenumber Index')
plt.ylabel('p-value')
plt.title('Univariate t-test p-values for each Wavenumber')
plt.axhline(0.05, color='r', linestyle='--', label='p = 0.05')
plt.legend()
plt.show()

# -------------------------------
# LASSO Logistic Regression with LOOCV
# -------------------------------

loo = LeaveOneOut()
y_true = []
y_pred = []
coefs = []

for train_index, test_index in loo.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    model = LogisticRegressionCV(Cs=10, penalty='l1', solver='liblinear', scoring='accuracy', cv=5, max_iter=1000)
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
plt.show()
print(f'Done!')

# -------------------------------
# 5) Supervised embedding with PLS-DA
# -------------------------------

pls = PLSRegression(n_components=2)
X_pls, _ = pls.fit_transform(X_scaled, y)

plt.figure(figsize=(8,6))
plt.scatter(X_pls[y==0, 0], X_pls[y==0, 1], label='Healthy', c='blue')
plt.scatter(X_pls[y==1, 0], X_pls[y==1, 1], label='Cancer', c='red')
plt.xlabel('PLS Component 1')
plt.ylabel('PLS Component 2')
plt.title('PLS-DA of Raman Spectra')
plt.legend()
plt.show()

# -------------------------------
# 6) Hierarchical clustering + Heatmap
# -------------------------------

variances = np.var(X_scaled, axis=0)
top50_idx = np.argsort(variances)[-50:]
data_top50 = X_scaled[:, top50_idx]
cols = wavenumbers[top50_idx]

df_heat = pd.DataFrame(data_top50, index=patient_ids, columns=cols.astype(str))
row_colors = pd.DataFrame({'Label': ['red' if lbl==1 else 'blue' for lbl in y]}, index=df_heat.index)

sns.clustermap(df_heat, row_colors=row_colors, cmap='viridis', figsize=(10,10))
plt.title('Hierarchical Clustering of Top-Variance Wavenumbers')
plt.show()
