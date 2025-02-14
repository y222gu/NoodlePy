import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import tensorflow as tf
from sklearn.impute import KNNImputer
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# KNN Imputation across the spatial grid
def knn_grid_impute(patient_data, n_neighbors=5):
    imputed_patient = np.empty_like(patient_data)
    for i in range(5):  # Loop through the 5 optical properties
        feature_grid = patient_data[i, :, :]
        feature_flat = feature_grid.reshape(-1, 1)  # Flatten for KNNImputer
        imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance')
        imputed_flat = imputer.fit_transform(feature_flat)
        imputed_grid = imputed_flat.reshape(24, 8)
        imputed_patient[i, :, :] = imputed_grid
    return imputed_patient

# load all patient data and labels
folder_path = os.path.join(os.getcwd(), 'noodlepy', 'data', 'bec_feature_reduced_data')
file_lst = os.listdir(folder_path)



data = []
labels = []
for file_name in file_lst:
    file_path = os.path.join(folder_path, file_name)
    data_patient_i = pd.read_csv(file_path, header=None, na_values=['NaN', 'nan'], float_precision='high')
    data_patient_i = data_patient_i.to_numpy()

    # Reshape to (5, 24, 8)
    data_patient_i = data_patient_i.reshape(5, 24, 8)
    print('the first region of the patient is:')
    print(data_patient_i[0, :, :])

    # extrac the label from the file name
    label_patient_i_str = os.path.basename(file_name).split('_')[2]
    if label_patient_i_str == 'Control':
        label_patient_i = 0
    else:
        label_patient_i = 1

    print(f'the label of the patient is {label_patient_i}')
    data_patient_i = np.array(data_patient_i)

    data.append(data_patient_i)
    labels.append(label_patient_i)

# Apply KNN imputation for each patient
imputed_data = np.array([knn_grid_impute(patient) for patient in data])

# Flatten the data for model training
data_flattened = imputed_data.reshape(imputed_data.shape[0], -1)

# Feature Engineering
features = pd.DataFrame(data_flattened)
# features['mean'] = features.mean(axis=1)
# features['std'] = features.std(axis=1)
# features['median'] = features.median(axis=1)

features.columns = features.columns.astype(str)

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Standardizing the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)

# SVM Classifier
svm = SVC(kernel='linear')
svm.fit(X_train_scaled, y_train)
y_pred_svm = svm.predict(X_test_scaled)

# Printing classification report for Random Forest
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

# Printing classification report for SVM
print("SVM Classification Report:")
print(classification_report(y_test, y_pred_svm))


# CNN Model
# Prepare data for PyTorch
X_tensor = torch.tensor(imputed_data, dtype=torch.float32).unsqueeze(1)  # Adding channel dimension
y_tensor = torch.tensor(labels, dtype=torch.float32)

X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4)

# Define CNN Model in PyTorch
class CNN3D(nn.Module):
    def __init__(self):
        super(CNN3D, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=(3, 3, 2))
        self.pool = nn.MaxPool3d(2)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(2, 2, 2))
        self.fc1 = nn.Linear(64 * 5 * 5 * 2, 64)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# Initialize model, loss, and optimizer
model = CNN3D()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
for epoch in range(20):
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs).squeeze()
        preds = (outputs > 0.5).float()
        correct += (preds == targets).sum().item()
        total += targets.size(0)

print(f"CNN Accuracy: {100 * correct / total:.2f}%")

# Feature Importance for RF
importances = rf.feature_importances_
print("Top PCA Component Importances:", importances[:5])