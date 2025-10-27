import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from collections import defaultdict
import pandas as pd
import random
import matplotlib.pyplot as plt
from noodlepy.utils.izabelladataset import OC_Dataset
from noodlepy.utils.spectrumpreprocessor import SpectrumPreprocessor
from matplotlib.gridspec import GridSpec
import pickle
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import pickle
import os
from pathlib import Path


def get_patient_groups(dataset):
    """Group spectrum indices by patient_id."""
    groups = defaultdict(list)
    for idx in range(len(dataset)):
        _, _, metadata = dataset[idx]  # Updated to unpack 3 values
        groups[metadata['patient_id']].append(idx)
    return groups


def prepare_data(dataset, indices):
    """Convert spectrum indices to X, y arrays and get Raman shift."""
    X, y, patient_ids = [], [], []
    raman_shift = None
    
    for idx in indices:
        intensity, raman_shift_tensor, metadata = dataset[idx]  # Updated to unpack 3 values
        X.append(intensity.squeeze().numpy())  # Convert tensor to numpy
        
        # Store raman shift (should be same for all spectra)
        if raman_shift is None:
            raman_shift = raman_shift_tensor.squeeze().numpy()
        
        # Label: 1 for cancer, 0 for control
        staging = metadata['staging'].lower()
        label = 1 if ('cancer' in staging or 'ca' in staging) else 0
        
        y.append(label)
        patient_ids.append(metadata['patient_id'])
    
    return np.array(X), np.array(y), patient_ids, raman_shift


def train_model(X_train, y_train, C=1.0, model_type='xgboost', random_seed=42):
    """
    Train model on training data only.
    No test set evaluation here - that's done separately in LOPO.
    """
    
    if model_type == 'xgboost':
        model = xgb.XGBClassifier(
            max_depth=3,
            learning_rate=0.1,
            n_estimators=50,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            eval_metric='logloss',
            random_state=random_seed
        )
        
    elif model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=3,              # Very shallow
            min_samples_split=3,     # Need 20 samples to split
            min_samples_leaf=1,      # Min 10 samples per leaf
            max_features='sqrt',      # ~27 features per tree
            n_jobs=-1,
            random_state=random_seed
        )
        
    elif model_type == 'decision_tree':
        model = DecisionTreeClassifier(
            max_depth=3,              # Very shallow
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=random_seed
        )
    elif model_type == 'logistic_regression':
        model = LogisticRegression(penalty='l1', solver='liblinear', C=C, random_state=random_seed)

    elif model_type == 'poly_svm':
        model = SVC(
            kernel='linear', 
            C=C, 
            probability=True,  # Enable probability estimates
            max_iter=-1,  # No maximum iteration limit
            random_state=random_seed
        )
    
    model.fit(X_train, y_train)
    
    # Calculate train accuracy
    train_acc = accuracy_score(y_train, model.predict(X_train))
    train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])

    n_features = np.sum(model.coef_ != 0) if hasattr(model, 'coef_') else X_train.shape[1]

    return model, train_acc, train_auc, n_features


def kfold_patient_training(train_dataset, train_patient_groups, n_iterations=100, C=1.0, 
                           random_seed=42, n_spectra=1, model_type='xgboost',
                           patient_threshold=0.5, n_folds=5):
    """
    Train model with K-fold cross-validation at the patient level.
    
    Args:
        train_dataset: Dataset containing spectra
        train_patient_groups: Dictionary mapping patient IDs to their spectrum indices
        n_iterations: Number of random spectrum sampling iterations
        C: Regularization parameter
        random_seed: Random seed for reproducibility
        n_spectra: Number of spectra to sample per patient
        model_type: Type of model to train (e.g., 'xgboost')
        patient_threshold: Threshold for patient-level classification
        n_folds: Number of folds for cross-validation (default: 5)
    
    Returns:
        results_df: DataFrame with iteration-level results
        detailed_predictions_df: DataFrame with patient-level predictions
        raman_shift: Raman shift values
    """

    # Get all patient IDs and their labels
    patient_ids = list(train_patient_groups.keys())
    patient_labels = {}
    
    for pid in patient_ids:
        _, _, metadata = train_dataset[train_patient_groups[pid][0]]
        staging = metadata['staging'].lower()
        patient_labels[pid] = 1 if ('cancer' in staging or 'ca' in staging) else 0
    
    print(f"Total patients: {len(patient_ids)}")
    print(f"  Cancer patients: {sum(patient_labels.values())}")
    print(f"  Control patients: {len(patient_ids) - sum(patient_labels.values())}")
    print(f"  Patient classification threshold: {patient_threshold:.2%}")
    print(f"\nRunning {n_folds}-Fold CV with {n_iterations} spectrum sampling iterations...")
    
    results = []
    detailed_prediction = []
    raman_shift = None
    
    # Outer loop: Different random spectrum samplings
    for iteration in range(n_iterations):
        print(f"\n=== Iteration {iteration + 1}/{n_iterations} ===")
        
        # Sample spectra for ALL patients once per iteration
        selected_spectra_index = {}
        for pid in patient_ids:
            available = train_patient_groups[pid]
            if len(available) >= n_spectra:
                selected = random.sample(available, n_spectra)
            else:
                selected = random.choices(available, k=n_spectra)
            selected_spectra_index[pid] = selected
        
        # Create stratified K-fold splits for patients
        # Separate cancer and control patients
        cancer_patients = [pid for pid in patient_ids if patient_labels[pid] == 1]
        control_patients = [pid for pid in patient_ids if patient_labels[pid] == 0]
        
        # Shuffle patients with iteration-specific seed
        rng = random.Random(random_seed + iteration)
        rng.shuffle(cancer_patients)
        rng.shuffle(control_patients)
        
        # Split each group into n_folds
        cancer_folds = [cancer_patients[i::n_folds] for i in range(n_folds)]
        control_folds = [control_patients[i::n_folds] for i in range(n_folds)]
        
        # Combine to create stratified folds
        patient_folds = [cancer_folds[i] + control_folds[i] for i in range(n_folds)]
        
        # Storage for predictions within this iteration
        k_folds_train_accs_spectrum_level = []
        k_folds_train_accs_patient_level = []
        k_folds_val_accs_patient_level = []
        k_folds_val_accs_spectrum_level = []
        # Inner loop: K-Fold Cross-Validation
        for fold_idx in range(n_folds):
            # Define validation patients (current fold)
            val_patients_ids = patient_folds[fold_idx]
            
            # Define train patients (all other folds)
            train_patients_ids = []
            for i in range(n_folds):
                if i != fold_idx:
                    train_patients_ids.extend(patient_folds[i])
            
            # Get train spectra (from the sampled spectra for this iteration)
            spectrum_index_for_train_set = []
            for patient_id in train_patients_ids:
                spectrum_index_for_train_set.extend(selected_spectra_index[patient_id])
            
            # Get validation spectra (from the sampled spectra for this iteration)
            spectrum_index_for_val_set = []
            for patient_id in val_patients_ids:
                spectrum_index_for_val_set.extend(selected_spectra_index[patient_id])
            
            # Prepare data
            X_train, y_train, _, current_raman_shift = prepare_data(train_dataset, spectrum_index_for_train_set)
            X_val, y_val, _, _ = prepare_data(train_dataset, spectrum_index_for_val_set)
            
            if raman_shift is None:
                raman_shift = current_raman_shift
            
            # Record spectrum selection info for this fold
            train_set_spectrum_info = []
            for idx in spectrum_index_for_train_set:
                _, _, metadata = train_dataset[idx]
                train_set_spectrum_info.append({
                    'patient_id': metadata['patient_id'],
                    'spectrum_id': metadata['spectrum_id'],
                    'dataset_index': idx
                })
            
            val_set_spectrum_info = []
            for idx in spectrum_index_for_val_set:
                _, _, metadata = train_dataset[idx]
                val_set_spectrum_info.append({
                    'patient_id': metadata['patient_id'],
                    'spectrum_id': metadata['spectrum_id'],
                    'dataset_index': idx
                })
            
            # Train model on train folds
            model, train_spectrum_level_acc, train_auc, n_feat = train_model(
                X_train, y_train, C=C, model_type=model_type, random_seed=random_seed + iteration
            )

            # calculate train accuracy at patient level
            train_spectrum_level_pred = model.predict(X_train)
            train_patient_level_pred = []
            for train_patient_id in train_patients_ids:
                # Get predictions for this specific patient's spectra
                patient_spectrum_indices = [i for i, idx in enumerate(spectrum_index_for_train_set) 
                                           if idx in selected_spectra_index[train_patient_id]]
                
                specific_patient_spectrum_preds = train_spectrum_level_pred[patient_spectrum_indices]
                
                # Make one prediction for the patient based on majority vote
                cancer_spectrum_fraction = np.mean(specific_patient_spectrum_preds)
                specific_patient_patient_pred = 1 if cancer_spectrum_fraction >= patient_threshold else 0
                train_patient_level_pred.append(specific_patient_patient_pred)
            train_patient_level_acc = accuracy_score(
                [patient_labels[pid] for pid in train_patients_ids], train_patient_level_pred
            )

            # Track train metrics
            k_folds_train_accs_spectrum_level.append(train_spectrum_level_acc)
            k_folds_train_accs_patient_level.append(train_patient_level_acc)

            # Predict each spectrum in the validation fold
            val_spectrum_level_pred = model.predict(X_val)
            val_spectrum_level_pred_proba = model.predict_proba(X_val)[:, 1]
            val_spectrum_level_acc = accuracy_score(y_val, val_spectrum_level_pred)
            k_folds_val_accs_spectrum_level.append(val_spectrum_level_acc)

            # Predict at patient level for the validation fold
            val_patient_level_pred = []
            for val_patient_id in val_patients_ids:
                # Get predictions for this specific patient's spectra
                patient_spectrum_indices = [i for i, idx in enumerate(spectrum_index_for_val_set) 
                                           if idx in selected_spectra_index[val_patient_id]]
                
                specific_patient_spectrum_preds = val_spectrum_level_pred[patient_spectrum_indices]
                
                # Make one prediction for the patient based on majority vote
                cancer_spectrum_fraction = np.mean(specific_patient_spectrum_preds)
                specific_patient_patient_pred = 1 if cancer_spectrum_fraction >= patient_threshold else 0
                val_patient_level_pred.append(specific_patient_patient_pred)

            val_patient_level_acc = accuracy_score(
                [patient_labels[pid] for pid in val_patients_ids], val_patient_level_pred
            )
            k_folds_val_accs_patient_level.append(val_patient_level_acc)

            # Store predictions for this fold
            detailed_prediction.append({
                'iteration': iteration,
                'fold': fold_idx,
                'train_spectra': train_set_spectrum_info,
                'val_spectra': val_set_spectrum_info,
                'true_val_spectrum_label': y_val,
                'predicted_val_spectrum_label': val_spectrum_level_pred,
                'predicted_val_spectrum_proba': val_spectrum_level_pred_proba,
                'accuracy_val_spectrum_level': val_spectrum_level_acc,
                'true_val_patient_label':  [patient_labels[pid] for pid in val_patients_ids],
                'predicted_val_patient_label': val_patient_level_pred,
                'accuracy_val_patient_level': val_patient_level_acc,
                'n_features_used': n_feat,
            })

        average_val_acc_spectrum_level_this_iteration = np.mean(k_folds_val_accs_spectrum_level)
        average_val_acc_patient_level_this_iteration = np.mean(k_folds_val_accs_patient_level)
        average_train_acc_spectrum_level_this_iteration = np.mean(k_folds_train_accs_spectrum_level)
        average_train_acc_patient_level_this_iteration = np.mean(k_folds_train_accs_patient_level)

        # Store iteration-level results
        results.append({
            'model_type': model_type,
            'iteration': iteration,
            'k_folds_train_accuracies_spectrum_level': k_folds_train_accs_spectrum_level,
            'k_folds_train_accuracies_patient_level': k_folds_train_accs_patient_level,
            'k_folds_val_accuracies_spectrum_level': k_folds_val_accs_spectrum_level,
            'k_folds_val_accuracies_patient_level': k_folds_val_accs_patient_level,
            'average_train_accuracy_spectrum': average_train_acc_spectrum_level_this_iteration,
            'average_train_accuracy_patient': average_train_acc_patient_level_this_iteration,
            'average_val_accuracy_spectrum': average_val_acc_spectrum_level_this_iteration,
            'average_val_accuracy_patient': average_val_acc_patient_level_this_iteration,
            'selected_spectra_index': selected_spectra_index
        })
        
        # Print progress for this iteration
        print(f"  Completed {n_folds}-fold CV for iteration {iteration + 1}")
        print(f"    Average train accuracy (spectrum-level): {average_train_acc_spectrum_level_this_iteration:.3f}")
        print(f"    Average train accuracy (patient-level): {average_train_acc_patient_level_this_iteration:.3f}")
        print(f"    Average val accuracy (spectrum-level): {average_val_acc_spectrum_level_this_iteration:.3f}")
        print(f"    Average val accuracy (patient-level): {average_val_acc_patient_level_this_iteration:.3f}")

    # Create results DataFrame
    results_df = pd.DataFrame(results)
    detailed_predictions_df = pd.DataFrame(detailed_prediction)
    
    # Print overall summary
    print(f"\n=== {n_folds}-Fold CV Summary ===")
    print(f"Total iterations: {n_iterations}")
    print(f"Overall average val accuracy (spectrum-level): {results_df['average_val_accuracy_spectrum'].mean():.3f} ± {results_df['average_val_accuracy_spectrum'].std():.3f}")
    print(f"Overall average val accuracy (patient-level): {results_df['average_val_accuracy_patient'].mean():.3f} ± {results_df['average_val_accuracy_patient'].std():.3f}")

    return results_df, detailed_predictions_df, raman_shift


def save_lopo_results(results_df, detailed_predictions_df, raman_shift, 
                      save_dir, file_name):

    # Create save directory if it doesn't exist
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save all data in a single file (convenient for loading everything at once)
    results = {
        'results_df': results_df,
        'detailed_predictions_df': detailed_predictions_df,
        'raman_shift': raman_shift
    }

    # Save all data in a single file (convenient for loading everything at once)
    all_data_path = save_dir / f'{file_name}.pkl'
    with open(all_data_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved all data to {all_data_path}")

def load_lopo_results(file_path):
    
    # Load results data
    with open(file_path, 'rb') as f:
        results = pickle.load(f)
    print(f"Loaded results from {file_path}")
    
    # Extract components
    results_df = results['results_df']
    detailed_predictions_df = results['detailed_predictions_df']
    raman_shift = results['raman_shift']
    
    # Print summary
    print(f"  - Results shape: {results_df.shape}")
    print(f"  - Detailed predictions shape: {detailed_predictions_df.shape}")
    
    return results_df, detailed_predictions_df, raman_shift


# Example usage
if __name__ == "__main__":

    # 1. Load your dataset
    train_data_folder = r'D:\ev_data_balanced\train_cleaned2'

    preprocessor = SpectrumPreprocessor(
        cropping=True,
        baseline_correction=True,
        remove_cosmic_rays=False,
        normalization=True,
        smoothing=True)
    
    train_dataset = OC_Dataset(train_data_folder, preprocessor, augmentor=None)
    train_patient_groups = get_patient_groups(train_dataset)

    for model_type in ['logistic_regression', 'xgboost']:

        for n_spectra in [12]:

            results_df, detailed_predictions_df, raman_shift = kfold_patient_training(
                train_dataset=train_dataset,
                train_patient_groups=train_patient_groups,
                n_iterations=5000,
                C=15,
                random_seed=42,
                n_spectra=n_spectra,
                patient_threshold=0.5,
                model_type=model_type,
                n_folds=5
            )

            # Save results
            save_lopo_results(results_df,
                detailed_predictions_df, 
                raman_shift,
                save_dir=r'D:\ev_data_balanced',
                file_name=f'both_bec_and_izabella_cleaned2_kfold_{model_type}_selected_{n_spectra}_spectra'
            )

            # # Load results
            # results_df, detailed_predictions_df, raman_shift = load_lopo_results(
            #     file_path=r'D:\ev_data_balanced\both_bec_and_izabella_cleaned2_logistic_regression_selected_12_spectra.pkl'
            # )
