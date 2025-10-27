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


def leave_one_patient_out_training(train_dataset, train_patient_groups, n_iterations=100, C=1.0, 
                                   random_seed=42, n_spectra=1, model_type='xgboost',
                                   patient_threshold=0.5):

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
    print(f"\nRunning LOPO CV with {n_iterations} spectrum sampling iterations...")
    
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
        
        # Storage for predictions within this iteration
        iteration_train_accs = []
        
        # Inner loop: Leave-One-Patient-Out
        for val_patient_idx, val_patient_id in enumerate(patient_ids):
            # Define train patients (all except the test patient)
            train_patients_ids = [p for p in patient_ids if p != val_patient_id]
            
            # Get train spectra (from the sampled spectra for this iteration)
            spectrum_index_for_train_set = []
            for patient_id in train_patients_ids:
                spectrum_index_for_train_set.extend(selected_spectra_index[patient_id])
            
            # Get test spectra (from the sampled spectra for this iteration)
            spectrum_index_for_val_set = selected_spectra_index[val_patient_id]
            
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
            
            
            # Train model on all remaining patients
            model, train_spectrum_level_acc, train_spectrum_level_auc, n_feat = train_model(
                X_train, y_train, C=C, model_type=model_type, random_seed=random_seed + iteration
            )
            # Track train metrics
            iteration_train_accs.append(train_spectrum_level_acc)

            # Predict on test set (held-out patient)
            val_spectrum_level_pred = model.predict(X_val)
            val_spectrum_level_pred_proba = model.predict_proba(X_val)[:, 1]

            # make one prediction for the test patient based on majority vote
            cancer_spectrum_fraction = np.mean(val_spectrum_level_pred)
            val_patient_level_pred = 1 if cancer_spectrum_fraction >= patient_threshold else 0
     
            # Store predictions for this iteration (NOT calculating accuracy yet!)
            detailed_prediction.append({
                'iteration': iteration,
                'val_patient': val_patient_id,
                'spectrum_number_threshold_for_cancer_used': patient_threshold,
                'cancer_spectrum_fraction': cancer_spectrum_fraction,
                'true_val_patient_label': patient_labels[val_patient_id],
                'predicted_val_patient_label': val_patient_level_pred,
                'true_label_per_spectrum': y_val,
                'predicted_label_per_spectrum': val_spectrum_level_pred,
                'predicted_proba_per_spectrum': val_spectrum_level_pred_proba,
                'train_spectra': train_set_spectrum_info,
                'val_spectra': val_set_spectrum_info
            })

        # pooled val accuracy and AUC for this iteration (spectrum-level)
        pooled_val_true_label = [p['true_label_per_spectrum'] for p in detailed_prediction]
        pooled_val_pred_label = [p['predicted_label_per_spectrum'] for p in detailed_prediction]
        # Flatten spectrum-level true/pred lists and compute accuracy manually
        flat_true = np.concatenate([np.asarray(t).ravel() for t in pooled_val_true_label]) if len(pooled_val_true_label) else np.array([])
        flat_pred = np.concatenate([np.asarray(t).ravel() for t in pooled_val_pred_label]) if len(pooled_val_pred_label) else np.array([])


        pooled_val_acc_spectrum_level = accuracy_score(flat_true, flat_pred)

        # Calculate patient-level accuracy and AUC for this iteration
        val_patient_level_true_label = [p['true_val_patient_label'] for p in detailed_prediction]
        val_patient_level_pred_label = [p['predicted_val_patient_label'] for p in detailed_prediction]
        val_patient_level_acc = accuracy_score(val_patient_level_true_label, val_patient_level_pred_label)

        # Store iteration-level results
        results.append({
            'model_type': model_type,
            'iteration': iteration,
            'train_accuracy_spectrum': np.mean(iteration_train_accs),
            'val_accuracy_spectrum': pooled_val_acc_spectrum_level,
            'val_accuracy_patient': val_patient_level_acc,
            'selected_spectra_index': selected_spectra_index
        })
        
        # Print progress for this iteration
        print(f"  Completed LOPO for iteration {iteration + 1}")
        print(f"  Average train acc: {np.mean(iteration_train_accs):.3f}")
        print(f"  Val acc (spectrum): {pooled_val_acc_spectrum_level:.3f}")
        print(f"  Val acc (patient): {val_patient_level_acc:.3f}")

    # Create results DataFrame
    results_df = pd.DataFrame(results)
    detailed_predictions_df = pd.DataFrame(detailed_prediction)
    
    # Print overall summary
    print("\n=== LOPO CV Summary ===")
    print(f"Model type: {model_type}")
    print(f"Average train accuracy (spectrum-level): {results_df['train_accuracy_spectrum'].mean():.3f} ± {results_df['train_accuracy_spectrum'].std():.3f}")
    print(f"Average val accuracy (spectrum-level): {results_df['val_accuracy_spectrum'].mean():.3f} ± {results_df['val_accuracy_spectrum'].std():.3f}")
    print(f"Average val accuracy (patient-level): {results_df['val_accuracy_patient'].mean():.3f} ± {results_df['val_accuracy_patient'].std():.3f}")

    return results_df,detailed_predictions_df, raman_shift


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

import matplotlib.pyplot as plt
import numpy as np

def plot_accuracy_histograms(results_df, save_path=None, figsize=(12, 5)):
    """
    Plot histograms of test accuracies at patient and spectrum levels.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame containing the results from LOPO CV
    save_path : str or Path, optional
        If provided, saves the figure to this path
    figsize : tuple, default=(12, 5)
        Figure size (width, height)
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Spectrum-level accuracy histogram
    ax1 = axes[0]
    spectrum_acc = results_df['val_accuracy_spectrum']
    ax1.hist(spectrum_acc, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axvline(spectrum_acc.mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {spectrum_acc.mean():.3f}')
    ax1.axvline(spectrum_acc.median(), color='orange', linestyle='--', linewidth=2,
                label=f'Median: {spectrum_acc.median():.3f}')
    ax1.set_xlabel('Test Accuracy', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Spectrum-Level Test Accuracy Distribution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Add statistics text
    stats_text1 = f'Std: {spectrum_acc.std():.3f}\nMin: {spectrum_acc.min():.3f}\nMax: {spectrum_acc.max():.3f}'
    ax1.text(0.02, 0.98, stats_text1, transform=ax1.transAxes, 
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Patient-level accuracy histogram
    ax2 = axes[1]
    patient_acc = results_df['val_accuracy_patient']
    ax2.hist(patient_acc, bins=20, color='forestgreen', edgecolor='black', alpha=0.7)
    ax2.axvline(patient_acc.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {patient_acc.mean():.3f}')
    ax2.axvline(patient_acc.median(), color='orange', linestyle='--', linewidth=2,
                label=f'Median: {patient_acc.median():.3f}')
    ax2.set_xlabel('Test Accuracy', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Patient-Level Test Accuracy Distribution', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Add statistics text
    stats_text2 = f'Std: {patient_acc.std():.3f}\nMin: {patient_acc.min():.3f}\nMax: {patient_acc.max():.3f}'
    ax2.text(0.02, 0.98, stats_text2, transform=ax2.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()
    
    # Print summary statistics
    print("\n=== Test Accuracy Summary Statistics ===")
    print("\nSpectrum-Level:")
    print(f"  Mean ± Std: {spectrum_acc.mean():.3f} ± {spectrum_acc.std():.3f}")
    print(f"  Median: {spectrum_acc.median():.3f}")
    print(f"  Range: [{spectrum_acc.min():.3f}, {spectrum_acc.max():.3f}]")
    
    print("\nPatient-Level:")
    print(f"  Mean ± Std: {patient_acc.mean():.3f} ± {patient_acc.std():.3f}")
    print(f"  Median: {patient_acc.median():.3f}")
    print(f"  Range: [{patient_acc.min():.3f}, {patient_acc.max():.3f}]")


def filter_and_plot_spectra(results_df, train_dataset, raman_shift, 
                            val_accuracy_threshold=0.8, 
                            metric='val_accuracy_spectrum',
                            max_iterations_to_plot=10,
                            save_path=None,
                            figsize=(15, 10)):
    """
    Filter models by validation accuracy and plot the spectra they were trained on.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame containing LOPO CV results with columns:
        - 'val_accuracy_spectrum' or 'val_accuracy_patient'
        - 'selected_spectra_index': dict mapping patient_id to list of spectrum indices
        - 'iteration': iteration number
    train_dataset : Dataset
        The training dataset containing spectra
    raman_shift : np.array
        Array of Raman shift values (x-axis)
    val_accuracy_threshold : float, default=0.8
        Minimum validation accuracy to filter models
    metric : str, default='val_accuracy_spectrum'
        Metric to use for filtering ('val_accuracy_spectrum' or 'val_accuracy_patient')
    max_iterations_to_plot : int, default=10
        Maximum number of filtered iterations to plot
    save_path : str or Path, optional
        If provided, saves the figure to this path
    figsize : tuple, default=(15, 10)
        Figure size (width, height)
    
    Returns:
    --------
    filtered_df : pd.DataFrame
        DataFrame containing only the filtered results
    """
    
    # Filter results by validation accuracy threshold
    filtered_df = results_df[results_df[metric] >= val_accuracy_threshold].copy()
    
    print(f"\n=== Filtering Results ===")
    print(f"Metric used: {metric}")
    print(f"Threshold: {val_accuracy_threshold:.3f}")
    print(f"Total iterations: {len(results_df)}")
    print(f"Filtered iterations (above threshold): {len(filtered_df)}")
    print(f"Percentage: {100 * len(filtered_df) / len(results_df):.1f}%")
    
    if len(filtered_df) == 0:
        print("No models meet the threshold criteria!")
        return filtered_df
    
    # Print statistics of filtered models
    print(f"\nFiltered model statistics:")
    print(f"  Mean {metric}: {filtered_df[metric].mean():.3f} ± {filtered_df[metric].std():.3f}")
    print(f"  Range: [{filtered_df[metric].min():.3f}, {filtered_df[metric].max():.3f}]")
    
    # Limit number of iterations to plot
    n_to_plot = min(len(filtered_df), max_iterations_to_plot)
    filtered_df_to_plot = filtered_df.head(n_to_plot)
    
    print(f"\nPlotting spectra from {n_to_plot} iterations...")
    
    # Create figure with subplots
    n_rows = (n_to_plot + 1) // 2  # 2 columns
    n_cols = 2
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.3)
    
    # Plot each filtered iteration
    for plot_idx, (_, row) in enumerate(filtered_df_to_plot.iterrows()):
        ax = fig.add_subplot(gs[plot_idx // n_cols, plot_idx % n_cols])
        
        iteration_num = row['iteration']
        accuracy = row[metric]
        selected_spectra_index = row['selected_spectra_index']
        
        # Collect all spectra indices from this iteration
        all_spectrum_indices = []
        for patient_id, indices in selected_spectra_index.items():
            all_spectrum_indices.extend(indices)
        
        # Plot each spectrum
        for idx in all_spectrum_indices:
            intensity, _, metadata = train_dataset[idx]
            intensity_np = intensity.squeeze().numpy()
            
            # Color by label
            staging = metadata['staging'].lower()
            label = 1 if ('cancer' in staging or 'ca' in staging) else 0
            color = 'red' if label == 1 else 'blue'
            alpha = 0.3
            
            ax.plot(raman_shift, intensity_np, color=color, alpha=alpha, linewidth=0.8)
        
        # Formatting
        ax.set_xlabel('Raman Shift (cm⁻¹)', fontsize=9)
        ax.set_ylabel('Intensity (a.u.)', fontsize=9)
        ax.set_title(f'Iteration {iteration_num} | Val Acc: {accuracy:.3f}', 
                    fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
        
        # Add legend for first plot only
        if plot_idx == 0:
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='red', lw=2, label='Cancer'),
                Line2D([0], [0], color='blue', lw=2, label='Control')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    # Add overall title
    fig.suptitle(f'Training Spectra for Models with {metric} ≥ {val_accuracy_threshold:.3f}',
                fontsize=14, fontweight='bold', y=0.995)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved plot to {save_path}")
    
    plt.show()
    
    return filtered_df

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

    # track = []
    # for patient_id, spectra_indices in train_patient_groups.items():
    #     track.append(len(spectra_indices))
    #     print(f"Patient ID: {patient_id}, Number of Spectra: {len(spectra_indices)}")

    # print(f"\nMin spectra per patient: {min(track)}")
    # print(f"Max spectra per patient: {max(track)}")


    for model_type in ['logistic_regression', 'xgboost']:

        for n_spectra in [12]:

            results_df, detailed_predictions_df, raman_shift = leave_one_patient_out_training(
                train_dataset=train_dataset,
                train_patient_groups=train_patient_groups,
                n_iterations=1000,
                C=10,
                random_seed=42,
                n_spectra=n_spectra,
                patient_threshold=0.5,
                model_type=model_type
            )

            # Save results
            save_lopo_results(results_df,
                detailed_predictions_df, 
                raman_shift,
                save_dir=r'D:\ev_data_balanced',
                file_name=f'both_bec_and_izabella_cleaned2_{model_type}_selected_{n_spectra}_spectra'
            )

            # # Load results
            # results_df, detailed_predictions_df, raman_shift = load_lopo_results(
            #     file_path=r'D:\ev_data_balanced\both_bec_and_izabella_cleaned2_logistic_regression_selected_12_spectra.pkl'
            # )

            # # Plot accuracy histograms
            # plot_accuracy_histograms(results_df, r"D:\ev_data_balanced\both_bec_and_izabella_cleaned2_logistic_regression_selected_12_spectra_histograms.png")


            # # Filter and plot individual spectra
            # filtered_df = filter_and_plot_spectra(
            #     results_df=results_df,
            #     train_dataset=train_dataset,
            #     raman_shift=raman_shift,
            #     val_accuracy_threshold=0.6,  # Adjust threshold as needed
            #     metric='val_accuracy_spectrum',  # or 'val_accuracy_patient'
            #     max_iterations_to_plot=10,
            #     save_path=r'D:\ev_data_balanced\both_bec_and_izabella_cleaned2_logistic_regression_selected_12_spectra_high_accuracy_spectra.png'
            # )