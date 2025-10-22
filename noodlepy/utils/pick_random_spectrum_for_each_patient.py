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


def train_model(X_train, y_train, X_test, y_test, C=1.0, model_type='xgboost'):
    """Optimized for high-dimensional, low-sample data."""
    
    if model_type == 'xgboost':
        model = xgb.XGBClassifier(
            max_depth=3,              # Shallow trees
            learning_rate=0.05,       # Slower learning
            n_estimators=100,
            min_child_weight=10,      # Require more samples per leaf
            subsample=0.8,            # Use 80% of samples per tree
            colsample_bytree=0.3,     # Use 30% of features per tree
            reg_alpha=1.0,            # L1 regularization
            reg_lambda=1.0,           # L2 regularization
            random_state=42
        )
        
    elif model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=4,              # Very shallow
            min_samples_split=20,     # Need 20 samples to split
            min_samples_leaf=10,      # Min 10 samples per leaf
            max_features='sqrt',      # ~27 features per tree
            random_state=42,
            n_jobs=-1
        )
        
    elif model_type == 'decision_tree':
        model = DecisionTreeClassifier(
            max_depth=3,              # Very shallow
            min_samples_split=30,
            min_samples_leaf=15,
            random_state=42
        )
    elif model_type == 'logistic_regression':
        model = LogisticRegression(penalty='l1', solver='liblinear', C=C, random_state=42)

    elif model_type == 'poly_svm':
        model = SVC(
            kernel='linear', 
            C=C, 
            random_state=42, 
            max_iter=-1  # No maximum iteration limit
        )
    
    model.fit(X_train, y_train)
    
    test_acc = accuracy_score(y_test, model.predict(X_test))
    test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    
    n_features = np.sum(model.coef_ != 0) if hasattr(model, 'coef_') else X_train.shape[1]

    return model, test_acc, test_auc, n_features


def random_subset_training(dataset, patient_groups, n_iterations=100, C=1.0, test_size=0.2, random_seed=42, n_spectra=1, model_type='xgboost'):
    """
    Train with FIXED patient split and random spectrum sampling.
    Instead of one spectrum per patient, sample n_spectra per patient.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Split patients into train/test (FIXED)
    patient_ids = list(patient_groups.keys())
    patient_labels = []
    
    for pid in patient_ids:
        _, _, metadata = dataset[patient_groups[pid][0]]
        staging = metadata['staging'].lower()
        patient_labels.append(1 if ('cancer' in staging or 'ca' in staging) else 0)

    train_patients, test_patients, train_labels, test_labels = train_test_split(
        patient_ids, patient_labels, 
        test_size=test_size, 
        random_state=random_seed, 
        stratify=patient_labels
    )
    
    train_cancer = sum(train_labels)
    train_control = len(train_labels) - train_cancer
    test_cancer = sum(test_labels)
    test_control = len(test_labels) - test_cancer
    
    print(f"Patient split: {len(train_patients)} train, {len(test_patients)} test")
    print(f"  Train set: {train_cancer} cancer, {train_control} control")
    print(f"  Test set:  {test_cancer} cancer, {test_control} control")
    print(f"\nTraining {n_iterations} models with different spectrum samples...")

    results = []
    models = []
    spectrum_selections = []
    raman_shift = None

    for i in range(n_iterations):
        train_subset = []
        for p in train_patients:
            available = patient_groups[p]
            # Sample without replacement if possible
            if len(available) >= n_spectra:
                selected = random.sample(available, n_spectra)
            else:
                selected = random.choices(available, k=n_spectra)
            train_subset.extend(selected)
            
        test_subset = []
        for p in test_patients:
            available = patient_groups[p]
            if len(available) >= n_spectra:
                selected = random.sample(available, n_spectra)
            else:
                selected = random.choices(available, k=n_spectra)
            test_subset.extend(selected)
        
        train_spectrum_info = []
        for idx in train_subset:
            _, _, metadata = dataset[idx]
            train_spectrum_info.append({
                'patient_id': metadata['patient_id'],
                'spectrum_id': metadata['spectrum_id'],
                'dataset_index': idx
            })
        
        test_spectrum_info = []
        for idx in test_subset:
            _, _, metadata = dataset[idx]
            test_spectrum_info.append({
                'patient_id': metadata['patient_id'],
                'spectrum_id': metadata['spectrum_id'],
                'dataset_index': idx
            })
        
        # Prepare data
        X_train, y_train, _, raman_shift = prepare_data(dataset, train_subset)
        X_test, y_test, _, _ = prepare_data(dataset, test_subset)
        
        # Train model
        model, acc, auc, n_feat = train_model(X_train, y_train, X_test, y_test, C, model_type=model_type)

        results.append({
            'iteration': i,
            'test_accuracy': acc,
            'test_auc': auc,
            'n_features': n_feat
        })
        models.append(model)
        
        spectrum_selections.append({
            'iteration': i,
            'test_accuracy': acc,
            'test_auc': auc,
            'train_spectra': train_spectrum_info,
            'test_spectra': test_spectrum_info
        })
        
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{n_iterations} completed")
    
    results_df = pd.DataFrame(results)
    print(f"\nResults: Accuracy = {results_df['test_accuracy'].mean():.3f} ± {results_df['test_accuracy'].std():.3f}")
    
    return results_df, models, spectrum_selections, train_patients, test_patients, raman_shift

def analyze_spectrum_performance(spectrum_selections, dataset, patient_groups):
    """
    Analyze which spectra lead to better model performance.
    
    Returns:
        spectrum_stats: DataFrame with statistics per spectrum
    """
    # Track performance for each spectrum
    spectrum_performance = defaultdict(list)
    
    for selection in spectrum_selections:
        acc = selection['test_accuracy']
        
        # Record accuracy for each train spectrum used
        for spec_info in selection['train_spectra']:
            key = (spec_info['patient_id'], spec_info['spectrum_id'])
            spectrum_performance[key].append(acc)
    
    # Calculate statistics
    stats = []
    for (patient_id, spectrum_id), accuracies in spectrum_performance.items():
        stats.append({
            'patient_id': patient_id,
            'spectrum_id': spectrum_id,
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'median_accuracy': np.median(accuracies),
            'min_accuracy': np.min(accuracies),
            'max_accuracy': np.max(accuracies),
            'n_times_selected': len(accuracies)
        })
    
    spectrum_stats = pd.DataFrame(stats)
    spectrum_stats = spectrum_stats.sort_values('mean_accuracy', ascending=False)
    
    return spectrum_stats


def plot_accuracy_histogram(results_df, save_path=None):
    """Plot histogram of test accuracies."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(results_df['test_accuracy'], bins=30, edgecolor='black', alpha=0.7)
    
    mean_acc = results_df['test_accuracy'].mean()
    ax.axvline(mean_acc, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_acc:.3f}')
    
    std_acc = results_df['test_accuracy'].std()
    ax.axvspan(mean_acc - std_acc, mean_acc + std_acc, 
               alpha=0.2, color='red', label=f'±1 SD: {std_acc:.3f}')
    
    ax.set_xlabel('Test Accuracy', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Distribution of Test Accuracies (n={len(results_df)} iterations)', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    stats_text = f'Min: {results_df["test_accuracy"].min():.3f}\n'
    stats_text += f'Max: {results_df["test_accuracy"].max():.3f}\n'
    stats_text += f'Median: {results_df["test_accuracy"].median():.3f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig

def plot_train_accuracy_histogram(results_df, save_path=None):
    """Plot histogram of train accuracies."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(results_df['train_accuracy'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    
    mean_acc = results_df['train_accuracy'].mean()
    ax.axvline(mean_acc, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_acc:.3f}')
    
    std_acc = results_df['train_accuracy'].std()
    ax.axvspan(mean_acc - std_acc, mean_acc + std_acc, 
               alpha=0.2, color='red', label=f'±1 SD: {std_acc:.3f}')
    
    ax.set_xlabel('Train Accuracy', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Distribution of Train Accuracies (n={len(results_df)} iterations)', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    stats_text = f'Min: {results_df["train_accuracy"].min():.3f}\n'
    stats_text += f'Max: {results_df["train_accuracy"].max():.3f}\n'
    stats_text += f'Median: {results_df["train_accuracy"].median():.3f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig

def calculate_train_accuracies(dataset, patient_groups, results_df, spectrum_selections, models):
    """
    Calculate train accuracies for all saved models by running their training data through them.
    
    Args:
        dataset: The dataset object
        patient_groups: Dictionary mapping patient_id to spectrum indices
        results_df: DataFrame with model results
        spectrum_selections: List of spectrum selections for each model
        models: List of trained models
    
    Returns:
        Updated results_df with 'train_accuracy' column added
    """
    print(f"Calculating train accuracies for {len(models)} models...")
    
    train_accuracies = []
    
    for i, model in enumerate(models):
        # Get the training spectra indices for this iteration
        train_spectrum_info = spectrum_selections[i]['train_spectra']
        train_indices = [spec['dataset_index'] for spec in train_spectrum_info]
        
        # Prepare training data
        X_train, y_train, _, _ = prepare_data(dataset, train_indices)
        
        # Predict on training data
        y_train_pred = model.predict(X_train)
        
        # Calculate accuracy
        train_acc = accuracy_score(y_train, y_train_pred)
        train_accuracies.append(train_acc)
        
        if (i + 1) % 1000 == 0:
            avg_train_acc = np.mean(train_accuracies[max(0, i-999):i+1])
            print(f"  {i + 1}/{len(models)} completed - Avg train accuracy (last 1000): {avg_train_acc:.4f}")
    
    # Add train_accuracy to results_df
    results_df_copy = results_df.copy()
    results_df_copy['train_accuracy'] = train_accuracies
    
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"Train Accuracy: {np.mean(train_accuracies):.4f} ± {np.std(train_accuracies):.4f}")
    print(f"Test Accuracy:  {results_df_copy['test_accuracy'].mean():.4f} ± {results_df_copy['test_accuracy'].std():.4f}")
    print(f"Average Gap (Train - Test): {np.mean(train_accuracies) - results_df_copy['test_accuracy'].mean():.4f}")
    
    return results_df_copy

def plot_high_accuracy_analysis(results_df, spectrum_selections, accuracy_threshold=0.8, save_path=None):
    """
    Plot analysis for models with accuracy above threshold.
    1. Distribution of number of features selected
    2. Stacked bar plot: total times selected vs high-accuracy selections
    
    Args:
        results_df: DataFrame from random_subset_training()
        spectrum_selections: List from random_subset_training()
        accuracy_threshold: Minimum accuracy to include
        save_path: Optional path to save figure
    """
    # Filter high accuracy models
    high_acc_mask = results_df['test_accuracy'] >= accuracy_threshold
    high_acc_results = results_df[high_acc_mask]
    high_acc_indices = high_acc_results.index.tolist()
    
    print(f"\nModels with accuracy >= {accuracy_threshold}: {len(high_acc_indices)} / {len(results_df)}")
    
    if len(high_acc_indices) == 0:
        print(f"No models found with accuracy >= {accuracy_threshold}")
        return None
    
    # Get n_features for high accuracy models
    high_acc_features = high_acc_results['n_features'].values
    
    # Count spectrum frequency in ALL models and in high accuracy models
    spectrum_counts_all = defaultdict(int)
    spectrum_counts_high_acc = defaultdict(int)
    
    for idx in range(len(spectrum_selections)):
        selection = spectrum_selections[idx]
        is_high_acc = idx in high_acc_indices
        
        for spec_info in selection['train_spectra']:
            key = f"{spec_info['patient_id']}-{spec_info['spectrum_id']}"
            spectrum_counts_all[key] += 1
            if is_high_acc:
                spectrum_counts_high_acc[key] += 1
    
    # Sort by high accuracy count (primary), then total frequency (secondary)
    spectrum_counts_sorted = sorted(
        spectrum_counts_all.items(), 
        key=lambda x: (spectrum_counts_high_acc[x[0]], x[1]), 
        reverse=True
    )
    
    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Number of features distribution
    axes[0].hist(high_acc_features, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].axvline(high_acc_features.mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {high_acc_features.mean():.1f}')
    axes[0].set_xlabel('Number of Non-zero Features', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title(f'Feature Selection for Models with Accuracy ≥ {accuracy_threshold}', 
                     fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add statistics text
    stats_text = f'Mean: {high_acc_features.mean():.1f}\n'
    stats_text += f'Median: {np.median(high_acc_features):.1f}\n'
    stats_text += f'Range: [{high_acc_features.min()}, {high_acc_features.max()}]'
    axes[0].text(0.98, 0.98, stats_text, transform=axes[0].transAxes, 
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Stacked bar plot (all spectra)
    all_spectra = spectrum_counts_sorted
    spectrum_names = [s[0] for s in all_spectra]
    total_counts = np.array([s[1] for s in all_spectra])
    high_acc_counts = np.array([spectrum_counts_high_acc[s[0]] for s in all_spectra])
    low_acc_counts = total_counts - high_acc_counts
    
    y_pos = np.arange(len(spectrum_names))
    
    # Create stacked horizontal bar plot
    axes[1].barh(y_pos, high_acc_counts, color='seagreen', alpha=0.8, 
                label=f'Accuracy ≥ {accuracy_threshold}')
    axes[1].barh(y_pos, low_acc_counts, left=high_acc_counts, color='lightcoral', alpha=0.8,
                label=f'Accuracy < {accuracy_threshold}')
    
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(spectrum_names, fontsize=6)
    axes[1].set_xlabel('Frequency (# times selected)', fontsize=12)
    axes[1].set_title(f'All Spectra Selection Frequency (n={len(spectrum_names)})', 
                     fontsize=13, fontweight='bold')
    axes[1].invert_yaxis()
    axes[1].legend(loc='lower right', fontsize=10)
    axes[1].grid(axis='x', alpha=0.3)
    
    # Add text with statistics
    total_text = f'Total unique spectra: {len(spectrum_counts_all)}\n'
    total_text += f'High-acc rate: {100*len(high_acc_indices)/len(results_df):.1f}%'
    axes[1].text(0.02, 0.02, total_text, transform=axes[1].transAxes, 
                fontsize=10, verticalalignment='bottom', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    # Print top 10 most frequent spectra with breakdown
    print(f"\nTop 10 most frequently selected spectra:")
    for i, (spectrum, total) in enumerate(spectrum_counts_sorted[:10], 1):
        high = spectrum_counts_high_acc[spectrum]
        pct = 100 * high / total if total > 0 else 0
        print(f"  {i}. {spectrum}: {total} times total ({high} high-acc, {pct:.1f}%)")
    
    return fig

def plot_patient_spectra_with_perfect_acc(dataset, patient_groups, spectrum_selections, 
                                          train_patients, raman_shift, threshold,
                                          save_path=None, figsize_per_patient=(8, 4)):
    """
    Plot all spectra for each patient in the training set using Raman shift x-axis.
    
    Color scheme:
    - Base color: Red for cancer patients, Blue for control patients
    - Highlight: Gold/Yellow for spectra that achieved test accuracy = 1.0
    """
    perfect_acc_spectra = set()
    
    for selection in spectrum_selections:
        if selection['test_accuracy'] == threshold:
            for spec_info in selection['train_spectra']:
                key = (spec_info['patient_id'], spec_info['spectrum_id'])
                perfect_acc_spectra.add(key)
    
    print(f"Found {len(perfect_acc_spectra)} unique spectra that achieved test accuracy = 1.0")
    
    patient_labels = {}
    for pid in train_patients:
        _, _, metadata = dataset[patient_groups[pid][0]]
        staging = metadata['staging'].lower()
        patient_labels[pid] = 'cancer' if ('cancer' in staging or 'ca' in staging) else 'control'
    
    cancer_patients = [p for p in train_patients if patient_labels[p] == 'cancer']
    control_patients = [p for p in train_patients if patient_labels[p] == 'control']
    sorted_patients = cancer_patients + control_patients
    
    n_patients = len(sorted_patients)
    n_cols = 3
    n_rows = int(np.ceil(n_patients / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, 
                            figsize=(figsize_per_patient[0] * n_cols, 
                                   figsize_per_patient[1] * n_rows))
    axes = axes.flatten() if n_patients > 1 else [axes]
    
    for idx, patient_id in enumerate(sorted_patients):
        ax = axes[idx]
        label_type = patient_labels[patient_id]
        base_color = 'crimson' if label_type == 'cancer' else 'steelblue'
        
        spectrum_indices = patient_groups[patient_id]
        
        n_perfect = 0
        n_total = len(spectrum_indices)
        
        for spec_idx in spectrum_indices:
            intensity_tensor, _, metadata = dataset[spec_idx]
            intensity = intensity_tensor.squeeze().numpy()
            spectrum_id = metadata['spectrum_id']
            
            key = (patient_id, spectrum_id)
            is_perfect = key in perfect_acc_spectra
            
            if is_perfect:
                n_perfect += 1
                ax.plot(raman_shift, intensity, color='gold', linewidth=2, alpha=0.9, zorder=2)
            else:
                ax.plot(raman_shift, intensity, color=base_color, linewidth=0.8, alpha=0.4, zorder=1)
        
        ax.set_title(f'Patient {patient_id} ({label_type.upper()})\n'
                    f'{n_perfect}/{n_total} spectra with acc=1.0',
                    fontsize=10, fontweight='bold')
        ax.set_xlabel('Raman Shift (cm⁻¹)', fontsize=8)
        ax.set_ylabel('Intensity', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
        
        if idx == 0:
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='crimson', lw=2, label='Cancer (regular)', alpha=0.6),
                Line2D([0], [0], color='steelblue', lw=2, label='Control (regular)', alpha=0.6),
                Line2D([0], [0], color='gold', lw=2, label='Perfect accuracy (1.0)')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=7)
    
    for idx in range(n_patients, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(f'Training Set Spectra: {len(cancer_patients)} Cancer Patients, '
                f'{len(control_patients)} Control Patients\n'
                f'Gold = Spectra that achieved test accuracy = 1.0',
                fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    print(f"\nSummary:")
    print(f"  Total training patients: {n_patients}")
    print(f"  Cancer patients: {len(cancer_patients)}")
    print(f"  Control patients: {len(control_patients)}")
    print(f"  Total unique spectra achieving acc=1.0: {len(perfect_acc_spectra)}")
    
    return fig


def plot_individual_high_accuracy_models(dataset, patient_groups, results_df, 
                                         spectrum_selections, train_patients, 
                                         test_patients, raman_shift, 
                                         accuracy_threshold=1.0,
                                         save_dir=None, figsize_per_patient=(8, 4)):
    """
    Plot spectra for each model that achieved accuracy >= threshold using Raman shift x-axis.
    Creates one figure per high-accuracy model showing all patient spectra,
    with selected spectra highlighted in gold/yellow.
    """
    high_acc_mask = results_df['test_accuracy'] >= accuracy_threshold
    high_acc_indices = results_df[high_acc_mask].index.tolist()
    
    print(f"Found {len(high_acc_indices)} models with test accuracy >= {accuracy_threshold}")
    
    if len(high_acc_indices) == 0:
        print(f"No models found with accuracy >= {accuracy_threshold}")
        return []
    
    patient_labels = {}
    for pid in train_patients + test_patients:
        _, _, metadata = dataset[patient_groups[pid][0]]
        staging = metadata['staging'].lower()
        patient_labels[pid] = 'cancer' if ('cancer' in staging or 'ca' in staging) else 'control'
    
    train_cancer = [p for p in train_patients if patient_labels[p] == 'cancer']
    train_control = [p for p in train_patients if patient_labels[p] == 'control']
    test_cancer = [p for p in test_patients if patient_labels[p] == 'cancer']
    test_control = [p for p in test_patients if patient_labels[p] == 'control']
    
    all_patients = train_cancer + train_control + test_cancer + test_control
    
    figures = []
    
    for model_idx in high_acc_indices:
        selection = spectrum_selections[model_idx]
        test_acc = selection['test_accuracy']
        test_auc = selection['test_auc']
        
        train_selected = {(s['patient_id'], s['spectrum_id']): s['dataset_index'] 
                         for s in selection['train_spectra']}
        test_selected = {(s['patient_id'], s['spectrum_id']): s['dataset_index'] 
                        for s in selection['test_spectra']}
        
        n_patients = len(all_patients)
        n_cols = 3
        n_rows = int(np.ceil(n_patients / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols,
                                figsize=(figsize_per_patient[0] * n_cols,
                                       figsize_per_patient[1] * n_rows))
        axes = axes.flatten() if n_patients > 1 else [axes]
        
        for idx, patient_id in enumerate(all_patients):
            ax = axes[idx]
            label_type = patient_labels[patient_id]
            is_train = patient_id in train_patients
            
            base_color = 'crimson' if label_type == 'cancer' else 'steelblue'
            
            spectrum_indices = patient_groups[patient_id]
            
            selected_spec = None
            
            for spec_idx in spectrum_indices:
                intensity_tensor, _, metadata = dataset[spec_idx]
                intensity = intensity_tensor.squeeze().numpy()
                spectrum_id = metadata['spectrum_id']
                key = (patient_id, spectrum_id)
                
                is_selected = key in train_selected or key in test_selected
                
                if is_selected:
                    selected_spec = spectrum_id
                    ax.plot(raman_shift, intensity, color='gold', linewidth=2.5, alpha=1.0, zorder=2)
                else:
                    ax.plot(raman_shift, intensity, color=base_color, linewidth=0.6, alpha=0.3, zorder=1)
            
            set_label = 'TRAIN' if is_train else 'TEST'
            title = f'{set_label}: Patient {patient_id} ({label_type.upper()})'
            if selected_spec:
                title += f'\nSelected: {selected_spec}'
            
            ax.set_title(title, fontsize=9, fontweight='bold')
            ax.set_xlabel('Raman Shift (cm⁻¹)', fontsize=8)
            ax.set_ylabel('Intensity', fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.3)
            
            if idx == 0:
                from matplotlib.lines import Line2D
                legend_elements = [
                    Line2D([0], [0], color='crimson', lw=2, label='Cancer (not selected)', alpha=0.4),
                    Line2D([0], [0], color='steelblue', lw=2, label='Control (not selected)', alpha=0.4),
                    Line2D([0], [0], color='gold', lw=2.5, label='Selected in this model')
                ]
                ax.legend(handles=legend_elements, loc='upper right', fontsize=7)
        
        for idx in range(n_patients, len(axes)):
            axes[idx].axis('off')
        
        fig.suptitle(
            f'Model {model_idx}: Test Accuracy = {test_acc:.4f}, AUC = {test_auc:.4f}\n'
            f'Train: {len(train_cancer)} Cancer + {len(train_control)} Control  |  '
            f'Test: {len(test_cancer)} Cancer + {len(test_control)} Control',
            fontsize=14, fontweight='bold', y=0.995
        )
        
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'model_{model_idx}_acc_{test_acc:.4f}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {save_path}")

        figures.append(fig)
    
    print(f"\nGenerated {len(figures)} figures for models with accuracy >= {accuracy_threshold}")
    
    return figures

def plot_perfect_models_with_coefficients(dataset, patient_groups, results_df, 
                                          spectrum_selections, models, raman_shift,
                                          train_patients, test_patients, 
                                          accuracy_threshold=1.0,
                                          save_dir=None):
    """
    Plot training and test spectra with model coefficients using Raman shift x-axis.
    """
    high_acc_mask = results_df['test_accuracy'] >= accuracy_threshold
    high_acc_indices = results_df[high_acc_mask].index.tolist()
    
    print(f"Found {len(high_acc_indices)} models with test accuracy >= {accuracy_threshold}")
    
    if len(high_acc_indices) == 0:
        return []
    
    patient_labels = {}
    for pid in train_patients + test_patients:
        _, _, metadata = dataset[patient_groups[pid][0]]
        staging = metadata['staging'].lower()
        patient_labels[pid] = 'cancer' if ('cancer' in staging or 'ca' in staging) else 'control'
    
    figures = []
    
    for model_idx in high_acc_indices:
        model = models[model_idx]
        selection = spectrum_selections[model_idx]
        test_acc = selection['test_accuracy']
        test_auc = selection['test_auc']
        
        coefficients = model.coef_[0]
        n_nonzero = np.sum(coefficients != 0)
        
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(3, 1, height_ratios=[1, 1.5, 1.5], hspace=0.3)
        
        ax_coef = fig.add_subplot(gs[0])
        ax_train = fig.add_subplot(gs[1])
        ax_test = fig.add_subplot(gs[2])
        
        # ===== Plot 1: Model Coefficients with Raman Shift =====
        ax_coef.plot(raman_shift, coefficients, color='black', linewidth=1, alpha=0.7)
        
        nonzero_mask = coefficients != 0
        ax_coef.scatter(raman_shift[nonzero_mask], coefficients[nonzero_mask],
                       color='orange', s=20, zorder=3, alpha=0.8,
                       label=f'Non-zero features ({n_nonzero})')
        
        ax_coef.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax_coef.set_xlabel('Raman Shift (cm⁻¹)', fontsize=11)
        ax_coef.set_ylabel('Coefficient Value', fontsize=11)
        ax_coef.set_title('Model Coefficients (L1-Regularized Logistic Regression)', 
                         fontsize=12, fontweight='bold')
        ax_coef.legend(loc='upper right', fontsize=9)
        ax_coef.grid(alpha=0.3)
        
        coef_stats = f'Non-zero: {n_nonzero}/{len(coefficients)}\n'
        coef_stats += f'Max: {np.max(np.abs(coefficients)):.3f}\n'
        coef_stats += f'Mean |coef|: {np.mean(np.abs(coefficients[nonzero_mask])):.3f}'
        ax_coef.text(0.02, 0.98, coef_stats, transform=ax_coef.transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        # ===== Plot 2: Training Spectra with Raman Shift =====
        train_cancer_count = 0
        train_control_count = 0
        
        for spec_info in selection['train_spectra']:
            idx = spec_info['dataset_index']
            patient_id = spec_info['patient_id']
            intensity_tensor, _, metadata = dataset[idx]
            intensity = intensity_tensor.squeeze().numpy()
            
            label_type = patient_labels[patient_id]
            
            if label_type == 'cancer':
                color = 'crimson'
                alpha = 0.7
                train_cancer_count += 1
                label = 'Cancer' if train_cancer_count == 1 else None
            else:
                color = 'steelblue'
                alpha = 0.7
                train_control_count += 1
                label = 'Control' if train_control_count == 1 else None
            
            ax_train.plot(raman_shift, intensity, color=color, linewidth=1.2, alpha=alpha, label=label)
        
        ax_train.set_xlabel('Raman Shift (cm⁻¹)', fontsize=11)
        ax_train.set_ylabel('Intensity', fontsize=11)
        ax_train.set_title(f'Training Spectra (n={len(selection["train_spectra"])}): '
                          f'{train_cancer_count} Cancer, {train_control_count} Control',
                          fontsize=12, fontweight='bold')
        ax_train.legend(loc='upper right', fontsize=9)
        ax_train.grid(alpha=0.3)
        
        # ===== Plot 3: Test Spectra with Raman Shift =====
        test_cancer_count = 0
        test_control_count = 0
        
        for spec_info in selection['test_spectra']:
            idx = spec_info['dataset_index']
            patient_id = spec_info['patient_id']
            intensity_tensor, _, metadata = dataset[idx]
            intensity = intensity_tensor.squeeze().numpy()
            
            label_type = patient_labels[patient_id]
            
            if label_type == 'cancer':
                color = 'crimson'
                alpha = 0.7
                test_cancer_count += 1
                label = 'Cancer' if test_cancer_count == 1 else None
            else:
                color = 'steelblue'
                alpha = 0.7
                test_control_count += 1
                label = 'Control' if test_control_count == 1 else None
            
            ax_test.plot(raman_shift, intensity, color=color, linewidth=1.2, alpha=alpha, label=label)
        
        ax_test.set_xlabel('Raman Shift (cm⁻¹)', fontsize=11)
        ax_test.set_ylabel('Intensity', fontsize=11)
        ax_test.set_title(f'Test Spectra (n={len(selection["test_spectra"])}): '
                         f'{test_cancer_count} Cancer, {test_control_count} Control',
                         fontsize=12, fontweight='bold')
        ax_test.legend(loc='upper right', fontsize=9)
        ax_test.grid(alpha=0.3)
        
        fig.suptitle(
            f'Model {model_idx}: Test Accuracy = {test_acc:.4f}, AUC = {test_auc:.4f}',
            fontsize=15, fontweight='bold', y=0.995
        )
        
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'model_{model_idx}_spectra_coef_acc_{test_acc:.4f}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {save_path}")

        figures.append(fig)
    
    return figures


def plot_feature_selection_analysis(results_df, models, raman_shift, 
                                    accuracy_threshold=1.0, save_path=None):
    """
    Analyze feature selection across high-accuracy models with Raman shift x-axis.
    """
    high_acc_mask = results_df['test_accuracy'] >= accuracy_threshold
    high_acc_indices = results_df[high_acc_mask].index.tolist()
    
    print(f"\nAnalyzing feature selection across {len(high_acc_indices)} models")
    
    if len(high_acc_indices) == 0:
        return None, None
    
    all_coefficients = []
    for model_idx in high_acc_indices:
        model = models[model_idx]
        all_coefficients.append(model.coef_[0])
    
    all_coefficients = np.array(all_coefficients)
    n_features = all_coefficients.shape[1]
    
    feature_selection_count = np.sum(all_coefficients != 0, axis=0)
    mean_coefficient = np.mean(all_coefficients, axis=0)
    mean_abs_coefficient = np.mean(np.abs(all_coefficients), axis=0)
    
    feature_stats = pd.DataFrame({
        'raman_shift': raman_shift,
        'selection_count': feature_selection_count,
        'mean_coefficient': mean_coefficient,
        'mean_abs_coefficient': mean_abs_coefficient,
        'selection_frequency': feature_selection_count / len(high_acc_indices)
    })
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # ===== Plot 1: Feature Selection Count =====
    ax1 = axes[0]
    
    ax1.bar(feature_stats['raman_shift'], feature_stats['selection_count'],
           width=(raman_shift[1] - raman_shift[0]), color='steelblue', 
           alpha=0.7, edgecolor='none')
    
    high_freq_mask = feature_stats['selection_count'] >= len(high_acc_indices) * 0.5
    if high_freq_mask.any():
        ax1.bar(feature_stats.loc[high_freq_mask, 'raman_shift'],
               feature_stats.loc[high_freq_mask, 'selection_count'],
               width=(raman_shift[1] - raman_shift[0]), color='crimson', 
               alpha=0.8, edgecolor='none',
               label=f'Selected ≥50% of models')
    
    ax1.set_xlabel('Raman Shift (cm⁻¹)', fontsize=12)
    ax1.set_ylabel('Selection Count', fontsize=12)
    ax1.set_title(f'Feature Selection Frequency Across {len(high_acc_indices)} High-Accuracy Models',
                 fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    n_ever_selected = np.sum(feature_selection_count > 0)
    n_always_selected = np.sum(feature_selection_count == len(high_acc_indices))
    
    stats_text = f'Models analyzed: {len(high_acc_indices)}\n'
    stats_text += f'Features ever selected: {n_ever_selected}/{n_features}\n'
    stats_text += f'Features always selected: {n_always_selected}\n'
    stats_text += f'Max selection count: {feature_selection_count.max()}'
    
    ax1.text(0.98, 0.98, stats_text, transform=ax1.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    if high_freq_mask.any():
        ax1.legend(loc='upper left', fontsize=10)
    
    # ===== Plot 2: Average Coefficient Value =====
    ax2 = axes[1]
    
    ax2.plot(feature_stats['raman_shift'], feature_stats['mean_coefficient'],
            color='darkblue', linewidth=1.5, alpha=0.8, label='Mean coefficient')
    
    ax2.fill_between(feature_stats['raman_shift'], 0, feature_stats['mean_coefficient'],
                     where=(feature_stats['mean_coefficient'] > 0),
                     color='green', alpha=0.3, label='Positive (cancer indicator)')
    ax2.fill_between(feature_stats['raman_shift'], 0, feature_stats['mean_coefficient'],
                     where=(feature_stats['mean_coefficient'] < 0),
                     color='purple', alpha=0.3, label='Negative (control indicator)')
    
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    ax2.set_xlabel('Raman Shift (cm⁻¹)', fontsize=12)
    ax2.set_ylabel('Mean Coefficient Value', fontsize=12)
    ax2.set_title('Average Coefficient Values Across High-Accuracy Models',
                 fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(alpha=0.3)
    
    max_pos_coef = feature_stats['mean_coefficient'].max()
    max_neg_coef = feature_stats['mean_coefficient'].min()
    max_pos_raman = feature_stats.loc[feature_stats['mean_coefficient'].idxmax(), 'raman_shift']
    max_neg_raman = feature_stats.loc[feature_stats['mean_coefficient'].idxmin(), 'raman_shift']
    
    stats_text2 = f'Max positive: {max_pos_coef:.4f} at {max_pos_raman:.1f} cm⁻¹\n'
    stats_text2 += f'Max negative: {max_neg_coef:.4f} at {max_neg_raman:.1f} cm⁻¹\n'
    stats_text2 += f'Mean |coef|: {feature_stats["mean_abs_coefficient"].mean():.4f}'
    
    ax2.text(0.98, 0.02, stats_text2, transform=ax2.transAxes,
            fontsize=10, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    print("\nTop 10 most frequently selected Raman shifts:")
    top_features = feature_stats.nlargest(10, 'selection_count')
    for idx, row in top_features.iterrows():
        print(f"  {row['raman_shift']:.1f} cm⁻¹: selected {row['selection_count']:.0f}/{len(high_acc_indices)} times "
              f"(mean coef: {row['mean_coefficient']:.4f})")
    
    return fig, feature_stats

def plot_spectrum_selection_heatmap(dataset, patient_groups, results_df, 
                                     spectrum_selections, train_patients,
                                     accuracy_threshold=1.0, save_path=None):
    """
    Create a heatmap showing how many times each spectrum was selected
    across high-accuracy models.
    
    Args:
        dataset: OC_Dataset object
        patient_groups: Dict mapping patient_id to list of spectrum indices
        results_df: DataFrame from random_subset_training()
        spectrum_selections: List from random_subset_training()
        train_patients: List of patient IDs in training set
        accuracy_threshold: Minimum accuracy to include (default=1.0)
        save_path: Optional path to save figure
    
    Returns:
        fig: Matplotlib figure object
        selection_matrix: DataFrame with selection counts
    """
    # Find models with accuracy >= threshold
    high_acc_mask = results_df['test_accuracy'] >= accuracy_threshold
    high_acc_indices = results_df[high_acc_mask].index.tolist()
    
    print(f"\nAnalyzing spectrum selection across {len(high_acc_indices)} models with accuracy >= {accuracy_threshold}")
    
    if len(high_acc_indices) == 0:
        print(f"No models found with accuracy >= {accuracy_threshold}")
        return None, None
    
    # Determine patient labels
    patient_labels = {}
    for pid in train_patients:
        _, _, metadata = dataset[patient_groups[pid][0]]
        staging = metadata['staging'].lower()
        patient_labels[pid] = 'cancer' if ('cancer' in staging or 'ca' in staging) else 'control'
    
    # Separate and sort patients
    cancer_patients = sorted([p for p in train_patients if patient_labels[p] == 'cancer'])
    control_patients = sorted([p for p in train_patients if patient_labels[p] == 'control'])
    sorted_patients = cancer_patients + control_patients
    
    # Find maximum number of spectra per patient
    max_spectra = max(len(patient_groups[pid]) for pid in sorted_patients)
    
    # Count spectrum selections
    spectrum_selection_counts = defaultdict(int)
    for model_idx in high_acc_indices:
        selection = spectrum_selections[model_idx]
        for spec_info in selection['train_spectra']:
            key = (spec_info['patient_id'], spec_info['spectrum_id'])
            spectrum_selection_counts[key] += 1
    
    # Build matrix: rows = patients, columns = spectrum IDs
    matrix = np.full((len(sorted_patients), max_spectra), np.nan)
    
    for row_idx, patient_id in enumerate(sorted_patients):
        spectrum_indices = patient_groups[patient_id]
        for spec_idx in spectrum_indices:
            _, _, metadata = dataset[spec_idx]
            spectrum_id = metadata['spectrum_id']
            col_idx = spectrum_id - 1  # Spectrum IDs are 1-indexed
            
            key = (patient_id, spectrum_id)
            count = spectrum_selection_counts.get(key, 0)
            matrix[row_idx, col_idx] = count
    
    # Create figure with extra space on the left for labels
    fig, ax = plt.subplots(figsize=(max(12, max_spectra * 0.5), max(8, len(sorted_patients) * 0.3)))
    
    # Create custom colormap: red (low) -> yellow (high), with black for NaN
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.colors as mcolors
    
    colors = ['darkred', 'red', 'orange', 'yellow']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('red_yellow', colors, N=n_bins)
    cmap.set_bad(color='black')  # NaN values will be black
    
    # Plot heatmap
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', interpolation='nearest')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Selection Count', fontsize=12, rotation=270, labelpad=20)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(max_spectra))
    ax.set_xticklabels(np.arange(1, max_spectra + 1), fontsize=8)
    ax.set_yticks(np.arange(len(sorted_patients)))
    
    # Create y-axis labels with patient ID and label
    y_labels = [f"{pid} ({patient_labels[pid][0].upper()})" for pid in sorted_patients]
    ax.set_yticklabels(y_labels, fontsize=8)
    
    ax.set_xlabel('Spectrum ID', fontsize=12)
    ax.set_ylabel('Patient ID', fontsize=12)
    ax.set_title(f'Spectrum Selection Frequency Across {len(high_acc_indices)} High-Accuracy Models\n'
                f'(Black = No spectrum, Dark Red = Low selection, Yellow = High selection)',
                fontsize=13, fontweight='bold', pad=20)
    
    # Add grid
    ax.set_xticks(np.arange(max_spectra) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(sorted_patients)) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Add separator line between cancer and control patients
    if len(cancer_patients) > 0 and len(control_patients) > 0:
        separator_y = len(cancer_patients) - 0.5
        ax.axhline(y=separator_y, color='white', linewidth=3, linestyle='--', alpha=0.8)
        
        # Add labels for groups outside the plot area
        # Use figure coordinates to place labels in the margin
        fig.text(0.02, 0.75 - (0.15 * len(cancer_patients) / len(sorted_patients)), 'CANCER',
                rotation=90, va='center', ha='center', fontsize=11,
                fontweight='bold', color='white',
                bbox=dict(boxstyle='round', facecolor='crimson', alpha=0.8))
        fig.text(0.02, 0.30 - (0.15 * len(control_patients) / len(sorted_patients)), 'CONTROL',
                rotation=90, va='center', ha='center', fontsize=11,
                fontweight='bold', color='white',
                bbox=dict(boxstyle='round', facecolor='steelblue', alpha=0.8))
    
    # Add statistics - position below the plot instead of on the side
    total_selections = sum(spectrum_selection_counts.values())
    unique_spectra = len(spectrum_selection_counts)
    max_count = int(np.nanmax(matrix))
    
    stats_text = f'Total selections: {total_selections} | '
    stats_text += f'Unique spectra selected: {unique_spectra} | '
    stats_text += f'Max selection count: {max_count} | '
    stats_text += f'Cancer patients: {len(cancer_patients)} | '
    stats_text += f'Control patients: {len(control_patients)}'
    
    # Add stats as text below the plot
    fig.text(0.5, 0.01, stats_text, ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # Adjust layout to minimize margins and leave extra room at the bottom for text
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()
    
    # Create DataFrame for easy analysis
    selection_data = []
    for patient_id in sorted_patients:
        spectrum_indices = patient_groups[patient_id]
        for spec_idx in spectrum_indices:
            _, _, metadata = dataset[spec_idx]
            spectrum_id = metadata['spectrum_id']
            key = (patient_id, spectrum_id)
            count = spectrum_selection_counts.get(key, 0)
            
            selection_data.append({
                'patient_id': patient_id,
                'patient_label': patient_labels[patient_id],
                'spectrum_id': spectrum_id,
                'selection_count': count,
                'selection_frequency': count / len(high_acc_indices)
            })
    
    selection_df = pd.DataFrame(selection_data)
    selection_df = selection_df.sort_values(['selection_count', 'patient_id', 'spectrum_id'], 
                                            ascending=[False, True, True])
    
    # Print top selected spectra
    print(f"\nTop 10 most frequently selected spectra:")
    for idx, row in selection_df.head(10).iterrows():
        print(f"  Patient {row['patient_id']} ({row['patient_label']}), Spectrum {row['spectrum_id']}: "
              f"{row['selection_count']}/{len(high_acc_indices)} times ({row['selection_frequency']*100:.1f}%)")
    
    return fig, selection_df

def plot_selected_spectra_and_clustering(dataset, patient_groups, results_df,
                                         spectrum_selections, train_patients,
                                         accuracy_threshold=1.0, save_path=None):
    """
    Plot all spectra from high-accuracy models and show their clustering via dimensionality reduction.
    
    Creates a figure with 3 subplots:
    1. All selected spectra overlaid (red=cancer, blue=control)
    2. 3D scatter plot of PCA (first 3 components)
    3. 3D scatter plot of t-SNE (first 3 components)
    
    Args:
        dataset: OC_Dataset object
        patient_groups: Dict mapping patient_id to list of spectrum indices
        results_df: DataFrame from random_subset_training()
        spectrum_selections: List from random_subset_training()
        train_patients: List of patient IDs in training set
        accuracy_threshold: Minimum accuracy to include (default=1.0)
        save_path: Optional path to save figure
    
    Returns:
        fig: Matplotlib figure object
        selected_data: Dictionary with spectrum data and labels
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting
    
    # Find models with accuracy >= threshold
    high_acc_mask = results_df['test_accuracy'] >= accuracy_threshold
    high_acc_indices = results_df[high_acc_mask].index.tolist()
    
    print(f"\nAnalyzing {len(high_acc_indices)} models with accuracy >= {accuracy_threshold}")
    
    if len(high_acc_indices) == 0:
        print(f"No models found with accuracy >= {accuracy_threshold}")
        return None, None
    
    # Determine patient labels
    patient_labels = {}
    for pid in train_patients:
        _, _, metadata = dataset[patient_groups[pid][0]]
        staging = metadata['staging'].lower()
        patient_labels[pid] = 'cancer' if ('cancer' in staging or 'ca' in staging) else 'control'
    
    # Collect all unique spectra used in high-accuracy models
    selected_spectra = set()
    for model_idx in high_acc_indices:
        selection = spectrum_selections[model_idx]
        for spec_info in selection['train_spectra']:
            key = (spec_info['patient_id'], spec_info['spectrum_id'], spec_info['dataset_index'])
            selected_spectra.add(key)
    
    print(f"Total unique spectra selected: {len(selected_spectra)}")
    
    # Extract spectrum data
    spectra_intensities = []
    spectra_labels = []
    spectra_info = []
    
    for patient_id, spectrum_id, dataset_idx in selected_spectra:
        intensity_tensor, raman_shift_tensor, metadata = dataset[dataset_idx]
        intensity = intensity_tensor.squeeze().numpy()
        
        spectra_intensities.append(intensity)
        spectra_labels.append(patient_labels[patient_id])
        spectra_info.append({
            'patient_id': patient_id,
            'spectrum_id': spectrum_id,
            'label': patient_labels[patient_id]
        })
    
    # Get Raman shift (same for all spectra)
    raman_shift = raman_shift_tensor.squeeze().numpy()
    
    # Convert to numpy array
    X = np.array(spectra_intensities)  # Shape: (n_spectra, n_features)
    
    # Perform PCA
    print("Performing PCA dimensionality reduction...")
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)

    print(f"PCA explained variance: {pca.explained_variance_ratio_[0]:.3f}, {pca.explained_variance_ratio_[1]:.3f}, {pca.explained_variance_ratio_[2]:.3f}")
    print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.3f}")
    
    # Perform t-SNE
    print("Performing t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=3, random_state=42, perplexity=min(30, len(X) - 1))
    X_tsne = tsne.fit_transform(X)
    print("t-SNE completed")
    
    # Create figure with 3 subplots (3D plots for PCA and t-SNE)
    fig = plt.figure(figsize=(24, 6))
    
    # ===== Plot 1: Overlay of all selected spectra =====
    ax1 = fig.add_subplot(1, 3, 1)
    
    cancer_count = 0
    control_count = 0
    
    for intensity, label in zip(spectra_intensities, spectra_labels):
        if label == 'cancer':
            color = 'crimson'
            alpha = 0.3
            cancer_count += 1
            plot_label = 'Cancer' if cancer_count == 1 else None
        else:
            color = 'steelblue'
            alpha = 0.3
            control_count += 1
            plot_label = 'Control' if control_count == 1 else None
        
        ax1.plot(raman_shift, intensity, color=color, linewidth=0.8, alpha=alpha, label=plot_label)
    
    ax1.set_xlabel('Raman Shift (cm⁻¹)', fontsize=12)
    ax1.set_ylabel('Intensity', fontsize=12)
    ax1.set_title(f'All Selected Spectra (n={len(selected_spectra)})\n{cancer_count} Cancer, {control_count} Control',
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=11)
    ax1.grid(alpha=0.3)
    
    # ===== Plot 2: 3D PCA scatter plot =====
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    # Scatter using the first three PCA components
    cancer_mask = np.array(spectra_labels) == 'cancer'
    control_mask = np.array(spectra_labels) == 'control'
    
    ax2.scatter(X_pca[cancer_mask, 0], X_pca[cancer_mask, 1], X_pca[cancer_mask, 2],
                c='crimson', s=80, alpha=0.6, edgecolors='darkred', linewidth=1,
                label=f'Cancer (n={cancer_mask.sum()})')
    ax2.scatter(X_pca[control_mask, 0], X_pca[control_mask, 1], X_pca[control_mask, 2],
                c='steelblue', s=80, alpha=0.6, edgecolors='darkblue', linewidth=1,
                label=f'Control (n={control_mask.sum()})')
    
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)', fontsize=12)
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)', fontsize=12)
    ax2.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}% var)', fontsize=12)
    ax2.set_title('3D PCA Clustering', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(alpha=0.3)
    
    # ===== Plot 3: 3D t-SNE scatter plot =====
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax3.scatter(X_tsne[cancer_mask, 0], X_tsne[cancer_mask, 1], X_tsne[cancer_mask, 2],
                c='crimson', s=80, alpha=0.6, edgecolors='darkred', linewidth=1,
                label=f'Cancer (n={cancer_mask.sum()})')
    ax3.scatter(X_tsne[control_mask, 0], X_tsne[control_mask, 1], X_tsne[control_mask, 2],
                c='steelblue', s=80, alpha=0.6, edgecolors='darkblue', linewidth=1,
                label=f'Control (n={control_mask.sum()})')
    
    ax3.set_xlabel('t-SNE Comp1', fontsize=12)
    ax3.set_ylabel('t-SNE Comp2', fontsize=12)
    ax3.set_zlabel('t-SNE Comp3', fontsize=12)
    ax3.set_title('3D t-SNE Clustering', fontsize=13, fontweight='bold')
    ax3.legend(loc='best', fontsize=11)
    ax3.grid(alpha=0.3)
    
    # Overall title
    fig.suptitle(f'Spectra Analysis for Models with Accuracy ≥ {accuracy_threshold}', fontsize=15, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()
    
    # Return data for further analysis
    selected_data = {
        'intensities': X,
        'raman_shift': raman_shift,
        'labels': spectra_labels,
        'info': spectra_info,
        'pca_coords': X_pca,
        'pca_model': pca,
        'tsne_coords': X_tsne
    }
    
    return fig, selected_data


def plot_individual_model_spectra_clustering(dataset, patient_groups, results_df,
                                             spectrum_selections, train_patients,
                                             accuracy_threshold=1.0, save_dir=None):
    """
    Plot spectra and clustering for EACH individual high-accuracy model.
    Creates one figure per model with 3 subplots:
    1. All spectra in that model (red=cancer, blue=control)
    2. 3D PCA clustering of spectra in that model (first three components)
    3. 3D t-SNE clustering of spectra in that model (first three components)

    Args:
        dataset: OC_Dataset object
        patient_groups: Dict mapping patient_id to list of spectrum indices
        results_df: DataFrame from random_subset_training()
        spectrum_selections: List from random_subset_training()
        train_patients: List of patient IDs in training set
        accuracy_threshold: Minimum accuracy to include (default=1.0)
        save_dir: Optional directory to save figures

    Returns:
        figures: List of matplotlib figure objects
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from mpl_toolkits.mplot3d import Axes3D

    # Find models with accuracy >= threshold
    high_acc_mask = results_df['test_accuracy'] >= accuracy_threshold
    high_acc_indices = results_df[high_acc_mask].index.tolist()

    print(f"\nPlotting spectra and clustering for {len(high_acc_indices)} models with accuracy >= {accuracy_threshold}")

    if len(high_acc_indices) == 0:
        print(f"No models found with accuracy >= {accuracy_threshold}")
        return []

    # Determine patient labels
    patient_labels = {}
    for pid in train_patients:
        _, _, metadata = dataset[patient_groups[pid][0]]
        staging = metadata['staging'].lower()
        patient_labels[pid] = 'cancer' if ('cancer' in staging or 'ca' in staging) else 'control'

    figures = []

    # Create one figure per high-accuracy model
    for model_idx in high_acc_indices:
        selection = spectrum_selections[model_idx]
        test_acc = selection['test_accuracy']
        test_auc = selection['test_auc']

        print(f"\nProcessing Model {model_idx}...")

        # Extract spectrum data for this model
        spectra_intensities = []
        spectra_labels = []
        spectra_info = []

        for spec_info in selection['train_spectra']:
            idx = spec_info['dataset_index']
            patient_id = spec_info['patient_id']
            intensity_tensor, raman_shift_tensor, metadata = dataset[idx]
            intensity = intensity_tensor.squeeze().numpy()

            spectra_intensities.append(intensity)
            spectra_labels.append(patient_labels[patient_id])
            spectra_info.append({
                'patient_id': patient_id,
                'spectrum_id': spec_info['spectrum_id'],
                'label': patient_labels[patient_id]
            })

        # Get Raman shift (same for all spectra)
        raman_shift = raman_shift_tensor.squeeze().numpy()

        # Convert to numpy array
        X = np.array(spectra_intensities)  # Shape: (n_spectra, n_features)

        # Perform PCA with 3 components
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X)

        # Perform t-SNE with 3 components
        perplexity = min(5, len(X) - 1) if len(X) < 30 else 30
        tsne = TSNE(n_components=3, random_state=42, perplexity=perplexity)
        X_tsne = tsne.fit_transform(X)

        # Create figure with 3 subplots; the 2nd and 3rd are 3D plots
        fig = plt.figure(figsize=(20, 6))
        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2, projection='3d')
        ax3 = fig.add_subplot(1, 3, 3, projection='3d')

        # Count cancer and control
        cancer_mask = np.array(spectra_labels) == 'cancer'
        control_mask = np.array(spectra_labels) == 'control'
        cancer_count = np.sum(cancer_mask)
        control_count = np.sum(control_mask)

        # ===== Plot 1: Overlay of spectra in this model =====
        cancer_plotted = False
        control_plotted = False

        for intensity, label in zip(spectra_intensities, spectra_labels):
            if label == 'cancer':
                color = 'crimson'
                alpha = 0.5
                plot_label = 'Cancer' if not cancer_plotted else None
                cancer_plotted = True
            else:
                color = 'steelblue'
                alpha = 0.5
                plot_label = 'Control' if not control_plotted else None
                control_plotted = True

            ax1.plot(raman_shift, intensity, color=color, linewidth=1.2, alpha=alpha, label=plot_label)

        ax1.set_xlabel('Raman Shift (cm⁻¹)', fontsize=12)
        ax1.set_ylabel('Intensity', fontsize=12)
        ax1.set_title(f'Training Spectra (n={len(spectra_intensities)})\n'
                      f'{cancer_count} Cancer, {control_count} Control',
                      fontsize=13, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=11)
        ax1.grid(alpha=0.3)

        # ===== Plot 2: 3D PCA scatter plot =====
        # Plot cancer spectra
        if cancer_count > 0:
            ax2.scatter(X_pca[cancer_mask, 0], X_pca[cancer_mask, 1], X_pca[cancer_mask, 2],
                        c='crimson', s=100, alpha=0.7, edgecolors='darkred', linewidth=1.5,
                        label=f'Cancer (n={cancer_count})')
        # Plot control spectra
        if control_count > 0:
            ax2.scatter(X_pca[control_mask, 0], X_pca[control_mask, 1], X_pca[control_mask, 2],
                        c='steelblue', s=100, alpha=0.7, edgecolors='darkblue', linewidth=1.5,
                        label=f'Control (n={control_count})')

        ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)', fontsize=12)
        ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)', fontsize=12)
        ax2.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}% var)', fontsize=12)
        ax2.set_title('3D PCA Clustering', fontsize=13, fontweight='bold')
        ax2.legend(loc='best', fontsize=11)
        ax2.grid(alpha=0.3)
        stats_text = f'Total variance: {sum(pca.explained_variance_ratio_)*100:.1f}%'
        ax2.text2D(0.02, 0.98, stats_text, transform=ax2.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        # ===== Plot 3: 3D t-SNE scatter plot =====
        # Plot cancer spectra
        if cancer_count > 0:
            ax3.scatter(X_tsne[cancer_mask, 0], X_tsne[cancer_mask, 1], X_tsne[cancer_mask, 2],
                        c='crimson', s=100, alpha=0.7, edgecolors='darkred', linewidth=1.5,
                        label=f'Cancer (n={cancer_count})')
        # Plot control spectra
        if control_count > 0:
            ax3.scatter(X_tsne[control_mask, 0], X_tsne[control_mask, 1], X_tsne[control_mask, 2],
                        c='steelblue', s=100, alpha=0.7, edgecolors='darkblue', linewidth=1.5,
                        label=f'Control (n={control_count})')

        ax3.set_xlabel('t-SNE Comp1', fontsize=12)
        ax3.set_ylabel('t-SNE Comp2', fontsize=12)
        ax3.set_zlabel('t-SNE Comp3', fontsize=12)
        ax3.set_title('3D t-SNE Clustering', fontsize=13, fontweight='bold')
        ax3.legend(loc='best', fontsize=11)
        ax3.grid(alpha=0.3)
        stats_text2 = f'Perplexity: {perplexity}'
        ax3.text2D(0.02, 0.98, stats_text2, transform=ax3.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        # Overall title
        fig.suptitle(f'Model {model_idx}: Test Accuracy = {test_acc:.4f}, AUC = {test_auc:.4f}',
                     fontsize=15, fontweight='bold', y=0.98)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'model_{model_idx}_spectra_clustering_acc_{test_acc:.4f}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {save_path}")

        plt.show()
        figures.append(fig)

    print(f"\nGenerated {len(figures)} figures for models with accuracy >= {accuracy_threshold}")

    return figures


def train_on_pooled_perfect_spectra(dataset, patient_groups, results_df,
                                     spectrum_selections, train_patients, test_patients,
                                     accuracy_threshold=1.0, C=1.0,
                                     random_state=42, plot_results=True, model_type='linear_svc'):
    """
    Train a model on all unique spectra selected from high-accuracy models.
    
    This pools all spectra that were used in models with accuracy >= threshold,
    then trains a single model on this pooled dataset.
    
    Args:
        dataset: OC_Dataset object
        patient_groups: Dict mapping patient_id to list of spectrum indices
        results_df: DataFrame from random_subset_training()
        spectrum_selections: List from random_subset_training()
        train_patients: List of patient IDs in training set
        test_patients: List of patient IDs in test set
        accuracy_threshold: Minimum accuracy to include (default=1.0)
        C: Regularization strength (default=1.0)
        random_state: Random seed for reproducibility (default=42)
        plot_results: Whether to generate visualizations (default=True)
    
    Returns:
        results_dict: Dictionary containing model, metrics, and data splits
    """
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Find models with accuracy >= threshold
    high_acc_mask = results_df['test_accuracy'] >= accuracy_threshold
    high_acc_indices = results_df[high_acc_mask].index.tolist()
    
    print(f"\n{'='*70}")
    print(f"Training on Pooled Spectra from {len(high_acc_indices)} High-Accuracy Models")
    print(f"{'='*70}")
    
    if len(high_acc_indices) == 0:
        print(f"No models found with accuracy >= {accuracy_threshold}")
        return None
    
    # Determine patient labels
    patient_labels = {}
    for pid in train_patients + test_patients:
        _, _, metadata = dataset[patient_groups[pid][0]]
        staging = metadata['staging'].lower()
        patient_labels[pid] = 'cancer' if ('cancer' in staging or 'ca' in staging) else 0
    
    # Collect all unique spectra used in high-accuracy models (training set only)
    selected_train_spectra = set()
    for model_idx in high_acc_indices:
        selection = spectrum_selections[model_idx]
        for spec_info in selection['train_spectra']:
            key = (spec_info['patient_id'], spec_info['spectrum_id'], spec_info['dataset_index'])
            selected_train_spectra.add(key)
    
    print(f"\nCollecting pooled training spectra...")
    print(f"  Unique training spectra selected: {len(selected_train_spectra)}")
    
    # Extract spectrum data for training
    X_train_pool = []
    y_train_pool = []
    train_info = []
    
    for patient_id, spectrum_id, dataset_idx in selected_train_spectra:
        intensity_tensor, _, metadata = dataset[dataset_idx]
        intensity = intensity_tensor.squeeze().numpy()
        
        X_train_pool.append(intensity)
        label = 1 if patient_labels[patient_id] == 'cancer' else 0
        y_train_pool.append(label)
        train_info.append({
            'patient_id': patient_id,
            'spectrum_id': spectrum_id,
            'label': 'cancer' if label == 1 else 'control'
        })
    
    X_train_pool = np.array(X_train_pool)
    y_train_pool = np.array(y_train_pool)
    
    # Collect ALL test spectra (all spectra from test patients)
    print(f"\nCollecting test spectra...")
    X_test_pool = []
    y_test_pool = []
    test_info = []
    
    for patient_id in test_patients:
        spectrum_indices = patient_groups[patient_id]
        for spec_idx in spectrum_indices:
            intensity_tensor, _, metadata = dataset[spec_idx]
            intensity = intensity_tensor.squeeze().numpy()
            
            X_test_pool.append(intensity)
            label = 1 if patient_labels[patient_id] == 'cancer' else 0
            y_test_pool.append(label)
            test_info.append({
                'patient_id': patient_id,
                'spectrum_id': metadata['spectrum_id'],
                'label': 'cancer' if label == 1 else 'control'
            })
    
    X_test_pool = np.array(X_test_pool)
    y_test_pool = np.array(y_test_pool)
    
    print(f"  Training set: {len(X_train_pool)} spectra ({np.sum(y_train_pool)} cancer, {len(y_train_pool) - np.sum(y_train_pool)} control)")
    print(f"  Test set: {len(X_test_pool)} spectra ({np.sum(y_test_pool)} cancer, {len(y_test_pool) - np.sum(y_test_pool)} control)")
    
    # Train model on pooled data using the train_model function
    model, test_acc, test_auc, n_features = train_model(X_train_pool, y_train_pool,
                                                         X_test_pool, y_test_pool,
                                                         C=C, model_type=model_type)

    # Make predictions
    y_train_pred = model.predict(X_train_pool)
    y_test_pred = model.predict(X_test_pool)
    
    # Calculate training accuracy
    train_acc = accuracy_score(y_train_pool, y_train_pred)
    
    # Note: LinearSVC doesn't have predict_proba, so we'll use decision_function for probabilities
    y_train_scores = model.decision_function(X_train_pool)
    y_test_scores = model.decision_function(X_test_pool)
    
    # Calculate metrics
    train_auc = 0  # LinearSVC doesn't support AUC calculation
    
    # Confusion matrices
    train_cm = confusion_matrix(y_train_pool, y_train_pred)
    test_cm = confusion_matrix(y_test_pool, y_test_pred)
    
    # Print results
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"\nTraining Performance:")
    print(f"  Accuracy: {train_acc:.4f}")
    print(f"  AUC: N/A (LinearSVC doesn't support predict_proba)")
    print(f"\nTest Performance:")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  AUC: N/A (LinearSVC doesn't support predict_proba)")
    print(f"\nModel Complexity:")
    print(f"  Non-zero features (in polynomial space): {n_features}")

    print(f"\nTest Confusion Matrix:")
    print(f"  TN={test_cm[0,0]}  FP={test_cm[0,1]}")
    print(f"  FN={test_cm[1,0]}  TP={test_cm[1,1]}")
    
    print(f"\nTest Classification Report:")
    print(classification_report(y_test_pool, y_test_pred, 
                                target_names=['Control', 'Cancer']))
    
    # Calculate patient-level metrics
    print(f"\n{'='*70}")
    print("PATIENT-LEVEL ANALYSIS")
    print(f"{'='*70}")
    
    patient_level_results = []
    
    for patient_id in test_patients:
        patient_spectra_info = [info for info in test_info if info['patient_id'] == patient_id]
        patient_indices = [i for i, info in enumerate(test_info) if info['patient_id'] == patient_id]
        
        patient_true_labels = y_test_pool[patient_indices]
        patient_predictions = y_test_pred[patient_indices]
        patient_scores = y_test_scores[patient_indices]
        
        n_spectra = len(patient_indices)
        n_correct = np.sum(patient_true_labels == patient_predictions)
        n_incorrect = n_spectra - n_correct
        pct_correct = 100 * n_correct / n_spectra
        pct_incorrect = 100 * n_incorrect / n_spectra
        
        true_label = patient_spectra_info[0]['label']
        
        majority_pred = 1 if np.sum(patient_predictions) > n_spectra / 2 else 0
        majority_pred_label = 'cancer' if majority_pred == 1 else 'control'
        majority_correct = (majority_pred == patient_true_labels[0])
        
        avg_score = np.mean(patient_scores)
        
        patient_level_results.append({
            'patient_id': patient_id,
            'true_label': true_label,
            'n_spectra': n_spectra,
            'n_correct': n_correct,
            'n_incorrect': n_incorrect,
            'pct_correct': pct_correct,
            'pct_incorrect': pct_incorrect,
            'majority_vote_pred': majority_pred_label,
            'majority_vote_correct': majority_correct,
            'avg_decision_score': avg_score
        })
        
        print(f"\nPatient {patient_id} ({true_label.upper()}):")
        print(f"  Total spectra: {n_spectra}")
        print(f"  Correctly classified: {n_correct} ({pct_correct:.1f}%)")
        print(f"  Incorrectly classified: {n_incorrect} ({pct_incorrect:.1f}%)")
        print(f"  Majority vote: {majority_pred_label} {'✓' if majority_correct else '✗'}")
        print(f"  Average decision score: {avg_score:.3f}")
    
    # Create DataFrame for patient-level results
    patient_results_df = pd.DataFrame(patient_level_results)
    
    # Overall patient-level accuracy (majority vote)
    patient_level_acc = np.mean(patient_results_df['majority_vote_correct'])
    
    print(f"\n{'='*70}")
    print(f"PATIENT-LEVEL SUMMARY (Majority Vote)")
    print(f"{'='*70}")
    print(f"Patient-level accuracy: {patient_level_acc:.4f} ({np.sum(patient_results_df['majority_vote_correct'])}/{len(test_patients)})")
    
    # Calculate average correct/incorrect percentages
    cancer_patients = patient_results_df[patient_results_df['true_label'] == 'cancer']
    control_patients = patient_results_df[patient_results_df['true_label'] == 'control']
    
    if len(cancer_patients) > 0:
        print(f"\nCancer patients (n={len(cancer_patients)}):")
        print(f"  Average % spectra correct: {cancer_patients['pct_correct'].mean():.1f}%")
        print(f"  Average % spectra incorrect: {cancer_patients['pct_incorrect'].mean():.1f}%")
        print(f"  Majority vote accuracy: {cancer_patients['majority_vote_correct'].mean():.2%}")
    
    if len(control_patients) > 0:
        print(f"\nControl patients (n={len(control_patients)}):")
        print(f"  Average % spectra correct: {control_patients['pct_correct'].mean():.1f}%")
        print(f"  Average % spectra incorrect: {control_patients['pct_incorrect'].mean():.1f}%")
        print(f"  Majority vote accuracy: {control_patients['majority_vote_correct'].mean():.2%}")
    
    # ============================================================================
    # VISUALIZATION SECTION
    # ============================================================================
    
    if plot_results:
        print(f"\n{'='*70}")
        print("GENERATING VISUALIZATIONS")
        print(f"{'='*70}")
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (16, 12)
        
        # Create a comprehensive figure with multiple subplots
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Confusion Matrix (Test Set)
        ax1 = fig.add_subplot(gs[0, 0])
        sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Control', 'Cancer'],
                    yticklabels=['Control', 'Cancer'],
                    cbar_kws={'label': 'Count'}, ax=ax1)
        ax1.set_title('Test Set Confusion Matrix', fontsize=12, fontweight='bold')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        # 2. Decision Score Distribution
        ax2 = fig.add_subplot(gs[0, 1])
        cancer_scores = y_test_scores[y_test_pool == 1]
        control_scores = y_test_scores[y_test_pool == 0]
        
        ax2.hist(control_scores, bins=30, alpha=0.6, label='Control', color='green', edgecolor='black')
        ax2.hist(cancer_scores, bins=30, alpha=0.6, label='Cancer', color='red', edgecolor='black')
        ax2.axvline(0, color='black', linestyle='--', linewidth=2, label='Decision Boundary')
        ax2.set_xlabel('Decision Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Decision Score Distribution (Test Set)', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # 3. Patient-Level Accuracy
        ax3 = fig.add_subplot(gs[0, 2])
        patient_results_sorted = patient_results_df.sort_values('pct_correct')
        colors = ['green' if label == 'control' else 'red' 
                  for label in patient_results_sorted['true_label']]
        
        y_pos = np.arange(len(patient_results_sorted))
        ax3.barh(y_pos, patient_results_sorted['pct_correct'], color=colors, alpha=0.7, edgecolor='black')
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(patient_results_sorted['patient_id'], fontsize=8)
        ax3.set_xlabel('% Spectra Correctly Classified')
        ax3.set_title('Per-Patient Classification Accuracy', fontsize=12, fontweight='bold')
        ax3.axvline(50, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax3.grid(axis='x', alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', alpha=0.7, label='Cancer'),
                          Patch(facecolor='green', alpha=0.7, label='Control')]
        ax3.legend(handles=legend_elements, loc='lower right')
        
        # 4. Patient Average Decision Scores
        ax4 = fig.add_subplot(gs[1, 0])
        patient_results_sorted2 = patient_results_df.sort_values('avg_decision_score')
        colors2 = ['green' if label == 'control' else 'red' 
                   for label in patient_results_sorted2['true_label']]
        
        y_pos2 = np.arange(len(patient_results_sorted2))
        ax4.barh(y_pos2, patient_results_sorted2['avg_decision_score'], 
                color=colors2, alpha=0.7, edgecolor='black')
        ax4.set_yticks(y_pos2)
        ax4.set_yticklabels(patient_results_sorted2['patient_id'], fontsize=8)
        ax4.set_xlabel('Average Decision Score')
        ax4.set_title('Per-Patient Average Decision Score', fontsize=12, fontweight='bold')
        ax4.axvline(0, color='black', linestyle='--', linewidth=2, label='Decision Boundary')
        ax4.grid(axis='x', alpha=0.3)
        ax4.legend()
        
        # 5. Spectrum Count per Patient
        ax5 = fig.add_subplot(gs[1, 1])
        cancer_spectra = patient_results_df[patient_results_df['true_label'] == 'cancer']['n_spectra']
        control_spectra = patient_results_df[patient_results_df['true_label'] == 'control']['n_spectra']
        
        data_to_plot = []
        labels_to_plot = []
        colors_to_plot = []
        
        if len(cancer_spectra) > 0:
            data_to_plot.append(cancer_spectra)
            labels_to_plot.append('Cancer')
            colors_to_plot.append('red')
        
        if len(control_spectra) > 0:
            data_to_plot.append(control_spectra)
            labels_to_plot.append('Control')
            colors_to_plot.append('green')
        
        if data_to_plot:
            bp = ax5.boxplot(data_to_plot, labels=labels_to_plot, patch_artist=True,
                            showmeans=True, meanline=True)
            for patch, color in zip(bp['boxes'], colors_to_plot):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
        
        ax5.set_ylabel('Number of Spectra')
        ax5.set_title('Spectra Count Distribution by Patient Group', fontsize=12, fontweight='bold')
        ax5.grid(axis='y', alpha=0.3)
        
        # 6. Majority Vote Results
        ax6 = fig.add_subplot(gs[1, 2])
        majority_vote_results = patient_results_df.groupby(['true_label', 'majority_vote_correct']).size().unstack(fill_value=0)
        
        majority_vote_results.plot(kind='bar', ax=ax6, color=['#d62728', '#2ca02c'], 
                                   alpha=0.7, edgecolor='black')
        ax6.set_xlabel('True Label')
        ax6.set_ylabel('Number of Patients')
        ax6.set_title('Majority Vote Results by True Label', fontsize=12, fontweight='bold')
        ax6.set_xticklabels(ax6.get_xticklabels(), rotation=0)
        ax6.legend(['Incorrect', 'Correct'], title='Majority Vote')
        ax6.grid(axis='y', alpha=0.3)
        
        # 7. Accuracy Comparison: Spectrum-Level vs Patient-Level
        ax7 = fig.add_subplot(gs[2, 0])
        metrics = ['Spectrum-Level\n(Test Accuracy)', 'Patient-Level\n(Majority Vote)']
        accuracies = [test_acc, patient_level_acc]
        colors_acc = ['#1f77b4', '#ff7f0e']
        
        bars = ax7.bar(metrics, accuracies, color=colors_acc, alpha=0.7, edgecolor='black')
        ax7.set_ylabel('Accuracy')
        ax7.set_ylim([0, 1.1])
        ax7.set_title('Accuracy Comparison', fontsize=12, fontweight='bold')
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.3f}',
                    ha='center', va='bottom', fontweight='bold')
        ax7.grid(axis='y', alpha=0.3)
        
        # 8. Class Balance Visualization
        ax8 = fig.add_subplot(gs[2, 1])
        
        train_counts = [np.sum(y_train_pool == 0), np.sum(y_train_pool == 1)]
        test_counts = [np.sum(y_test_pool == 0), np.sum(y_test_pool == 1)]
        
        x = np.arange(2)
        width = 0.35
        
        bars1 = ax8.bar(x - width/2, train_counts, width, label='Training', 
                       color='#1f77b4', alpha=0.7, edgecolor='black')
        bars2 = ax8.bar(x + width/2, test_counts, width, label='Test', 
                       color='#ff7f0e', alpha=0.7, edgecolor='black')
        
        ax8.set_ylabel('Number of Spectra')
        ax8.set_title('Class Balance: Training vs Test', fontsize=12, fontweight='bold')
        ax8.set_xticks(x)
        ax8.set_xticklabels(['Control', 'Cancer'])
        ax8.legend()
        ax8.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax8.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=9)
        
        # 9. Individual Spectrum Decision Scores (Scatter)
        ax9 = fig.add_subplot(gs[2, 2])
        
        for i, (patient_id, group) in enumerate(patient_results_df.groupby('patient_id')):
            patient_indices = [j for j, info in enumerate(test_info) if info['patient_id'] == patient_id]
            patient_scores_scatter = y_test_scores[patient_indices]
            true_label = group['true_label'].iloc[0]
            
            color = 'red' if true_label == 'cancer' else 'green'
            marker = 'o' if group['majority_vote_correct'].iloc[0] else 'x'
            
            ax9.scatter([i] * len(patient_scores_scatter), patient_scores_scatter, 
                       c=color, marker=marker, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        ax9.axhline(0, color='black', linestyle='--', linewidth=2, label='Decision Boundary')
        ax9.set_xlabel('Patient Index')
        ax9.set_ylabel('Decision Score')
        ax9.set_title('Individual Spectrum Decision Scores by Patient', fontsize=12, fontweight='bold')
        ax9.grid(alpha=0.3)
        
        # Custom legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                   markersize=8, label='Cancer (Correct MV)', markeredgecolor='black'),
            Line2D([0], [0], marker='x', color='w', markerfacecolor='red', 
                   markersize=8, label='Cancer (Incorrect MV)', markeredgecolor='black'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                   markersize=8, label='Control (Correct MV)', markeredgecolor='black'),
            Line2D([0], [0], marker='x', color='w', markerfacecolor='green', 
                   markersize=8, label='Control (Incorrect MV)', markeredgecolor='black')
        ]
        ax9.legend(handles=legend_elements, fontsize=8, loc='best')
        
        # 10. Summary Statistics Table
        ax10 = fig.add_subplot(gs[3, :])
        ax10.axis('tight')
        ax10.axis('off')
        
        # Prepare summary data
        summary_data = [
            ['Metric', 'Value'],
            ['', ''],
            ['Training Set Size', f"{len(X_train_pool)} spectra"],
            ['Test Set Size', f"{len(X_test_pool)} spectra"],
            ['Number of Test Patients', f"{len(test_patients)}"],
            ['', ''],
            ['Spectrum-Level Test Accuracy', f"{test_acc:.4f}"],
            ['Patient-Level Accuracy (Majority Vote)', f"{patient_level_acc:.4f}"],
            ['', ''],
            ['Non-zero Features', f"{n_features}"],
            ['Regularization (C)', f"{C}"],
        ]
        
        if len(cancer_patients) > 0:
            summary_data.extend([
                ['', ''],
                [f'Cancer Patients (n={len(cancer_patients)})', ''],
                ['  Avg % Spectra Correct', f"{cancer_patients['pct_correct'].mean():.1f}%"],
                ['  Majority Vote Accuracy', f"{cancer_patients['majority_vote_correct'].mean():.2%}"],
            ])
        
        if len(control_patients) > 0:
            summary_data.extend([
                ['', ''],
                [f'Control Patients (n={len(control_patients)})', ''],
                ['  Avg % Spectra Correct', f"{control_patients['pct_correct'].mean():.1f}%"],
                ['  Majority Vote Accuracy', f"{control_patients['majority_vote_correct'].mean():.2%}"],
            ])
        
        table = ax10.table(cellText=summary_data, cellLoc='left', loc='center',
                          colWidths=[0.5, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style the header row
        for i in range(2):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style alternating rows
        for i in range(1, len(summary_data)):
            if i % 2 == 0:
                for j in range(2):
                    table[(i, j)].set_facecolor('#E7E6E6')
        
        plt.suptitle('Pooled Spectra Model: Comprehensive Results', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        plt.show()
        
        print("Visualizations generated successfully!")
    
    # Return results
    results_dict = {
        'model': model,
        'X_train': X_train_pool,
        'y_train': y_train_pool,
        'X_test': X_test_pool,
        'y_test': y_test_pool,
        'train_info': train_info,
        'test_info': test_info,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'train_auc': train_auc,
        'test_auc': test_auc,
        'n_features': n_features,
        'train_predictions': y_train_pred,
        'test_predictions': y_test_pred,
        'train_scores': y_train_scores,
        'test_scores': y_test_scores,
        'train_confusion_matrix': train_cm,
        'test_confusion_matrix': test_cm,
        'patient_level_results': patient_results_df,
        'patient_level_accuracy': patient_level_acc
    }
    
    return results_dict


# Example usage
if __name__ == "__main__":

    # 1. Load your dataset
    data_folder = r'C:\Users\Yifei\Downloads\data_izabella_filtered\data_izabella_filtered'

    preprocessor = SpectrumPreprocessor(baseline_correction=True,
                remove_cosmic_rays=False,
                normalization=True,
                smoothing=True)
    
    dataset = OC_Dataset(data_folder, preprocessor, augmentor=None)
    patient_groups = get_patient_groups(dataset)

    # # Train with spectrum tracking
    # results, models, spectrum_selections, train_patients, test_patients, raman_shift = random_subset_training(
    #     dataset, patient_groups, 
    #     n_iterations=100000, 
    #     C=5.0,
    #     test_size=0.2,
    #     n_spectra=5,
    #     model_type = 'random_forest',
    # )
    # # Save the training outputs for later analysis
    # output = {
    #     "results": results,
    #     "models": models,
    #     "spectrum_selections": spectrum_selections,
    #     "train_patients": train_patients,
    #     "test_patients": test_patients,
    #     "raman_shift": raman_shift
    # }
    # with open("n100000_c5_t02_s5_with_preprocessing_random_forest_training_output.pkl", "wb") as f:
    #     pickle.dump(output, f)

    # Load training outputs
    with open(r"D:\n100000_c5_t02_s5_with_preprocessing_logistics_training_output.pkl", "rb") as f:
        output = pickle.load(f)
        
    results = output["results"]
    models = output["models"]
    spectrum_selections = output["spectrum_selections"]
    train_patients = output["train_patients"]
    test_patients = output["test_patients"]
    raman_shift = output["raman_shift"]

    # # Analyze which spectra perform best
    # spectrum_stats = analyze_spectrum_performance(spectrum_selections, dataset, patient_groups)

    # # Plot results
    # plot_accuracy_histogram(results, save_path='n100000_c5_t02_s5_with_preprocessing_random_forest_accuracy_histogram.png')

    results_with_train = calculate_train_accuracies(
        dataset, patient_groups, results, spectrum_selections, models
    )

    # save updated results with train accuracies to file
    with open("train_accuracies.pkl", "wb") as f:
        pickle.dump(results_with_train, f)


    # Plot only train accuracy
    plot_train_accuracy_histogram(results_with_train, save_path="train_accuracy_hist.png")

    # # NEW: Analyze high-accuracy models
    # plot_high_accuracy_analysis(results, spectrum_selections, 
    #                            accuracy_threshold=0.8, 
    #                            save_path='n100000_c5_t02_s5_with_preprocessing_random_forest_high_accuracy_analysis.png')
    
    # plot_patient_spectra_with_perfect_acc(
    #     dataset, 
    #     patient_groups, 
    #     spectrum_selections, 
    #     train_patients,
    #     raman_shift,  # New parameter
    #     threshold=0.8,
    #     save_path='n100000_c5_t02_s1_patient_spectra_perfect_accuracy.png'
    # )

    # # Plot individual high accuracy models
    # figures = plot_individual_high_accuracy_models(
    #     dataset, 
    #     patient_groups, 
    #     results_df=results,
    #     spectrum_selections=spectrum_selections,
    #     train_patients=train_patients,
    #     test_patients=test_patients,
    #     raman_shift=raman_shift,  # New parameter
    #     accuracy_threshold=0.8,
    #     save_dir='n100000_c5_t02_high_accuracy_models'
    # )

    ## Pass raman_shift to plotting functions
    # figures = plot_perfect_models_with_coefficients(
    #     dataset, patient_groups, 
    #     results_df=results,
    #     spectrum_selections=spectrum_selections,
    #     models=models,
    #     raman_shift=raman_shift,  # New parameter
    #     train_patients=train_patients,
    #     test_patients=test_patients,
    #     accuracy_threshold=0.8,
    #     save_dir='n100000_c5_t02_s5_with_preprocessing_random_forest_perfect_models_with_coefficients'
    #     )

    fig_features, feature_stats = plot_feature_selection_analysis(
        results_df=results,
        models=models,
        raman_shift=raman_shift,  # New parameter
        accuracy_threshold=0.8,
        save_path='n100000_c5_t02_s5_with_preprocessing_random_forest_feature_selection_analysis.png'
        )

    # fig_heatmap, selection_df = plot_spectrum_selection_heatmap(
    #     dataset=dataset,
    #     patient_groups=patient_groups,
    #     results_df=results,
    #     spectrum_selections=spectrum_selections,
    #     train_patients=train_patients,
    #     accuracy_threshold=0.8,
    #     save_path='n100000_c5_t02_s5_with_preprocessing_random_forest_spectrum_selection_heatmap.png'
    #     )
    
    # fig_clustering, selected_data = plot_selected_spectra_and_clustering(
    # dataset=dataset,
    # patient_groups=patient_groups,
    # results_df=results,
    # spectrum_selections=spectrum_selections,
    # train_patients=train_patients,
    # accuracy_threshold=0.8,
    # save_path='n100000_c5_t02_s5_with_preprocessing_random_forest_selected_spectra_clustering.png'
    # )

    # figs_individual_clustering = plot_individual_model_spectra_clustering(
    # dataset=dataset,
    # patient_groups=patient_groups,
    # results_df=results,
    # spectrum_selections=spectrum_selections,
    # train_patients=train_patients,
    # accuracy_threshold=0.8,
    # save_dir='n100000_c5_t02_s5_with_preprocessing_random_forest_individual_model_clustering'
    # )

    # pooled_results = train_on_pooled_perfect_spectra(
    #     dataset, patient_groups, results, spectrum_selections,
    #     train_patients, test_patients, 
    #     accuracy_threshold=0.8, 
    #     plot_results=True,  # Set to False to disable visualizations
    #     C=5.0,
    #     model_type='random_forest'  # Change model type as needed,
    # )

    # Get best model
    best_idx = results['test_accuracy'].idxmax()
    best_model = models[best_idx]
    best_selection = spectrum_selections[best_idx]
    
    print(f"\nBest model (iteration {best_idx}):")
    print(f"  Accuracy: {results.loc[best_idx, 'test_accuracy']:.3f}")
    print(f"  AUC: {results.loc[best_idx, 'test_auc']:.3f}")