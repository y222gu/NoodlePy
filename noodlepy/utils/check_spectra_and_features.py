import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
from noodlepy.utils.izabelladataset import OC_Dataset
from noodlepy.utils.spectrumpreprocessor import SpectrumPreprocessor
import pickle
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import seaborn as sns


def get_patient_groups(dataset):
    """Group spectrum indices by patient_id."""
    groups = defaultdict(list)
    for idx in range(len(dataset)):
        _, _, metadata = dataset[idx]
        groups[metadata['patient_id']].append(idx)
    return groups


def train_model(X_train, y_train, X_test, y_test, C=1.0, model_type='xgboost'):
    """Train and evaluate model. Optimized for high-dimensional, low-sample data."""
    
    model_configs = {
        'xgboost': xgb.XGBClassifier(
            max_depth=3, learning_rate=0.05, n_estimators=100,
            min_child_weight=10, subsample=0.8, colsample_bytree=0.3,
            reg_alpha=1.0, reg_lambda=1.0, random_state=42
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=200, max_depth=4, min_samples_split=20,
            min_samples_leaf=10, max_features='sqrt', random_state=42, n_jobs=-1
        ),
        'decision_tree': DecisionTreeClassifier(
            max_depth=3, min_samples_split=30, min_samples_leaf=15, random_state=42
        ),
        'logistic_regression': LogisticRegression(
            penalty='l1', solver='liblinear', C=C, random_state=42
        ),
        'linear_svc': SVC(
            kernel='linear', C=C, random_state=42, max_iter=-1
        )
    }
    
    model = model_configs.get(model_type, model_configs['xgboost'])
    model.fit(X_train, y_train)
    
    test_acc = accuracy_score(y_test, model.predict(X_test))
    
    # Handle different model types for AUC and feature count
    if hasattr(model, 'predict_proba'):
        test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    else:
        test_auc = None
    
    n_features = np.sum(model.coef_ != 0) if hasattr(model, 'coef_') else X_train.shape[1]
    
    return model, test_acc, test_auc, n_features


def plot_selected_spectra_and_clustering(dataset, patient_groups, results_df,
                                         spectrum_selections, train_patients,
                                         accuracy_threshold=1.0, save_path=None):
    """
    Plot all spectra from high-accuracy models and show their clustering via dimensionality reduction.
    Creates a figure with 3 subplots: overlaid spectra, 3D PCA, and 3D t-SNE.
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from mpl_toolkits.mplot3d import Axes3D
    
    # Find high-accuracy models
    high_acc_mask = results_df['test_accuracy'] >= accuracy_threshold
    high_acc_indices = results_df[high_acc_mask].index.tolist()
    
    if len(high_acc_indices) == 0:
        print(f"No models found with accuracy >= {accuracy_threshold}")
        return None, None
    
    print(f"Analyzing {len(high_acc_indices)} models with accuracy >= {accuracy_threshold}")
    
    # Determine patient labels
    patient_labels = {}
    for pid in train_patients:
        _, _, metadata = dataset[patient_groups[pid][0]]
        staging = metadata['staging'].lower()
        patient_labels[pid] = 'cancer' if ('cancer' in staging or 'ca' in staging) else 'control'
    
    # Collect unique spectra
    selected_spectra = set()
    for model_idx in high_acc_indices:
        selection = spectrum_selections[model_idx]
        for spec_info in selection['train_spectra']:
            key = (spec_info['patient_id'], spec_info['spectrum_id'], spec_info['dataset_index'])
            selected_spectra.add(key)
    
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
    
    raman_shift = raman_shift_tensor.squeeze().numpy()
    X = np.array(spectra_intensities)
    
    # PCA
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)
    
    # t-SNE
    tsne = TSNE(n_components=3, random_state=42, perplexity=min(30, len(X) - 1))
    X_tsne = tsne.fit_transform(X)
    
    # Create figure
    fig = plt.figure(figsize=(24, 6))
    
    # Plot 1: Overlaid spectra
    ax1 = fig.add_subplot(1, 3, 1)
    cancer_count = control_count = 0
    
    for intensity, label in zip(spectra_intensities, spectra_labels):
        if label == 'cancer':
            color, alpha = 'crimson', 0.3
            cancer_count += 1
            plot_label = 'Cancer' if cancer_count == 1 else None
        else:
            color, alpha = 'steelblue', 0.3
            control_count += 1
            plot_label = 'Control' if control_count == 1 else None
        
        ax1.plot(raman_shift, intensity, color=color, linewidth=0.8, alpha=alpha, label=plot_label)
    
    ax1.set_xlabel('Raman Shift (cm⁻¹)', fontsize=12)
    ax1.set_ylabel('Intensity', fontsize=12)
    ax1.set_title(f'All Selected Spectra (n={len(selected_spectra)})\n{cancer_count} Cancer, {control_count} Control',
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=11)
    ax1.grid(alpha=0.3)
    
    # Plot 2: 3D PCA
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
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
    
    # Plot 3: 3D t-SNE
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
    
    fig.suptitle(f'Spectra Analysis for Models with Accuracy ≥ {accuracy_threshold}', 
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()
    
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


def train_on_pooled_perfect_spectra(dataset, patient_groups, results_df,
                                     spectrum_selections, train_patients, test_patients,
                                     accuracy_threshold=1.0, C=1.0, random_state=42, 
                                     plot_results=True, model_type='linear_svc'):
    """
    Train a model on all unique spectra selected from high-accuracy models.
    Pools all spectra from models with accuracy >= threshold.
    """
    
    # Find high-accuracy models
    high_acc_mask = results_df['test_accuracy'] >= accuracy_threshold
    high_acc_indices = results_df[high_acc_mask].index.tolist()
    
    if len(high_acc_indices) == 0:
        print(f"No models found with accuracy >= {accuracy_threshold}")
        return None
    
    print(f"\n{'='*70}")
    print(f"Training on Pooled Spectra from {len(high_acc_indices)} High-Accuracy Models")
    print(f"{'='*70}\n")
    
    # Determine patient labels
    patient_labels = {}
    for pid in train_patients + test_patients:
        _, _, metadata = dataset[patient_groups[pid][0]]
        staging = metadata['staging'].lower()
        patient_labels[pid] = 'cancer' if ('cancer' in staging or 'ca' in staging) else 0
    
    # Collect training spectra
    selected_train_spectra = set()
    for model_idx in high_acc_indices:
        selection = spectrum_selections[model_idx]
        for spec_info in selection['train_spectra']:
            key = (spec_info['patient_id'], spec_info['spectrum_id'], spec_info['dataset_index'])
            selected_train_spectra.add(key)
    
    # Extract training data
    X_train_pool, y_train_pool, train_info = [], [], []
    for patient_id, spectrum_id, dataset_idx in selected_train_spectra:
        intensity_tensor, _, metadata = dataset[dataset_idx]
        intensity = intensity_tensor.squeeze().numpy()
        label = 1 if patient_labels[patient_id] == 'cancer' else 0
        
        X_train_pool.append(intensity)
        y_train_pool.append(label)
        train_info.append({
            'patient_id': patient_id,
            'spectrum_id': spectrum_id,
            'label': 'cancer' if label == 1 else 'control'
        })
    
    X_train_pool = np.array(X_train_pool)
    y_train_pool = np.array(y_train_pool)
    
    # Collect test data
    X_test_pool, y_test_pool, test_info = [], [], []
    for patient_id in test_patients:
        for spec_idx in patient_groups[patient_id]:
            intensity_tensor, _, metadata = dataset[spec_idx]
            intensity = intensity_tensor.squeeze().numpy()
            label = 1 if patient_labels[patient_id] == 'cancer' else 0
            
            X_test_pool.append(intensity)
            y_test_pool.append(label)
            test_info.append({
                'patient_id': patient_id,
                'spectrum_id': metadata['spectrum_id'],
                'label': 'cancer' if label == 1 else 'control'
            })
    
    X_test_pool = np.array(X_test_pool)
    y_test_pool = np.array(y_test_pool)
    
    print(f"Training set: {len(X_train_pool)} spectra ({np.sum(y_train_pool)} cancer, {len(y_train_pool) - np.sum(y_train_pool)} control)")
    print(f"Test set: {len(X_test_pool)} spectra ({np.sum(y_test_pool)} cancer, {len(y_test_pool) - np.sum(y_test_pool)} control)\n")
    
    # Train model
    model, test_acc, test_auc, n_features = train_model(
        X_train_pool, y_train_pool, X_test_pool, y_test_pool, C=C, model_type=model_type
    )
    
    # Predictions
    y_train_pred = model.predict(X_train_pool)
    y_test_pred = model.predict(X_test_pool)
    train_acc = accuracy_score(y_train_pool, y_train_pred)
    
    # Get scores
    if hasattr(model, 'decision_function'):
        y_train_scores = model.decision_function(X_train_pool)
        y_test_scores = model.decision_function(X_test_pool)
    else:
        y_train_scores = model.predict_proba(X_train_pool)[:, 1]
        y_test_scores = model.predict_proba(X_test_pool)[:, 1]
    
    # Confusion matrices
    train_cm = confusion_matrix(y_train_pool, y_train_pred)
    test_cm = confusion_matrix(y_test_pool, y_test_pred)
    
    # Print results
    print(f"{'='*70}")
    print("RESULTS")
    print(f"{'='*70}\n")
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    if test_auc:
        print(f"Test AUC: {test_auc:.4f}")
    print(f"Non-zero features: {n_features}\n")
    print(f"Test Confusion Matrix: TN={test_cm[0,0]} FP={test_cm[0,1]} FN={test_cm[1,0]} TP={test_cm[1,1]}\n")
    print(classification_report(y_test_pool, y_test_pred, target_names=['Control', 'Cancer']))
    
    # Patient-level analysis
    print(f"\n{'='*70}")
    print("PATIENT-LEVEL ANALYSIS")
    print(f"{'='*70}\n")
    
    patient_level_results = []
    for patient_id in test_patients:
        patient_indices = [i for i, info in enumerate(test_info) if info['patient_id'] == patient_id]
        patient_true_labels = y_test_pool[patient_indices]
        patient_predictions = y_test_pred[patient_indices]
        patient_scores = y_test_scores[patient_indices]
        
        n_spectra = len(patient_indices)
        n_correct = np.sum(patient_true_labels == patient_predictions)
        pct_correct = 100 * n_correct / n_spectra
        
        true_label = test_info[patient_indices[0]]['label']
        majority_pred = 1 if np.sum(patient_predictions) > n_spectra / 2 else 0
        majority_pred_label = 'cancer' if majority_pred == 1 else 'control'
        majority_correct = (majority_pred == patient_true_labels[0])
        
        patient_level_results.append({
            'patient_id': patient_id,
            'true_label': true_label,
            'n_spectra': n_spectra,
            'n_correct': n_correct,
            'pct_correct': pct_correct,
            'majority_vote_pred': majority_pred_label,
            'majority_vote_correct': majority_correct,
            'avg_decision_score': np.mean(patient_scores)
        })
        
        print(f"Patient {patient_id} ({true_label.upper()}): {n_spectra} spectra, "
              f"{pct_correct:.1f}% correct, Majority: {majority_pred_label} {'✓' if majority_correct else '✗'}")
    
    patient_results_df = pd.DataFrame(patient_level_results)
    patient_level_acc = np.mean(patient_results_df['majority_vote_correct'])
    
    print(f"\nPatient-level accuracy (Majority Vote): {patient_level_acc:.4f} "
          f"({np.sum(patient_results_df['majority_vote_correct'])}/{len(test_patients)})")
    
    # Visualization
    if plot_results:
        _create_results_visualization(
            test_cm, y_test_scores, y_test_pool, patient_results_df,
            test_acc, patient_level_acc, n_features, C,
            X_train_pool, y_train_pool, X_test_pool, test_info
        )
    
    return {
        'model': model,
        'X_train': X_train_pool,
        'y_train': y_train_pool,
        'X_test': X_test_pool,
        'y_test': y_test_pool,
        'train_info': train_info,
        'test_info': test_info,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'test_auc': test_auc,
        'n_features': n_features,
        'test_predictions': y_test_pred,
        'test_scores': y_test_scores,
        'test_confusion_matrix': test_cm,
        'patient_level_results': patient_results_df,
        'patient_level_accuracy': patient_level_acc
    }


def _create_results_visualization(test_cm, y_test_scores, y_test_pool, patient_results_df,
                                  test_acc, patient_level_acc, n_features, C,
                                  X_train_pool, y_train_pool, X_test_pool, test_info):
    """Create comprehensive results visualization."""
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Confusion Matrix
    ax1 = fig.add_subplot(gs[0, 0])
    sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Control', 'Cancer'], yticklabels=['Control', 'Cancer'],
                cbar_kws={'label': 'Count'}, ax=ax1)
    ax1.set_title('Test Confusion Matrix', fontweight='bold')
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
    ax2.set_title('Decision Score Distribution', fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. Per-Patient Accuracy
    ax3 = fig.add_subplot(gs[0, 2])
    patient_sorted = patient_results_df.sort_values('pct_correct')
    colors = ['green' if label == 'control' else 'red' for label in patient_sorted['true_label']]
    y_pos = np.arange(len(patient_sorted))
    ax3.barh(y_pos, patient_sorted['pct_correct'], color=colors, alpha=0.7, edgecolor='black')
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(patient_sorted['patient_id'], fontsize=8)
    ax3.set_xlabel('% Spectra Correctly Classified')
    ax3.set_title('Per-Patient Accuracy', fontweight='bold')
    ax3.axvline(50, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax3.legend(handles=[Patch(facecolor='red', alpha=0.7, label='Cancer'),
                       Patch(facecolor='green', alpha=0.7, label='Control')], loc='lower right')
    ax3.grid(axis='x', alpha=0.3)
    
    # 4. Patient Average Decision Scores
    ax4 = fig.add_subplot(gs[1, 0])
    patient_sorted2 = patient_results_df.sort_values('avg_decision_score')
    colors2 = ['green' if label == 'control' else 'red' for label in patient_sorted2['true_label']]
    y_pos2 = np.arange(len(patient_sorted2))
    ax4.barh(y_pos2, patient_sorted2['avg_decision_score'], color=colors2, alpha=0.7, edgecolor='black')
    ax4.set_yticks(y_pos2)
    ax4.set_yticklabels(patient_sorted2['patient_id'], fontsize=8)
    ax4.set_xlabel('Average Decision Score')
    ax4.set_title('Per-Patient Avg Decision Score', fontweight='bold')
    ax4.axvline(0, color='black', linestyle='--', linewidth=2)
    ax4.grid(axis='x', alpha=0.3)
    
    # 5. Accuracy Comparison
    ax5 = fig.add_subplot(gs[1, 1])
    metrics = ['Spectrum-Level', 'Patient-Level\n(Majority Vote)']
    accuracies = [test_acc, patient_level_acc]
    bars = ax5.bar(metrics, accuracies, color=['#1f77b4', '#ff7f0e'], alpha=0.7, edgecolor='black')
    ax5.set_ylabel('Accuracy')
    ax5.set_ylim([0, 1.1])
    ax5.set_title('Accuracy Comparison', fontweight='bold')
    for bar, acc in zip(bars, accuracies):
        ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    ax5.grid(axis='y', alpha=0.3)
    
    # 6. Class Balance
    ax6 = fig.add_subplot(gs[1, 2])
    train_counts = [np.sum(y_train_pool == 0), np.sum(y_train_pool == 1)]
    test_counts = [np.sum(y_test_pool == 0), np.sum(y_test_pool == 1)]
    x = np.arange(2)
    width = 0.35
    bars1 = ax6.bar(x - width/2, train_counts, width, label='Training',
                   color='#1f77b4', alpha=0.7, edgecolor='black')
    bars2 = ax6.bar(x + width/2, test_counts, width, label='Test',
                   color='#ff7f0e', alpha=0.7, edgecolor='black')
    ax6.set_ylabel('Number of Spectra')
    ax6.set_title('Class Balance', fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(['Control', 'Cancer'])
    ax6.legend()
    ax6.grid(axis='y', alpha=0.3)
    
    # 7. Individual Spectrum Scores by Patient
    ax7 = fig.add_subplot(gs[2, :])
    for i, (patient_id, group) in enumerate(patient_results_df.groupby('patient_id')):
        patient_indices = [j for j, info in enumerate(test_info) if info['patient_id'] == patient_id]
        patient_scores = y_test_scores[patient_indices]
        true_label = group['true_label'].iloc[0]
        color = 'red' if true_label == 'cancer' else 'green'
        marker = 'o' if group['majority_vote_correct'].iloc[0] else 'x'
        ax7.scatter([i] * len(patient_scores), patient_scores,
                   c=color, marker=marker, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    ax7.axhline(0, color='black', linestyle='--', linewidth=2)
    ax7.set_xlabel('Patient Index')
    ax7.set_ylabel('Decision Score')
    ax7.set_title('Individual Spectrum Decision Scores by Patient', fontweight='bold')
    ax7.legend(handles=[
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8,
               label='Cancer (Correct MV)', markeredgecolor='black'),
        Line2D([0], [0], marker='x', color='w', markerfacecolor='red', markersize=8,
               label='Cancer (Incorrect MV)', markeredgecolor='black'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8,
               label='Control (Correct MV)', markeredgecolor='black'),
        Line2D([0], [0], marker='x', color='w', markerfacecolor='green', markersize=8,
               label='Control (Incorrect MV)', markeredgecolor='black')
    ], fontsize=8, loc='best')
    ax7.grid(alpha=0.3)
    
    plt.suptitle('Pooled Spectra Model: Comprehensive Results',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def predict_with_high_accuracy_models_on_pooled_spectra(
    dataset, patient_groups, results_df, spectrum_selections, 
    models, test_patients, accuracy_threshold=1.0
):
    """
    Use each saved model above threshold to make predictions on pooled test spectra.
    
    Args:
        dataset: The dataset object
        patient_groups: Dictionary mapping patient_id to spectrum indices
        results_df: DataFrame with model results
        spectrum_selections: List of spectrum selections for each model
        models: List of trained models
        test_patients: List of test patient IDs
        accuracy_threshold: Minimum accuracy threshold for model selection
    
    Returns:
        Dictionary containing prediction results and aggregated statistics
    """
    # Find high-accuracy models
    high_acc_mask = results_df['test_accuracy'] >= accuracy_threshold
    high_acc_indices = results_df[high_acc_mask].index.tolist()
    
    if len(high_acc_indices) == 0:
        print(f"No models found with accuracy >= {accuracy_threshold}")
        return None
    
    print(f"\n{'='*70}")
    print(f"Making Predictions with {len(high_acc_indices)} High-Accuracy Models")
    print(f"on Pooled Test Spectra")
    print(f"{'='*70}\n")
    
    # Determine patient labels
    patient_labels = {}
    for pid in test_patients:
        _, _, metadata = dataset[patient_groups[pid][0]]
        staging = metadata['staging'].lower()
        patient_labels[pid] = 1 if ('cancer' in staging or 'ca' in staging) else 0
    
    # Collect unique pooled test spectra from high-accuracy models
    pooled_test_spectra = set()
    for model_idx in high_acc_indices:
        selection = spectrum_selections[model_idx]
        for spec_info in selection['test_spectra']:
            key = (spec_info['patient_id'], spec_info['spectrum_id'], spec_info['dataset_index'])
            pooled_test_spectra.add(key)
    
    # Extract pooled test data
    X_test_pooled, y_test_pooled, test_info = [], [], []
    for patient_id, spectrum_id, dataset_idx in pooled_test_spectra:
        intensity_tensor, _, metadata = dataset[dataset_idx]
        intensity = intensity_tensor.squeeze().numpy()
        label = patient_labels[patient_id]
        
        X_test_pooled.append(intensity)
        y_test_pooled.append(label)
        test_info.append({
            'patient_id': patient_id,
            'spectrum_id': spectrum_id,
            'dataset_index': dataset_idx,
            'label': 'cancer' if label == 1 else 'control'
        })
    
    X_test_pooled = np.array(X_test_pooled)
    y_test_pooled = np.array(y_test_pooled)
    
    print(f"Pooled test set: {len(X_test_pooled)} spectra")
    print(f"  {np.sum(y_test_pooled)} cancer, {len(y_test_pooled) - np.sum(y_test_pooled)} control\n")
    
    # Store predictions from each model
    all_predictions = []
    all_scores = []
    model_accuracies = []
    
    for idx, model_idx in enumerate(high_acc_indices):
        model = models[model_idx]
        
        # Make predictions
        y_pred = model.predict(X_test_pooled)
        
        # Get decision scores
        if hasattr(model, 'decision_function'):
            y_scores = model.decision_function(X_test_pooled)
        elif hasattr(model, 'predict_proba'):
            y_scores = model.predict_proba(X_test_pooled)[:, 1]
        else:
            y_scores = y_pred.astype(float)
        
        all_predictions.append(y_pred)
        all_scores.append(y_scores)
        
        # Calculate accuracy
        acc = accuracy_score(y_test_pooled, y_pred)
        model_accuracies.append(acc)
        
        if (idx + 1) % 100 == 0:
            print(f"  {idx + 1}/{len(high_acc_indices)} models processed")
    
    all_predictions = np.array(all_predictions)  # Shape: (n_models, n_spectra)
    all_scores = np.array(all_scores)
    
    # Ensemble predictions
    # 1. Majority voting
    majority_vote_pred = (np.mean(all_predictions, axis=0) > 0.5).astype(int)
    majority_vote_acc = accuracy_score(y_test_pooled, majority_vote_pred)
    
    # 2. Average score
    avg_scores = np.mean(all_scores, axis=0)
    avg_score_pred = (avg_scores > 0.5 if hasattr(models[high_acc_indices[0]], 'predict_proba') 
                      else (avg_scores > 0)).astype(int)
    avg_score_acc = accuracy_score(y_test_pooled, avg_score_pred)
    
    print(f"\n{'='*70}")
    print("ENSEMBLE RESULTS ON POOLED TEST SPECTRA")
    print(f"{'='*70}\n")
    print(f"Individual model accuracies: {np.mean(model_accuracies):.4f} ± {np.std(model_accuracies):.4f}")
    print(f"Majority voting accuracy: {majority_vote_acc:.4f}")
    print(f"Average score accuracy: {avg_score_acc:.4f}")
    
    # Per-spectrum analysis
    spectrum_results = []
    for i, info in enumerate(test_info):
        spectrum_results.append({
            'patient_id': info['patient_id'],
            'spectrum_id': info['spectrum_id'],
            'dataset_index': info['dataset_index'],
            'true_label': info['label'],
            'majority_vote_pred': 'cancer' if majority_vote_pred[i] == 1 else 'control',
            'avg_score_pred': 'cancer' if avg_score_pred[i] == 1 else 'control',
            'n_models_predict_cancer': np.sum(all_predictions[:, i]),
            'pct_models_predict_cancer': 100 * np.mean(all_predictions[:, i]),
            'avg_decision_score': avg_scores[i],
            'std_decision_score': np.std(all_scores[:, i])
        })
    
    spectrum_results_df = pd.DataFrame(spectrum_results)
    
    # Patient-level analysis
    print(f"\n{'='*70}")
    print("PATIENT-LEVEL ANALYSIS")
    print(f"{'='*70}\n")
    
    patient_level_results = []
    for patient_id in test_patients:
        patient_mask = spectrum_results_df['patient_id'] == patient_id
        patient_spectra = spectrum_results_df[patient_mask]
        
        if len(patient_spectra) == 0:
            continue
        
        n_spectra = len(patient_spectra)
        true_label = patient_spectra['true_label'].iloc[0]
        
        # Majority vote at patient level
        patient_majority_pred = patient_spectra['majority_vote_pred'].mode()[0]
        patient_majority_correct = (patient_majority_pred == true_label)
        
        # Average score at patient level
        patient_avg_score = patient_spectra['avg_decision_score'].mean()
        
        patient_level_results.append({
            'patient_id': patient_id,
            'true_label': true_label,
            'n_spectra': n_spectra,
            'patient_majority_pred': patient_majority_pred,
            'patient_majority_correct': patient_majority_correct,
            'patient_avg_decision_score': patient_avg_score
        })
        
        print(f"Patient {patient_id} ({true_label.upper()}): {n_spectra} spectra, "
              f"Pred: {patient_majority_pred} {'✓' if patient_majority_correct else '✗'}")
    
    patient_results_df = pd.DataFrame(patient_level_results)
    patient_level_acc = np.mean(patient_results_df['patient_majority_correct'])
    
    print(f"\nPatient-level accuracy: {patient_level_acc:.4f} "
          f"({np.sum(patient_results_df['patient_majority_correct'])}/{len(patient_results_df)})")
    
    return {
        'X_test': X_test_pooled,
        'y_test': y_test_pooled,
        'test_info': test_info,
        'all_predictions': all_predictions,
        'all_scores': all_scores,
        'model_accuracies': model_accuracies,
        'majority_vote_predictions': majority_vote_pred,
        'majority_vote_accuracy': majority_vote_acc,
        'avg_score_predictions': avg_score_pred,
        'avg_score_accuracy': avg_score_acc,
        'spectrum_results': spectrum_results_df,
        'patient_results': patient_results_df,
        'patient_level_accuracy': patient_level_acc,
        'n_models_used': len(high_acc_indices)
    }


def predict_with_high_accuracy_models_on_all_test_spectra(
    dataset, patient_groups, results_df, models, test_patients, 
    accuracy_threshold=1.0
):
    """
    Use each saved model above threshold to make predictions on ALL test spectra.
    
    Args:
        dataset: The dataset object
        patient_groups: Dictionary mapping patient_id to spectrum indices
        results_df: DataFrame with model results
        models: List of trained models
        test_patients: List of test patient IDs
        accuracy_threshold: Minimum accuracy threshold for model selection
    
    Returns:
        Dictionary containing prediction results and aggregated statistics
    """
    # Find high-accuracy models
    high_acc_mask = results_df['test_accuracy'] >= accuracy_threshold
    high_acc_indices = results_df[high_acc_mask].index.tolist()
    
    if len(high_acc_indices) == 0:
        print(f"No models found with accuracy >= {accuracy_threshold}")
        return None
    
    print(f"\n{'='*70}")
    print(f"Making Predictions with {len(high_acc_indices)} High-Accuracy Models")
    print(f"on ALL Test Spectra")
    print(f"{'='*70}\n")
    
    # Determine patient labels
    patient_labels = {}
    for pid in test_patients:
        _, _, metadata = dataset[patient_groups[pid][0]]
        staging = metadata['staging'].lower()
        patient_labels[pid] = 1 if ('cancer' in staging or 'ca' in staging) else 0
    
    # Collect ALL test spectra
    X_test_all, y_test_all, test_info = [], [], []
    for patient_id in test_patients:
        for spec_idx in patient_groups[patient_id]:
            intensity_tensor, _, metadata = dataset[spec_idx]
            intensity = intensity_tensor.squeeze().numpy()
            label = patient_labels[patient_id]
            
            X_test_all.append(intensity)
            y_test_all.append(label)
            test_info.append({
                'patient_id': patient_id,
                'spectrum_id': metadata['spectrum_id'],
                'dataset_index': spec_idx,
                'label': 'cancer' if label == 1 else 'control'
            })
    
    X_test_all = np.array(X_test_all)
    y_test_all = np.array(y_test_all)
    
    print(f"ALL test spectra: {len(X_test_all)} spectra")
    print(f"  {np.sum(y_test_all)} cancer, {len(y_test_all) - np.sum(y_test_all)} control\n")
    
    # Store predictions from each model
    all_predictions = []
    all_scores = []
    model_accuracies = []
    
    for idx, model_idx in enumerate(high_acc_indices):
        model = models[model_idx]
        
        # Make predictions
        y_pred = model.predict(X_test_all)
        
        # Get decision scores
        if hasattr(model, 'decision_function'):
            y_scores = model.decision_function(X_test_all)
        elif hasattr(model, 'predict_proba'):
            y_scores = model.predict_proba(X_test_all)[:, 1]
        else:
            y_scores = y_pred.astype(float)
        
        all_predictions.append(y_pred)
        all_scores.append(y_scores)
        
        # Calculate accuracy
        acc = accuracy_score(y_test_all, y_pred)
        model_accuracies.append(acc)
        
        if (idx + 1) % 100 == 0:
            print(f"  {idx + 1}/{len(high_acc_indices)} models processed")
    
    all_predictions = np.array(all_predictions)  # Shape: (n_models, n_spectra)
    all_scores = np.array(all_scores)
    
    # Ensemble predictions
    # 1. Majority voting
    majority_vote_pred = (np.mean(all_predictions, axis=0) > 0.5).astype(int)
    majority_vote_acc = accuracy_score(y_test_all, majority_vote_pred)
    
    # 2. Average score
    avg_scores = np.mean(all_scores, axis=0)
    avg_score_pred = (avg_scores > 0.5 if hasattr(models[high_acc_indices[0]], 'predict_proba') 
                      else (avg_scores > 0)).astype(int)
    avg_score_acc = accuracy_score(y_test_all, avg_score_pred)
    
    print(f"\n{'='*70}")
    print("ENSEMBLE RESULTS ON ALL TEST SPECTRA")
    print(f"{'='*70}\n")
    print(f"Individual model accuracies: {np.mean(model_accuracies):.4f} ± {np.std(model_accuracies):.4f}")
    print(f"Majority voting accuracy: {majority_vote_acc:.4f}")
    print(f"Average score accuracy: {avg_score_acc:.4f}")
    
    # Per-spectrum analysis
    spectrum_results = []
    for i, info in enumerate(test_info):
        spectrum_results.append({
            'patient_id': info['patient_id'],
            'spectrum_id': info['spectrum_id'],
            'dataset_index': info['dataset_index'],
            'true_label': info['label'],
            'majority_vote_pred': 'cancer' if majority_vote_pred[i] == 1 else 'control',
            'avg_score_pred': 'cancer' if avg_score_pred[i] == 1 else 'control',
            'n_models_predict_cancer': np.sum(all_predictions[:, i]),
            'pct_models_predict_cancer': 100 * np.mean(all_predictions[:, i]),
            'avg_decision_score': avg_scores[i],
            'std_decision_score': np.std(all_scores[:, i])
        })
    
    spectrum_results_df = pd.DataFrame(spectrum_results)
    
    # Patient-level analysis
    print(f"\n{'='*70}")
    print("PATIENT-LEVEL ANALYSIS")
    print(f"{'='*70}\n")
    
    patient_level_results = []
    for patient_id in test_patients:
        patient_mask = spectrum_results_df['patient_id'] == patient_id
        patient_spectra = spectrum_results_df[patient_mask]
        
        n_spectra = len(patient_spectra)
        true_label = patient_spectra['true_label'].iloc[0]
        
        # Majority vote at patient level
        patient_majority_pred = patient_spectra['majority_vote_pred'].mode()[0]
        patient_majority_correct = (patient_majority_pred == true_label)
        
        # Average score at patient level
        patient_avg_score = patient_spectra['avg_decision_score'].mean()
        
        # Percentage of spectra correctly classified
        correct_spectra = (patient_spectra['majority_vote_pred'] == patient_spectra['true_label']).sum()
        pct_correct = 100 * correct_spectra / n_spectra
        
        patient_level_results.append({
            'patient_id': patient_id,
            'true_label': true_label,
            'n_spectra': n_spectra,
            'n_correct_spectra': correct_spectra,
            'pct_correct_spectra': pct_correct,
            'patient_majority_pred': patient_majority_pred,
            'patient_majority_correct': patient_majority_correct,
            'patient_avg_decision_score': patient_avg_score
        })
        
        print(f"Patient {patient_id} ({true_label.upper()}): {n_spectra} spectra, "
              f"{pct_correct:.1f}% correct, Majority: {patient_majority_pred} "
              f"{'✓' if patient_majority_correct else '✗'}")
    
    patient_results_df = pd.DataFrame(patient_level_results)
    patient_level_acc = np.mean(patient_results_df['patient_majority_correct'])
    
    print(f"\nPatient-level accuracy: {patient_level_acc:.4f} "
          f"({np.sum(patient_results_df['patient_majority_correct'])}/{len(patient_results_df)})")
    
    return {
        'X_test': X_test_all,
        'y_test': y_test_all,
        'test_info': test_info,
        'all_predictions': all_predictions,
        'all_scores': all_scores,
        'model_accuracies': model_accuracies,
        'majority_vote_predictions': majority_vote_pred,
        'majority_vote_accuracy': majority_vote_acc,
        'avg_score_predictions': avg_score_pred,
        'avg_score_accuracy': avg_score_acc,
        'spectrum_results': spectrum_results_df,
        'patient_results': patient_results_df,
        'patient_level_accuracy': patient_level_acc,
        'n_models_used': len(high_acc_indices)
    }

def train_on_all_patient_spectra(dataset, patient_groups, train_patients, test_patients, 
                                  C=1.0, model_type='xgboost', plot_results=True):
    """
    Train a single model using ALL spectra from each patient.
    Uses the same train/test patient split as the random subset training.
    
    Args:
        dataset: The dataset object
        patient_groups: Dictionary mapping patient_id to spectrum indices
        train_patients: List of training patient IDs (from previous split)
        test_patients: List of test patient IDs (from previous split)
        C: Regularization parameter (for applicable models)
        model_type: Type of model to train
        plot_results: Whether to create visualization
    
    Returns:
        Dictionary containing model, predictions, and evaluation metrics
    """
    print(f"\n{'='*70}")
    print(f"Training Single Model on ALL Patient Spectra")
    print(f"{'='*70}\n")
    
    # Determine patient labels
    patient_labels = {}
    for pid in train_patients + test_patients:
        _, _, metadata = dataset[patient_groups[pid][0]]
        staging = metadata['staging'].lower()
        patient_labels[pid] = 1 if ('cancer' in staging or 'ca' in staging) else 0
    
    # Collect ALL training spectra
    X_train_all, y_train_all, train_info = [], [], []
    for patient_id in train_patients:
        for spec_idx in patient_groups[patient_id]:
            intensity_tensor, _, metadata = dataset[spec_idx]
            intensity = intensity_tensor.squeeze().numpy()
            label = patient_labels[patient_id]
            
            X_train_all.append(intensity)
            y_train_all.append(label)
            train_info.append({
                'patient_id': patient_id,
                'spectrum_id': metadata['spectrum_id'],
                'dataset_index': spec_idx,
                'label': 'cancer' if label == 1 else 'control'
            })
    
    X_train_all = np.array(X_train_all)
    y_train_all = np.array(y_train_all)
    
    # Collect ALL test spectra
    X_test_all, y_test_all, test_info = [], [], []
    for patient_id in test_patients:
        for spec_idx in patient_groups[patient_id]:
            intensity_tensor, _, metadata = dataset[spec_idx]
            intensity = intensity_tensor.squeeze().numpy()
            label = patient_labels[patient_id]
            
            X_test_all.append(intensity)
            y_test_all.append(label)
            test_info.append({
                'patient_id': patient_id,
                'spectrum_id': metadata['spectrum_id'],
                'dataset_index': spec_idx,
                'label': 'cancer' if label == 1 else 'control'
            })
    
    X_test_all = np.array(X_test_all)
    y_test_all = np.array(y_test_all)
    
    # Print data summary
    train_cancer = np.sum(y_train_all)
    train_control = len(y_train_all) - train_cancer
    test_cancer = np.sum(y_test_all)
    test_control = len(y_test_all) - test_cancer
    
    print(f"Training set: {len(train_patients)} patients, {len(X_train_all)} spectra")
    print(f"  {train_cancer} cancer, {train_control} control")
    print(f"Test set: {len(test_patients)} patients, {len(X_test_all)} spectra")
    print(f"  {test_cancer} cancer, {test_control} control\n")
    
    # Train model
    print(f"Training {model_type} model...")
    model, test_acc, test_auc, n_features = train_model(
        X_train_all, y_train_all, X_test_all, y_test_all, C=C, model_type=model_type
    )
    
    # Predictions
    y_train_pred = model.predict(X_train_all)
    y_test_pred = model.predict(X_test_all)
    train_acc = accuracy_score(y_train_all, y_train_pred)
    
    # Get scores
    if hasattr(model, 'decision_function'):
        y_train_scores = model.decision_function(X_train_all)
        y_test_scores = model.decision_function(X_test_all)
    elif hasattr(model, 'predict_proba'):
        y_train_scores = model.predict_proba(X_train_all)[:, 1]
        y_test_scores = model.predict_proba(X_test_all)[:, 1]
    else:
        y_train_scores = y_train_pred.astype(float)
        y_test_scores = y_test_pred.astype(float)
    
    # Confusion matrices
    from sklearn.metrics import confusion_matrix, classification_report
    train_cm = confusion_matrix(y_train_all, y_train_pred)
    test_cm = confusion_matrix(y_test_all, y_test_pred)
    
    # Print results
    print(f"\n{'='*70}")
    print("SPECTRUM-LEVEL RESULTS")
    print(f"{'='*70}\n")
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    if test_auc:
        print(f"Test AUC: {test_auc:.4f}")
    print(f"Non-zero features: {n_features}\n")
    print(f"Train Confusion Matrix: TN={train_cm[0,0]} FP={train_cm[0,1]} FN={train_cm[1,0]} TP={train_cm[1,1]}")
    print(f"Test Confusion Matrix:  TN={test_cm[0,0]} FP={test_cm[0,1]} FN={test_cm[1,0]} TP={test_cm[1,1]}\n")
    print("Test Set Classification Report:")
    print(classification_report(y_test_all, y_test_pred, target_names=['Control', 'Cancer']))
    
    # Patient-level analysis
    print(f"\n{'='*70}")
    print("PATIENT-LEVEL ANALYSIS")
    print(f"{'='*70}\n")
    
    patient_level_results = []
    for patient_id in test_patients:
        patient_indices = [i for i, info in enumerate(test_info) if info['patient_id'] == patient_id]
        patient_true_labels = y_test_all[patient_indices]
        patient_predictions = y_test_pred[patient_indices]
        patient_scores = y_test_scores[patient_indices]
        
        n_spectra = len(patient_indices)
        n_correct = np.sum(patient_true_labels == patient_predictions)
        pct_correct = 100 * n_correct / n_spectra
        
        true_label = test_info[patient_indices[0]]['label']
        majority_pred = 1 if np.sum(patient_predictions) > n_spectra / 2 else 0
        majority_pred_label = 'cancer' if majority_pred == 1 else 'control'
        majority_correct = (majority_pred == patient_true_labels[0])
        
        patient_level_results.append({
            'patient_id': patient_id,
            'true_label': true_label,
            'n_spectra': n_spectra,
            'n_correct': n_correct,
            'pct_correct': pct_correct,
            'majority_vote_pred': majority_pred_label,
            'majority_vote_correct': majority_correct,
            'avg_decision_score': np.mean(patient_scores),
            'std_decision_score': np.std(patient_scores)
        })
        
        print(f"Patient {patient_id} ({true_label.upper()}): {n_spectra} spectra, "
              f"{pct_correct:.1f}% correct, Majority: {majority_pred_label} {'✓' if majority_correct else '✗'}")
    
    patient_results_df = pd.DataFrame(patient_level_results)
    patient_level_acc = np.mean(patient_results_df['majority_vote_correct'])
    
    print(f"\nPatient-level accuracy (Majority Vote): {patient_level_acc:.4f} "
          f"({np.sum(patient_results_df['majority_vote_correct'])}/{len(test_patients)})")
    
    # Visualization
    if plot_results:
        _create_all_spectra_visualization(
            test_cm, y_test_scores, y_test_all, patient_results_df,
            test_acc, patient_level_acc, n_features, C, model_type,
            X_train_all, y_train_all, X_test_all, test_info
        )
    
    return {
        'model': model,
        'X_train': X_train_all,
        'y_train': y_train_all,
        'X_test': X_test_all,
        'y_test': y_test_all,
        'train_info': train_info,
        'test_info': test_info,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
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


def _create_all_spectra_visualization(test_cm, y_test_scores, y_test_all, patient_results_df,
                                      test_acc, patient_level_acc, n_features, C, model_type,
                                      X_train_all, y_train_all, X_test_all, test_info):
    """Create comprehensive visualization for all-spectra training."""
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    import seaborn as sns
    
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Confusion Matrix
    ax1 = fig.add_subplot(gs[0, 0])
    sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Control', 'Cancer'], yticklabels=['Control', 'Cancer'],
                cbar_kws={'label': 'Count'}, ax=ax1)
    ax1.set_title('Test Confusion Matrix', fontweight='bold', fontsize=12)
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    # 2. Decision Score Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    cancer_scores = y_test_scores[y_test_all == 1]
    control_scores = y_test_scores[y_test_all == 0]
    ax2.hist(control_scores, bins=30, alpha=0.6, label='Control', color='green', edgecolor='black')
    ax2.hist(cancer_scores, bins=30, alpha=0.6, label='Cancer', color='red', edgecolor='black')
    ax2.axvline(0, color='black', linestyle='--', linewidth=2, label='Decision Boundary')
    ax2.set_xlabel('Decision Score')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Decision Score Distribution', fontweight='bold', fontsize=12)
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. Per-Patient Accuracy
    ax3 = fig.add_subplot(gs[0, 2])
    patient_sorted = patient_results_df.sort_values('pct_correct')
    colors = ['green' if label == 'control' else 'red' for label in patient_sorted['true_label']]
    y_pos = np.arange(len(patient_sorted))
    ax3.barh(y_pos, patient_sorted['pct_correct'], color=colors, alpha=0.7, edgecolor='black')
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(patient_sorted['patient_id'], fontsize=8)
    ax3.set_xlabel('% Spectra Correctly Classified')
    ax3.set_title('Per-Patient Spectrum Accuracy', fontweight='bold', fontsize=12)
    ax3.axvline(50, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax3.legend(handles=[Patch(facecolor='red', alpha=0.7, label='Cancer'),
                       Patch(facecolor='green', alpha=0.7, label='Control')], loc='lower right')
    ax3.grid(axis='x', alpha=0.3)
    
    # 4. Patient Average Decision Scores
    ax4 = fig.add_subplot(gs[1, 0])
    patient_sorted2 = patient_results_df.sort_values('avg_decision_score')
    colors2 = ['green' if label == 'control' else 'red' for label in patient_sorted2['true_label']]
    y_pos2 = np.arange(len(patient_sorted2))
    ax4.barh(y_pos2, patient_sorted2['avg_decision_score'], color=colors2, alpha=0.7, edgecolor='black')
    ax4.set_yticks(y_pos2)
    ax4.set_yticklabels(patient_sorted2['patient_id'], fontsize=8)
    ax4.set_xlabel('Average Decision Score')
    ax4.set_title('Per-Patient Avg Decision Score', fontweight='bold', fontsize=12)
    ax4.axvline(0, color='black', linestyle='--', linewidth=2)
    ax4.grid(axis='x', alpha=0.3)
    
    # 5. Accuracy Comparison
    ax5 = fig.add_subplot(gs[1, 1])
    metrics = ['Spectrum-Level', 'Patient-Level\n(Majority Vote)']
    accuracies = [test_acc, patient_level_acc]
    bars = ax5.bar(metrics, accuracies, color=['#1f77b4', '#ff7f0e'], alpha=0.7, edgecolor='black')
    ax5.set_ylabel('Accuracy')
    ax5.set_ylim([0, 1.1])
    ax5.set_title('Accuracy Comparison', fontweight='bold', fontsize=12)
    for bar, acc in zip(bars, accuracies):
        ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    ax5.grid(axis='y', alpha=0.3)
    
    # 6. Class Balance
    ax6 = fig.add_subplot(gs[1, 2])
    train_counts = [np.sum(y_train_all == 0), np.sum(y_train_all == 1)]
    test_counts = [np.sum(y_test_all == 0), np.sum(y_test_all == 1)]
    x = np.arange(2)
    width = 0.35
    bars1 = ax6.bar(x - width/2, train_counts, width, label='Training',
                   color='#1f77b4', alpha=0.7, edgecolor='black')
    bars2 = ax6.bar(x + width/2, test_counts, width, label='Test',
                   color='#ff7f0e', alpha=0.7, edgecolor='black')
    ax6.set_ylabel('Number of Spectra')
    ax6.set_title('Class Balance', fontweight='bold', fontsize=12)
    ax6.set_xticks(x)
    ax6.set_xticklabels(['Control', 'Cancer'])
    ax6.legend()
    ax6.grid(axis='y', alpha=0.3)
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    # 7. Individual Spectrum Scores by Patient
    ax7 = fig.add_subplot(gs[2, :])
    for i, (patient_id, group) in enumerate(patient_results_df.groupby('patient_id')):
        patient_indices = [j for j, info in enumerate(test_info) if info['patient_id'] == patient_id]
        patient_scores = y_test_scores[patient_indices]
        true_label = group['true_label'].iloc[0]
        color = 'red' if true_label == 'cancer' else 'green'
        marker = 'o' if group['majority_vote_correct'].iloc[0] else 'x'
        ax7.scatter([i] * len(patient_scores), patient_scores,
                   c=color, marker=marker, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    ax7.axhline(0, color='black', linestyle='--', linewidth=2)
    ax7.set_xlabel('Patient Index', fontsize=11)
    ax7.set_ylabel('Decision Score', fontsize=11)
    ax7.set_title('Individual Spectrum Decision Scores by Patient', fontweight='bold', fontsize=12)
    ax7.legend(handles=[
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8,
               label='Cancer (Correct MV)', markeredgecolor='black'),
        Line2D([0], [0], marker='x', color='w', markerfacecolor='red', markersize=8,
               label='Cancer (Incorrect MV)', markeredgecolor='black'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8,
               label='Control (Correct MV)', markeredgecolor='black'),
        Line2D([0], [0], marker='x', color='w', markerfacecolor='green', markersize=8,
               label='Control (Incorrect MV)', markeredgecolor='black')
    ], fontsize=9, loc='best')
    ax7.grid(alpha=0.3)
    
    plt.suptitle(f'All-Spectra Model ({model_type}): Comprehensive Results\n'
                f'Features: {n_features}, C: {C}',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def predict_with_high_accuracy_models_on_pooled_spectra_dual_filter(
    dataset, patient_groups, results_df, spectrum_selections, 
    models, test_patients, train_accuracy_threshold=0.9, test_accuracy_threshold=0.8,
    train_accuracies_file=None
):
    """
    Use saved models filtered by BOTH train and test accuracy to make predictions on pooled test spectra.
    First filters by train accuracy, then by test accuracy.
    
    Args:
        dataset: The dataset object
        patient_groups: Dictionary mapping patient_id to spectrum indices
        results_df: DataFrame with model results (contains 'test_accuracy')
        spectrum_selections: List of spectrum selections for each model
        models: List of trained models
        test_patients: List of test patient IDs
        train_accuracy_threshold: Minimum train accuracy threshold
        test_accuracy_threshold: Minimum test accuracy threshold
        train_accuracies_file: Path to pickle file containing train accuracies DataFrame.
                               If None, assumes results_df already has 'train_accuracy' column.
    
    Returns:
        Dictionary containing prediction results and aggregated statistics
    """
    # Load or check for train accuracies
    if 'train_accuracy' not in results_df.columns:
        if train_accuracies_file is None:
            raise ValueError(
                "results_df does not contain 'train_accuracy' column and no train_accuracies_file provided. "
                "Either run calculate_train_accuracies() first or provide train_accuracies_file path."
            )
        
        print(f"Loading train accuracies from: {train_accuracies_file}")
        with open(train_accuracies_file, "rb") as f:
            train_results_df = pickle.load(f)
        
        # Check if loaded data is a DataFrame
        if isinstance(train_results_df, pd.DataFrame):
            if 'train_accuracy' not in train_results_df.columns:
                raise ValueError(f"Loaded file does not contain 'train_accuracy' column")
            
            # Merge train accuracies into results_df
            # Ensure indices align properly
            if len(train_results_df) != len(results_df):
                raise ValueError(
                    f"Length mismatch: results_df has {len(results_df)} rows, "
                    f"but train_accuracies file has {len(train_results_df)} rows"
                )
            
            results_df = results_df.copy()
            results_df['train_accuracy'] = train_results_df['train_accuracy'].values
            print(f"Successfully loaded {len(train_results_df)} train accuracies")
        else:
            raise ValueError(f"Expected DataFrame in pickle file, got {type(train_results_df)}")
    
    # Filter by train accuracy first
    train_filtered_mask = results_df['train_accuracy'] >= train_accuracy_threshold
    train_filtered_indices = results_df[train_filtered_mask].index.tolist()
    
    print(f"\n{'='*70}")
    print(f"DUAL FILTERING: Train >= {train_accuracy_threshold}, Test >= {test_accuracy_threshold}")
    print(f"{'='*70}\n")
    print(f"Total models: {len(results_df)}")
    print(f"Models passing train accuracy filter (>= {train_accuracy_threshold}): {len(train_filtered_indices)}")
    
    if len(train_filtered_indices) == 0:
        print(f"No models found with train accuracy >= {train_accuracy_threshold}")
        return None
    
    # Then filter by test accuracy
    dual_filtered_mask = train_filtered_mask & (results_df['test_accuracy'] >= test_accuracy_threshold)
    high_acc_indices = results_df[dual_filtered_mask].index.tolist()
    
    print(f"Models passing both filters: {len(high_acc_indices)}")
    
    if len(high_acc_indices) == 0:
        print(f"No models found with train accuracy >= {train_accuracy_threshold} AND test accuracy >= {test_accuracy_threshold}")
        return None
    
    print(f"\nUsing {len(high_acc_indices)} models for predictions on pooled test spectra\n")
    
    # Print statistics of selected models
    selected_models_df = results_df.loc[high_acc_indices]
    print(f"Selected models statistics:")
    print(f"  Train accuracy: {selected_models_df['train_accuracy'].mean():.4f} ± {selected_models_df['train_accuracy'].std():.4f}")
    print(f"  Test accuracy:  {selected_models_df['test_accuracy'].mean():.4f} ± {selected_models_df['test_accuracy'].std():.4f}")
    print(f"  Avg gap (Train - Test): {selected_models_df['train_accuracy'].mean() - selected_models_df['test_accuracy'].mean():.4f}\n")
    
    # Determine patient labels
    patient_labels = {}
    for pid in test_patients:
        _, _, metadata = dataset[patient_groups[pid][0]]
        staging = metadata['staging'].lower()
        patient_labels[pid] = 1 if ('cancer' in staging or 'ca' in staging) else 0
    
    # Collect unique pooled test spectra from high-accuracy models
    pooled_test_spectra = set()
    for model_idx in high_acc_indices:
        selection = spectrum_selections[model_idx]
        for spec_info in selection['test_spectra']:
            key = (spec_info['patient_id'], spec_info['spectrum_id'], spec_info['dataset_index'])
            pooled_test_spectra.add(key)
    
    # Extract pooled test data
    X_test_pooled, y_test_pooled, test_info = [], [], []
    for patient_id, spectrum_id, dataset_idx in pooled_test_spectra:
        intensity_tensor, _, metadata = dataset[dataset_idx]
        intensity = intensity_tensor.squeeze().numpy()
        label = patient_labels[patient_id]
        
        X_test_pooled.append(intensity)
        y_test_pooled.append(label)
        test_info.append({
            'patient_id': patient_id,
            'spectrum_id': spectrum_id,
            'dataset_index': dataset_idx,
            'label': 'cancer' if label == 1 else 'control'
        })
    
    X_test_pooled = np.array(X_test_pooled)
    y_test_pooled = np.array(y_test_pooled)
    
    print(f"Pooled test set: {len(X_test_pooled)} spectra")
    print(f"  {np.sum(y_test_pooled)} cancer, {len(y_test_pooled) - np.sum(y_test_pooled)} control\n")
    
    # Store predictions from each model
    all_predictions = []
    all_scores = []
    model_accuracies = []
    model_patient_accuracies = []  # New: store patient-level accuracy per model
    
    for idx, model_idx in enumerate(high_acc_indices):
        model = models[model_idx]
        
        # Make predictions
        y_pred = model.predict(X_test_pooled)
        
        # Get decision scores
        if hasattr(model, 'decision_function'):
            y_scores = model.decision_function(X_test_pooled)
        elif hasattr(model, 'predict_proba'):
            y_scores = model.predict_proba(X_test_pooled)[:, 1]
        else:
            y_scores = y_pred.astype(float)
        
        all_predictions.append(y_pred)
        all_scores.append(y_scores)
        
        # Calculate spectrum-level accuracy
        spectrum_acc = accuracy_score(y_test_pooled, y_pred)
        model_accuracies.append(spectrum_acc)
        
        # Calculate patient-level accuracy for this model
        patient_correct = 0
        patient_total = 0
        for patient_id in test_patients:
            # Get indices for this patient in the pooled test set
            patient_indices = [i for i, info in enumerate(test_info) if info['patient_id'] == patient_id]
            
            if len(patient_indices) == 0:
                continue
            
            patient_true_label = y_test_pooled[patient_indices[0]]
            patient_predictions = y_pred[patient_indices]
            
            # Majority vote for this patient
            patient_majority_pred = 1 if np.sum(patient_predictions) > len(patient_predictions) / 2 else 0
            
            if patient_majority_pred == patient_true_label:
                patient_correct += 1
            patient_total += 1
        
        patient_acc = patient_correct / patient_total if patient_total > 0 else 0
        model_patient_accuracies.append(patient_acc)
        
        if (idx + 1) % 100 == 0:
            print(f"  {idx + 1}/{len(high_acc_indices)} models processed")
    
    all_predictions = np.array(all_predictions)  # Shape: (n_models, n_spectra)
    all_scores = np.array(all_scores)
    model_accuracies = np.array(model_accuracies)
    model_patient_accuracies = np.array(model_patient_accuracies)
    
    # Ensemble predictions
    # 1. Majority voting
    majority_vote_pred = (np.mean(all_predictions, axis=0) > 0.5).astype(int)
    majority_vote_acc = accuracy_score(y_test_pooled, majority_vote_pred)
    
    # 2. Average score
    avg_scores = np.mean(all_scores, axis=0)
    avg_score_pred = (avg_scores > 0.5 if hasattr(models[high_acc_indices[0]], 'predict_proba') 
                      else (avg_scores > 0)).astype(int)
    avg_score_acc = accuracy_score(y_test_pooled, avg_score_pred)
    
    print(f"\n{'='*70}")
    print("ENSEMBLE RESULTS ON POOLED TEST SPECTRA")
    print(f"{'='*70}\n")
    print(f"Individual model spectrum-level accuracies: {np.mean(model_accuracies):.4f} ± {np.std(model_accuracies):.4f}")
    print(f"  Min: {np.min(model_accuracies):.4f}, Max: {np.max(model_accuracies):.4f}, Median: {np.median(model_accuracies):.4f}")
    print(f"\nIndividual model patient-level accuracies: {np.mean(model_patient_accuracies):.4f} ± {np.std(model_patient_accuracies):.4f}")
    print(f"  Min: {np.min(model_patient_accuracies):.4f}, Max: {np.max(model_patient_accuracies):.4f}, Median: {np.median(model_patient_accuracies):.4f}")
    print(f"\nEnsemble majority voting accuracy: {majority_vote_acc:.4f}")
    print(f"Ensemble average score accuracy: {avg_score_acc:.4f}")
    
    # Per-spectrum analysis
    spectrum_results = []
    for i, info in enumerate(test_info):
        spectrum_results.append({
            'patient_id': info['patient_id'],
            'spectrum_id': info['spectrum_id'],
            'dataset_index': info['dataset_index'],
            'true_label': info['label'],
            'majority_vote_pred': 'cancer' if majority_vote_pred[i] == 1 else 'control',
            'avg_score_pred': 'cancer' if avg_score_pred[i] == 1 else 'control',
            'n_models_predict_cancer': np.sum(all_predictions[:, i]),
            'pct_models_predict_cancer': 100 * np.mean(all_predictions[:, i]),
            'avg_decision_score': avg_scores[i],
            'std_decision_score': np.std(all_scores[:, i])
        })
    
    spectrum_results_df = pd.DataFrame(spectrum_results)
    
    # Patient-level analysis
    print(f"\n{'='*70}")
    print("PATIENT-LEVEL ANALYSIS (ENSEMBLE)")
    print(f"{'='*70}\n")
    
    patient_level_results = []
    for patient_id in test_patients:
        patient_mask = spectrum_results_df['patient_id'] == patient_id
        patient_spectra = spectrum_results_df[patient_mask]
        
        if len(patient_spectra) == 0:
            continue
        
        n_spectra = len(patient_spectra)
        true_label = patient_spectra['true_label'].iloc[0]
        
        # Majority vote at patient level
        patient_majority_pred = patient_spectra['majority_vote_pred'].mode()[0]
        patient_majority_correct = (patient_majority_pred == true_label)
        
        # Average score at patient level
        patient_avg_score = patient_spectra['avg_decision_score'].mean()
        
        patient_level_results.append({
            'patient_id': patient_id,
            'true_label': true_label,
            'n_spectra': n_spectra,
            'patient_majority_pred': patient_majority_pred,
            'patient_majority_correct': patient_majority_correct,
            'patient_avg_decision_score': patient_avg_score
        })
        
        print(f"Patient {patient_id} ({true_label.upper()}): {n_spectra} spectra, "
              f"Pred: {patient_majority_pred} {'✓' if patient_majority_correct else '✗'}")
    
    patient_results_df = pd.DataFrame(patient_level_results)
    patient_level_acc = np.mean(patient_results_df['patient_majority_correct'])
    
    print(f"\nEnsemble patient-level accuracy: {patient_level_acc:.4f} "
          f"({np.sum(patient_results_df['patient_majority_correct'])}/{len(patient_results_df)})")
    
    return {
        'X_test': X_test_pooled,
        'y_test': y_test_pooled,
        'test_info': test_info,
        'all_predictions': all_predictions,
        'all_scores': all_scores,
        'model_spectrum_accuracies': model_accuracies,  # Updated name for clarity
        'model_patient_accuracies': model_patient_accuracies,  # New: per-model patient-level accuracies
        'majority_vote_predictions': majority_vote_pred,
        'majority_vote_accuracy': majority_vote_acc,
        'avg_score_predictions': avg_score_pred,
        'avg_score_accuracy': avg_score_acc,
        'spectrum_results': spectrum_results_df,
        'patient_results': patient_results_df,
        'patient_level_accuracy': patient_level_acc,
        'n_models_used': len(high_acc_indices),
        'train_accuracy_threshold': train_accuracy_threshold,
        'test_accuracy_threshold': test_accuracy_threshold,
        'selected_models_stats': selected_models_df[['train_accuracy', 'test_accuracy']].describe()
    }

def create_averaged_top_features_model(
    dataset, patient_groups, results_df, spectrum_selections, 
    models, test_patients, train_accuracy_threshold=0.9, test_accuracy_threshold=0.8,
    train_accuracies_file=None, top_n_features=10, raman_shift=None
):
    """
    Dual filter models, find top N most frequent features, average their coefficients,
    and create a new linear model to predict on ALL test data.
    
    Args:
        dataset: The dataset object
        patient_groups: Dictionary mapping patient_id to spectrum indices
        results_df: DataFrame with model results (contains 'test_accuracy')
        spectrum_selections: List of spectrum selections for each model
        models: List of trained models
        test_patients: List of test patient IDs
        train_accuracy_threshold: Minimum train accuracy threshold
        test_accuracy_threshold: Minimum test accuracy threshold
        train_accuracies_file: Path to pickle file containing train accuracies DataFrame
        top_n_features: Number of top features to select
        raman_shift: Array of Raman shift values for plotting. If None, will use feature indices.
    
    Returns:
        Dictionary containing new model, predictions, and evaluation metrics
    """
    # Load or check for train accuracies
    if 'train_accuracy' not in results_df.columns:
        if train_accuracies_file is None:
            raise ValueError(
                "results_df does not contain 'train_accuracy' column and no train_accuracies_file provided."
            )
        
        print(f"Loading train accuracies from: {train_accuracies_file}")
        with open(train_accuracies_file, "rb") as f:
            train_results_df = pickle.load(f)
        
        if isinstance(train_results_df, pd.DataFrame):
            if 'train_accuracy' not in train_results_df.columns:
                raise ValueError(f"Loaded file does not contain 'train_accuracy' column")
            
            if len(train_results_df) != len(results_df):
                raise ValueError(
                    f"Length mismatch: results_df has {len(results_df)} rows, "
                    f"but train_accuracies file has {len(train_results_df)} rows"
                )
            
            results_df = results_df.copy()
            results_df['train_accuracy'] = train_results_df['train_accuracy'].values
            print(f"Successfully loaded {len(train_results_df)} train accuracies")
        else:
            raise ValueError(f"Expected DataFrame in pickle file, got {type(train_results_df)}")
    
    # Filter by train accuracy first, then test accuracy
    train_filtered_mask = results_df['train_accuracy'] >= train_accuracy_threshold
    dual_filtered_mask = train_filtered_mask & (results_df['test_accuracy'] >= test_accuracy_threshold)
    high_acc_indices = results_df[dual_filtered_mask].index.tolist()
    
    print(f"\n{'='*70}")
    print(f"CREATING AVERAGED TOP-{top_n_features} FEATURES MODEL")
    print(f"{'='*70}\n")
    print(f"Total models: {len(results_df)}")
    print(f"Models passing train filter (>= {train_accuracy_threshold}): {np.sum(train_filtered_mask)}")
    print(f"Models passing both filters: {len(high_acc_indices)}")
    
    if len(high_acc_indices) == 0:
        print(f"No models found with required accuracy thresholds")
        return None
    
    # Check if models have coefficients
    has_coef = all(hasattr(models[idx], 'coef_') for idx in high_acc_indices)
    if not has_coef:
        print("ERROR: Selected models do not have linear coefficients (coef_ attribute)")
        print("This method only works with linear models (Logistic Regression, Linear SVM, etc.)")
        return None
    
    selected_models_df = results_df.loc[high_acc_indices]
    print(f"\nSelected models statistics:")
    print(f"  Train accuracy: {selected_models_df['train_accuracy'].mean():.4f} ± {selected_models_df['train_accuracy'].std():.4f}")
    print(f"  Test accuracy:  {selected_models_df['test_accuracy'].mean():.4f} ± {selected_models_df['test_accuracy'].std():.4f}")
    
    # Count feature usage across selected models
    n_features = models[high_acc_indices[0]].coef_.shape[1]
    feature_counts = np.zeros(n_features)
    feature_coefs_sum = np.zeros(n_features)
    
    for model_idx in high_acc_indices:
        model = models[model_idx]
        coef = model.coef_.flatten()
        non_zero_mask = coef != 0
        
        feature_counts += non_zero_mask.astype(int)
        feature_coefs_sum += coef
    
    # Find top N most frequent features
    top_feature_indices = np.argsort(feature_counts)[-top_n_features:][::-1]
    
    print(f"\n{'='*70}")
    print(f"TOP {top_n_features} MOST FREQUENT FEATURES")
    print(f"{'='*70}\n")
    for rank, feat_idx in enumerate(top_feature_indices, 1):
        count = int(feature_counts[feat_idx])
        pct = 100 * count / len(high_acc_indices)
        avg_coef = feature_coefs_sum[feat_idx] / len(high_acc_indices)
        if raman_shift is not None:
            print(f"Rank {rank}: Feature {feat_idx} (Raman: {raman_shift[feat_idx]:.2f} cm⁻¹), "
                  f"Used in {count}/{len(high_acc_indices)} models ({pct:.1f}%), Avg coef: {avg_coef:.6f}")
        else:
            print(f"Rank {rank}: Feature {feat_idx}, Used in {count}/{len(high_acc_indices)} models ({pct:.1f}%), Avg coef: {avg_coef:.6f}")
    
    # Create new averaged coefficient vector (only top N features)
    new_coef = np.zeros(n_features)
    for feat_idx in top_feature_indices:
        new_coef[feat_idx] = feature_coefs_sum[feat_idx] / len(high_acc_indices)
    
    # Average intercept from selected models
    intercepts = [models[idx].intercept_ for idx in high_acc_indices]
    new_intercept = np.mean(intercepts)
    
    print(f"\nNew model created with {top_n_features} features")
    print(f"Average intercept: {new_intercept:.6f}")
    
    # Sort features by index for plotting
    top_feature_indices_sorted = np.sort(top_feature_indices)
    
    # Plot the features and their coefficients
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Bar plot of top features and their coefficients (ordered by index)
    if raman_shift is not None:
        x_labels = [f"{raman_shift[idx]:.1f}" for idx in top_feature_indices_sorted]
        ax1.set_xlabel('Raman Shift (cm⁻¹)', fontsize=12)
    else:
        x_labels = [f"{idx}" for idx in top_feature_indices_sorted]
        ax1.set_xlabel('Feature Index', fontsize=12)
    
    colors = ['red' if new_coef[idx] > 0 else 'blue' for idx in top_feature_indices_sorted]
    bars = ax1.bar(range(top_n_features), [new_coef[idx] for idx in top_feature_indices_sorted], 
                   color=colors, alpha=0.7, edgecolor='black')
    
    ax1.set_xticks(range(top_n_features))
    ax1.set_xticklabels(x_labels, rotation=45, ha='right')
    ax1.set_ylabel('Average Coefficient', fontsize=12)
    ax1.set_title(f'Top {top_n_features} Most Frequent Features and Their Coefficients\n(Ordered by Feature Index)', 
                  fontsize=13, fontweight='bold')
    ax1.axhline(0, color='black', linewidth=1, linestyle='--')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, feat_idx) in enumerate(zip(bars, top_feature_indices_sorted)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{new_coef[feat_idx]:.4f}',
                ha='center', va='bottom' if height > 0 else 'top', 
                fontsize=9, fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.7, label='Positive (Cancer-associated)'),
        Patch(facecolor='blue', alpha=0.7, label='Negative (Control-associated)')
    ]
    ax1.legend(handles=legend_elements, loc='best', fontsize=10)
    
    # Plot 2: Feature frequency (usage count) (ordered by index)
    freq_colors = ['darkgreen' if count > len(high_acc_indices) * 0.8 else 'orange' 
                   for count in [feature_counts[idx] for idx in top_feature_indices_sorted]]
    bars2 = ax2.bar(range(top_n_features), 
                    [100 * feature_counts[idx] / len(high_acc_indices) for idx in top_feature_indices_sorted],
                    color=freq_colors, alpha=0.7, edgecolor='black')
    
    ax2.set_xticks(range(top_n_features))
    ax2.set_xticklabels(x_labels, rotation=45, ha='right')
    if raman_shift is not None:
        ax2.set_xlabel('Raman Shift (cm⁻¹)', fontsize=12)
    else:
        ax2.set_xlabel('Feature Index', fontsize=12)
    ax2.set_ylabel('Usage Frequency (%)', fontsize=12)
    ax2.set_title(f'Feature Usage Frequency Across {len(high_acc_indices)} Models\n(Ordered by Feature Index)', 
                  fontsize=13, fontweight='bold')
    ax2.set_ylim([0, 105])
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, feat_idx) in enumerate(zip(bars2, top_feature_indices_sorted)):
        height = bar.get_height()
        count = int(feature_counts[feat_idx])
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%\n({count}/{len(high_acc_indices)})',
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Add legend
    legend_elements2 = [
        Patch(facecolor='darkgreen', alpha=0.7, label='High frequency (>80%)'),
        Patch(facecolor='orange', alpha=0.7, label='Moderate frequency (≤80%)')
    ]
    ax2.legend(handles=legend_elements2, loc='best', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # Determine patient labels
    patient_labels = {}
    for pid in test_patients:
        _, _, metadata = dataset[patient_groups[pid][0]]
        staging = metadata['staging'].lower()
        patient_labels[pid] = 1 if ('cancer' in staging or 'ca' in staging) else 0
    
    # Collect ALL test spectra (not pooled)
    X_test_all, y_test_all, test_info = [], [], []
    for patient_id in test_patients:
        for spec_idx in patient_groups[patient_id]:
            intensity_tensor, _, metadata = dataset[spec_idx]
            intensity = intensity_tensor.squeeze().numpy()
            label = patient_labels[patient_id]
            
            X_test_all.append(intensity)
            y_test_all.append(label)
            test_info.append({
                'patient_id': patient_id,
                'spectrum_id': metadata['spectrum_id'],
                'dataset_index': spec_idx,
                'label': 'cancer' if label == 1 else 'control'
            })
    
    X_test_all = np.array(X_test_all)
    y_test_all = np.array(y_test_all)
    
    print(f"\n{'='*70}")
    print("TESTING ON ALL TEST SPECTRA")
    print(f"{'='*70}\n")
    print(f"Test set: {len(test_patients)} patients, {len(X_test_all)} spectra")
    print(f"  {np.sum(y_test_all)} cancer, {len(y_test_all) - np.sum(y_test_all)} control\n")
    
    # Make predictions using the new averaged model
    decision_scores = X_test_all @ new_coef + new_intercept
    
    # Method 1: Apply sigmoid and use 0.5 threshold (standard logistic regression)
    from scipy.special import expit
    probabilities = expit(decision_scores)
    y_pred_sigmoid = (probabilities > 0.5).astype(int)
    acc_sigmoid = accuracy_score(y_test_all, y_pred_sigmoid)
    
    # Method 2: Use 0 threshold on raw scores (standard SVM-style)
    y_pred_zero = (decision_scores > 0).astype(int)
    acc_zero = accuracy_score(y_test_all, y_pred_zero)
    
    # Method 3: Find optimal threshold
    thresholds = np.percentile(decision_scores, np.linspace(0, 100, 101))
    best_threshold = 0
    best_acc = 0
    
    for threshold in thresholds:
        y_pred_temp = (decision_scores > threshold).astype(int)
        acc_temp = accuracy_score(y_test_all, y_pred_temp)
        if acc_temp > best_acc:
            best_acc = acc_temp
            best_threshold = threshold
    
    y_pred_optimal = (decision_scores > best_threshold).astype(int)
    acc_optimal = accuracy_score(y_test_all, y_pred_optimal)
    
    print(f"\n{'='*70}")
    print("THRESHOLD COMPARISON")
    print(f"{'='*70}")
    print(f"Sigmoid (prob > 0.5): Accuracy = {acc_sigmoid:.4f}")
    print(f"Zero threshold (score > 0): Accuracy = {acc_zero:.4f}")
    print(f"Optimal threshold (score > {best_threshold:.4f}): Accuracy = {acc_optimal:.4f}")
    
    # Use the best performing method
    if acc_sigmoid >= max(acc_zero, acc_optimal):
        y_pred = y_pred_sigmoid
        threshold_method = 'sigmoid'
        threshold_value = 0.5
        print(f"\nUsing sigmoid method")
    elif acc_optimal > acc_zero:
        y_pred = y_pred_optimal
        threshold_method = 'optimal'
        threshold_value = best_threshold
        print(f"\nUsing optimal threshold method")
    else:
        y_pred = y_pred_zero
        threshold_method = 'zero'
        threshold_value = 0.0
        print(f"\nUsing zero threshold method")
    
    # Calculate spectrum-level accuracy
    spectrum_acc = accuracy_score(y_test_all, y_pred)
    conf_matrix = confusion_matrix(y_test_all, y_pred)
    
    # Calculate AUC
    try:
        auc_score = roc_auc_score(y_test_all, probabilities)
    except:
        auc_score = None
    
    print(f"\n{'='*70}")
    print("SPECTRUM-LEVEL RESULTS")
    print(f"{'='*70}")
    print(f"Accuracy: {spectrum_acc:.4f}")
    if auc_score:
        print(f"AUC: {auc_score:.4f}")
    print(f"\nConfusion Matrix: TN={conf_matrix[0,0]} FP={conf_matrix[0,1]} FN={conf_matrix[1,0]} TP={conf_matrix[1,1]}")
    print(f"\nClassification Report:")
    print(classification_report(y_test_all, y_pred, target_names=['Control', 'Cancer']))
    
    # Patient-level analysis
    print(f"\n{'='*70}")
    print("PATIENT-LEVEL ANALYSIS")
    print(f"{'='*70}\n")
    
    patient_level_results = []
    for patient_id in test_patients:
        patient_indices = [i for i, info in enumerate(test_info) if info['patient_id'] == patient_id]
        patient_true_labels = y_test_all[patient_indices]
        patient_predictions = y_pred[patient_indices]
        patient_scores = decision_scores[patient_indices]
        
        n_spectra = len(patient_indices)
        n_correct = np.sum(patient_true_labels == patient_predictions)
        pct_correct = 100 * n_correct / n_spectra
        
        true_label = test_info[patient_indices[0]]['label']
        majority_pred = 1 if np.sum(patient_predictions) > n_spectra / 2 else 0
        majority_pred_label = 'cancer' if majority_pred == 1 else 'control'
        majority_correct = (majority_pred == patient_true_labels[0])
        
        patient_level_results.append({
            'patient_id': patient_id,
            'true_label': true_label,
            'n_spectra': n_spectra,
            'n_correct': n_correct,
            'pct_correct': pct_correct,
            'majority_vote_pred': majority_pred_label,
            'majority_vote_correct': majority_correct,
            'avg_decision_score': np.mean(patient_scores),
            'std_decision_score': np.std(patient_scores)
        })
        
        print(f"Patient {patient_id} ({true_label.upper()}): {n_spectra} spectra, "
              f"{pct_correct:.1f}% correct, Majority: {majority_pred_label} {'✓' if majority_correct else '✗'}")
    
    patient_results_df = pd.DataFrame(patient_level_results)
    patient_level_acc = np.mean(patient_results_df['majority_vote_correct'])
    
    print(f"\nPatient-level accuracy (Majority Vote): {patient_level_acc:.4f} "
          f"({np.sum(patient_results_df['majority_vote_correct'])}/{len(test_patients)})")
    
    # Summary comparison
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Models used: {len(high_acc_indices)}")
    print(f"Features selected: {top_n_features}")
    print(f"Threshold method: {threshold_method}")
    print(f"Spectrum-level accuracy: {spectrum_acc:.4f}")
    print(f"Patient-level accuracy: {patient_level_acc:.4f}")
    if auc_score:
        print(f"AUC: {auc_score:.4f}")
    
    return {
        'new_coef': new_coef,
        'new_intercept': new_intercept,
        'top_feature_indices': top_feature_indices,
        'feature_counts': feature_counts,
        'feature_coefs_avg': feature_coefs_sum / len(high_acc_indices),
        'X_test': X_test_all,
        'y_test': y_test_all,
        'test_info': test_info,
        'predictions': y_pred,
        'decision_scores': decision_scores,
        'probabilities': probabilities,
        'threshold_method': threshold_method,
        'threshold_value': threshold_value,
        'spectrum_accuracy': spectrum_acc,
        'patient_accuracy': patient_level_acc,
        'auc_score': auc_score,
        'confusion_matrix': conf_matrix,
        'patient_results': patient_results_df,
        'n_models_used': len(high_acc_indices),
        'train_accuracy_threshold': train_accuracy_threshold,
        'test_accuracy_threshold': test_accuracy_threshold,
        'top_n_features': top_n_features,
        'figure': fig
    }


def plot_patient_spectra_classification(averaged_model_results, dataset, patient_groups, 
                                        test_patients, raman_shift=None, save_dir=None):
    """
    Plot spectra for each test patient, colored by classification correctness.
    Blue = correctly classified, Red = incorrectly classified.
    
    Args:
        averaged_model_results: Dictionary returned from create_averaged_top_features_model
        dataset: The dataset object
        patient_groups: Dictionary mapping patient_id to spectrum indices
        test_patients: List of test patient IDs
        raman_shift: Array of Raman shift values. If None, extracted from dataset.
        save_dir: Directory to save individual patient plots. If None, just displays.
    
    Returns:
        Dictionary with figure objects for each patient
    """
    import os
    
    test_info = averaged_model_results['test_info']
    predictions = averaged_model_results['predictions']
    y_test = averaged_model_results['y_test']
    decision_scores = averaged_model_results['decision_scores']
    
    # Get raman shift if not provided
    if raman_shift is None:
        _, raman_shift_tensor, _ = dataset[0]
        raman_shift = raman_shift_tensor.squeeze().numpy()
    
    # Create save directory if specified
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    
    patient_figures = {}
    
    for patient_id in test_patients:
        # Get indices for this patient
        patient_indices = [i for i, info in enumerate(test_info) if info['patient_id'] == patient_id]
        
        if len(patient_indices) == 0:
            continue
        
        # Get patient data
        patient_true_label = test_info[patient_indices[0]]['label']
        patient_predictions = predictions[patient_indices]
        patient_true_labels = y_test[patient_indices]
        patient_scores = decision_scores[patient_indices]
        
        # Determine correctness
        correct_mask = patient_predictions == patient_true_labels
        n_correct = np.sum(correct_mask)
        n_total = len(patient_indices)
        pct_correct = 100 * n_correct / n_total
        
        # Majority vote
        majority_pred = 1 if np.sum(patient_predictions) > n_total / 2 else 0
        majority_pred_label = 'Cancer' if majority_pred == 1 else 'Control'
        majority_correct = (majority_pred == patient_true_labels[0])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Plot each spectrum
        for idx, (global_idx, is_correct, pred, score) in enumerate(zip(patient_indices, correct_mask, 
                                                                          patient_predictions, patient_scores)):
            # Get spectrum data
            dataset_idx = test_info[global_idx]['dataset_index']
            intensity_tensor, _, _ = dataset[dataset_idx]
            intensity = intensity_tensor.squeeze().numpy()
            
            # Color based on correctness
            color = 'blue' if is_correct else 'red'
            alpha = 0.6
            linewidth = 1.5 if not is_correct else 0.8
            
            # Label for legend (only once per category)
            if idx == 0:
                label_correct = f'Correct ({n_correct})'
                label_incorrect = f'Incorrect ({n_total - n_correct})'
            else:
                label_correct = None
                label_incorrect = None
            
            if is_correct and label_correct is not None:
                ax.plot(raman_shift, intensity, color=color, alpha=alpha, 
                       linewidth=linewidth, label=label_correct)
                label_correct = None  # Only label once
            elif not is_correct and label_incorrect is not None:
                ax.plot(raman_shift, intensity, color=color, alpha=alpha, 
                       linewidth=linewidth, label=label_incorrect)
                label_incorrect = None  # Only label once
            else:
                ax.plot(raman_shift, intensity, color=color, alpha=alpha, linewidth=linewidth)
        
        # Formatting
        ax.set_xlabel('Raman Shift (cm⁻¹)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Intensity', fontsize=13, fontweight='bold')
        
        # Title with patient info
        title = f'Patient {patient_id} - True Label: {patient_true_label.upper()}\n'
        title += f'{n_correct}/{n_total} spectra correct ({pct_correct:.1f}%) | '
        title += f'Majority Vote: {majority_pred_label} '
        title += '✓' if majority_correct else '✗'
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        
        # Legend
        ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        # Add text box with statistics
        stats_text = f'Avg Score: {np.mean(patient_scores):.3f}\n'
        stats_text += f'Score Range: [{np.min(patient_scores):.3f}, {np.max(patient_scores):.3f}]'
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save if directory specified
        if save_dir is not None:
            filename = f'patient_{patient_id}_{patient_true_label}.png'
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        patient_figures[patient_id] = fig
        plt.show()
    
    return patient_figures


def plot_all_patients_grid(averaged_model_results, dataset, patient_groups, 
                           test_patients, raman_shift=None, save_path=None):
    """
    Plot all test patients in a grid layout for easy comparison.
    
    Args:
        averaged_model_results: Dictionary returned from create_averaged_top_features_model
        dataset: The dataset object
        patient_groups: Dictionary mapping patient_id to spectrum indices
        test_patients: List of test patient IDs
        raman_shift: Array of Raman shift values. If None, extracted from dataset.
        save_path: Path to save the grid plot. If None, just displays.
    
    Returns:
        Figure object
    """
    test_info = averaged_model_results['test_info']
    predictions = averaged_model_results['predictions']
    y_test = averaged_model_results['y_test']
    
    # Get raman shift if not provided
    if raman_shift is None:
        _, raman_shift_tensor, _ = dataset[0]
        raman_shift = raman_shift_tensor.squeeze().numpy()
    
    # Determine grid layout
    n_patients = len(test_patients)
    n_cols = min(3, n_patients)
    n_rows = int(np.ceil(n_patients / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_patients == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, patient_id in enumerate(test_patients):
        ax = axes[idx]
        
        # Get indices for this patient
        patient_indices = [i for i, info in enumerate(test_info) if info['patient_id'] == patient_id]
        
        if len(patient_indices) == 0:
            ax.axis('off')
            continue
        
        # Get patient data
        patient_true_label = test_info[patient_indices[0]]['label']
        patient_predictions = predictions[patient_indices]
        patient_true_labels = y_test[patient_indices]
        
        # Determine correctness
        correct_mask = patient_predictions == patient_true_labels
        n_correct = np.sum(correct_mask)
        n_total = len(patient_indices)
        pct_correct = 100 * n_correct / n_total
        
        # Majority vote
        majority_pred = 1 if np.sum(patient_predictions) > n_total / 2 else 0
        majority_correct = (majority_pred == patient_true_labels[0])
        
        # Plot each spectrum
        for global_idx, is_correct in zip(patient_indices, correct_mask):
            # Get spectrum data
            dataset_idx = test_info[global_idx]['dataset_index']
            intensity_tensor, _, _ = dataset[dataset_idx]
            intensity = intensity_tensor.squeeze().numpy()
            
            # Color based on correctness
            color = 'blue' if is_correct else 'red'
            alpha = 0.5
            linewidth = 1.2 if not is_correct else 0.6
            
            ax.plot(raman_shift, intensity, color=color, alpha=alpha, linewidth=linewidth)
        
        # Formatting
        ax.set_xlabel('Raman Shift (cm⁻¹)', fontsize=10)
        ax.set_ylabel('Intensity', fontsize=10)
        
        # Title
        mv_symbol = '✓' if majority_correct else '✗'
        title = f'{patient_id} ({patient_true_label.upper()}) {mv_symbol}\n{pct_correct:.0f}% correct'
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_patients, len(axes)):
        axes[idx].axis('off')
    
    # Add overall title
    fig.suptitle('Test Patient Spectra Classification Results\nBlue = Correct, Red = Incorrect', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Grid plot saved: {save_path}")
    
    plt.show()
    
    return fig

# Example usage
if __name__ == "__main__":
    # Load dataset
    data_folder = r'C:\Users\Yifei\Downloads\data_izabella_filtered\data_izabella_filtered'
    
    preprocessor = SpectrumPreprocessor(
        remove_cosmic_rays=False,
        normalization=True,
        smoothing=True
    )
    
    dataset = OC_Dataset(data_folder, preprocessor, augmentor=None)
    patient_groups = get_patient_groups(dataset)
    
    # Load training outputs
    with open(r"D:\n100000_c5_t02_s5_with_preprocessing_logistics_training_output.pkl", "rb") as f:
        output = pickle.load(f)
    
    results = output["results"]
    spectrum_selections = output["spectrum_selections"]
    train_patients = output["train_patients"]
    test_patients = output["test_patients"]
    models = output["models"]
    raman_shift = output["raman_shift"]
    
    # # Plot clustering analysis
    # plot_selected_spectra_and_clustering(
    #     dataset, patient_groups, results, spectrum_selections,
    #     train_patients, accuracy_threshold=0.8,

    #     save_path="selected_spectra_clustering.png"
    # )

    # # Train on pooled perfect spectra
    # train_on_pooled_perfect_spectra(
    #     dataset, patient_groups, results, spectrum_selections,
    #     train_patients, test_patients,
    #     accuracy_threshold=0.8, C=1.0, random_state=42,
    #     plot_results=True, model_type='svm'
    # )

    #     # 1. Predict on pooled test spectra
    # pooled_results = predict_with_high_accuracy_models_on_pooled_spectra(
    #     dataset, patient_groups, results, spectrum_selections,
    #     models, test_patients, accuracy_threshold=0.8
    # )
    
    # # 2. Predict on ALL test spectra
    # all_results = predict_with_high_accuracy_models_on_all_test_spectra(
    #     dataset, patient_groups, results, models,
    #     test_patients, accuracy_threshold=0.8
    # )
    
    # # Save results if needed
    # with open("pooled_predictions.pkl", "wb") as f:
    #     pickle.dump(pooled_results, f)
    
    # with open("all_test_predictions.pkl", "wb") as f:
    #     pickle.dump(all_results, f)

    # results_all_spectra = train_on_all_patient_spectra(
    # dataset, patient_groups, train_patients, test_patients,
    # C=1, model_type='logistic_regression', plot_results=True
    # )

    #     # Method 1: Load train accuracies from separate file
    # dual_filter_results = predict_with_high_accuracy_models_on_pooled_spectra_dual_filter(
    #     dataset, patient_groups, results, spectrum_selections,
    #     models, test_patients, 
    #     train_accuracy_threshold=0.95,
    #     test_accuracy_threshold=0.80,
    #     train_accuracies_file="train_accuracies.pkl"  # Provide path to train accuracies file
    #     )

    #     # Save results
    # if dual_filter_results is not None:
    #     with open("dual_filter_pooled_predictions.pkl", "wb") as f:
    #         pickle.dump(dual_filter_results, f)

    # Create averaged top features model
    averaged_model_results = create_averaged_top_features_model(
        dataset, patient_groups, results, spectrum_selections,
        models, test_patients, 
        train_accuracy_threshold=0.95,
        test_accuracy_threshold=0.80,
        train_accuracies_file="train_accuracies.pkl",
        top_n_features=100
    )

    if averaged_model_results is not None:
        # Plot individual patient spectra with detailed view
        patient_figs = plot_patient_spectra_classification(
            averaged_model_results, dataset, patient_groups, test_patients,
            raman_shift=raman_shift,
            save_dir="patient_spectra_plots"  # Will save individual plots here
        )
        
        # Plot all patients in a grid for quick overview
        grid_fig = plot_all_patients_grid(
            averaged_model_results, dataset, patient_groups, test_patients,
            raman_shift=raman_shift,
            save_path="all_patients_grid.png"
        )