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


def predict_on_external_dataset(averaged_model_results, external_dataset, external_patient_groups, 
                                 raman_shift=None, plot_results=True):
    """
    Use the averaged top-features model to predict on an external (new) dataset.
    
    Args:
        averaged_model_results: Dictionary returned from create_averaged_top_features_model
        external_dataset: External dataset object to predict on
        external_patient_groups: Dictionary mapping patient_id to spectrum indices for external data
        raman_shift: Array of Raman shift values. If None, extracted from dataset.
        plot_results: Whether to create visualization plots
    
    Returns:
        Dictionary containing predictions and evaluation metrics on external data
    """
    # Extract model parameters
    new_coef = averaged_model_results['new_coef']
    new_intercept = averaged_model_results['new_intercept']
    threshold_method = averaged_model_results['threshold_method']
    threshold_value = averaged_model_results['threshold_value']
    
    print(f"\n{'='*70}")
    print("PREDICTING ON EXTERNAL DATASET")
    print(f"{'='*70}\n")
    print(f"Using model with {np.sum(new_coef != 0)} non-zero features")
    print(f"Threshold method: {threshold_method} (threshold = {threshold_value:.4f})\n")
    
    # Get all external patients
    external_patient_ids = list(external_patient_groups.keys())
    
    # Determine patient labels
    patient_labels = {}
    for pid in external_patient_ids:
        _, _, metadata = external_dataset[external_patient_groups[pid][0]]
        staging = metadata['staging'].lower()
        patient_labels[pid] = 1 if ('cancer' in staging or 'ca' in staging) else 0
    
    # Collect ALL external spectra
    X_external, y_external, external_info = [], [], []
    for patient_id in external_patient_ids:
        for spec_idx in external_patient_groups[patient_id]:
            intensity_tensor, _, metadata = external_dataset[spec_idx]
            intensity = intensity_tensor.squeeze().numpy()
            label = patient_labels[patient_id]
            
            X_external.append(intensity)
            y_external.append(label)
            external_info.append({
                'patient_id': patient_id,
                'spectrum_id': metadata['spectrum_id'],
                'dataset_index': spec_idx,
                'label': 'cancer' if label == 1 else 'control'
            })
    
    X_external = np.array(X_external)
    y_external = np.array(y_external)
    
    # Count patients by label
    cancer_patients = sum([1 for pid in external_patient_ids if patient_labels[pid] == 1])
    control_patients = len(external_patient_ids) - cancer_patients
    
    print(f"External dataset: {len(external_patient_ids)} patients, {len(X_external)} spectra")
    print(f"  Patients: {cancer_patients} cancer, {control_patients} control")
    print(f"  Spectra: {np.sum(y_external)} cancer, {len(y_external) - np.sum(y_external)} control\n")
    
    # Make predictions using the averaged model
    decision_scores = X_external @ new_coef + new_intercept
    
    # Apply the same threshold method as training
    from scipy.special import expit
    probabilities = expit(decision_scores)
    
    if threshold_method == 'sigmoid':
        y_pred = (probabilities > threshold_value).astype(int)
    elif threshold_method == 'optimal':
        y_pred = (decision_scores > threshold_value).astype(int)
    else:  # zero
        y_pred = (decision_scores > threshold_value).astype(int)
    
    # Calculate spectrum-level accuracy
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
    spectrum_acc = accuracy_score(y_external, y_pred)
    conf_matrix = confusion_matrix(y_external, y_pred)
    
    # Calculate AUC
    try:
        auc_score = roc_auc_score(y_external, probabilities)
    except:
        auc_score = None
    
    print(f"{'='*70}")
    print("SPECTRUM-LEVEL RESULTS ON EXTERNAL DATA")
    print(f"{'='*70}")
    print(f"Accuracy: {spectrum_acc:.4f}")
    if auc_score:
        print(f"AUC: {auc_score:.4f}")
    print(f"\nConfusion Matrix: TN={conf_matrix[0,0]} FP={conf_matrix[0,1]} FN={conf_matrix[1,0]} TP={conf_matrix[1,1]}")
    print(f"\nClassification Report:")
    print(classification_report(y_external, y_pred, target_names=['Control', 'Cancer']))
    
    # Patient-level analysis
    print(f"\n{'='*70}")
    print("PATIENT-LEVEL ANALYSIS ON EXTERNAL DATA")
    print(f"{'='*70}\n")
    
    patient_level_results = []
    for patient_id in external_patient_ids:
        patient_indices = [i for i, info in enumerate(external_info) if info['patient_id'] == patient_id]
        patient_true_labels = y_external[patient_indices]
        patient_predictions = y_pred[patient_indices]
        patient_scores = decision_scores[patient_indices]
        
        n_spectra = len(patient_indices)
        n_correct = np.sum(patient_true_labels == patient_predictions)
        pct_correct = 100 * n_correct / n_spectra
        
        true_label = external_info[patient_indices[0]]['label']
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
          f"({np.sum(patient_results_df['majority_vote_correct'])}/{len(external_patient_ids)})")
    
    # Summary comparison with training performance
    print(f"\n{'='*70}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*70}")
    print(f"Training set (original test data):")
    print(f"  Spectrum-level accuracy: {averaged_model_results['spectrum_accuracy']:.4f}")
    print(f"  Patient-level accuracy:  {averaged_model_results['patient_accuracy']:.4f}")
    if averaged_model_results['auc_score']:
        print(f"  AUC: {averaged_model_results['auc_score']:.4f}")
    print(f"\nExternal dataset:")
    print(f"  Spectrum-level accuracy: {spectrum_acc:.4f}")
    print(f"  Patient-level accuracy:  {patient_level_acc:.4f}")
    if auc_score:
        print(f"  AUC: {auc_score:.4f}")
    
    # Visualization
    if plot_results:
        _plot_external_results(
            conf_matrix, decision_scores, y_external, patient_results_df,
            spectrum_acc, patient_level_acc, external_info, probabilities,
            averaged_model_results
        )
    
    return {
        'X_external': X_external,
        'y_external': y_external,
        'external_info': external_info,
        'predictions': y_pred,
        'decision_scores': decision_scores,
        'probabilities': probabilities,
        'spectrum_accuracy': spectrum_acc,
        'patient_accuracy': patient_level_acc,
        'auc_score': auc_score,
        'confusion_matrix': conf_matrix,
        'patient_results': patient_results_df,
        'n_patients': len(external_patient_ids),
        'n_spectra': len(X_external)
    }


def _plot_external_results(conf_matrix, decision_scores, y_external, patient_results_df,
                           spectrum_acc, patient_level_acc, external_info, probabilities,
                           averaged_model_results):
    """Create visualization for external dataset predictions."""
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    import seaborn as sns
    
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Confusion Matrix
    ax1 = fig.add_subplot(gs[0, 0])
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Control', 'Cancer'], yticklabels=['Control', 'Cancer'],
                cbar_kws={'label': 'Count'}, ax=ax1)
    ax1.set_title('Confusion Matrix (External Data)', fontweight='bold', fontsize=12)
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    # 2. Decision Score Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    cancer_scores = decision_scores[y_external == 1]
    control_scores = decision_scores[y_external == 0]
    ax2.hist(control_scores, bins=30, alpha=0.6, label='Control', color='green', edgecolor='black')
    ax2.hist(cancer_scores, bins=30, alpha=0.6, label='Cancer', color='red', edgecolor='black')
    
    threshold_val = averaged_model_results['threshold_value']
    if averaged_model_results['threshold_method'] == 'sigmoid':
        # For sigmoid, convert threshold back to decision score
        from scipy.special import logit
        threshold_score = logit(threshold_val) if threshold_val > 0 and threshold_val < 1 else 0
    else:
        threshold_score = threshold_val
    
    ax2.axvline(threshold_score, color='black', linestyle='--', linewidth=2, label='Decision Threshold')
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
    ax4.axvline(threshold_score, color='black', linestyle='--', linewidth=2)
    ax4.grid(axis='x', alpha=0.3)
    
    # 5. Accuracy Comparison (Training vs External)
    ax5 = fig.add_subplot(gs[1, 1])
    x = np.arange(2)
    width = 0.35
    
    train_accs = [averaged_model_results['spectrum_accuracy'], averaged_model_results['patient_accuracy']]
    external_accs = [spectrum_acc, patient_level_acc]
    
    bars1 = ax5.bar(x - width/2, train_accs, width, label='Training Test Set',
                   color='#1f77b4', alpha=0.7, edgecolor='black')
    bars2 = ax5.bar(x + width/2, external_accs, width, label='External Dataset',
                   color='#ff7f0e', alpha=0.7, edgecolor='black')
    
    ax5.set_ylabel('Accuracy')
    ax5.set_ylim([0, 1.1])
    ax5.set_title('Training vs External Performance', fontweight='bold', fontsize=12)
    ax5.set_xticks(x)
    ax5.set_xticklabels(['Spectrum-Level', 'Patient-Level'])
    ax5.legend()
    ax5.grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # 6. ROC Comparison (if AUC available)
    ax6 = fig.add_subplot(gs[1, 2])
    
    # External ROC
    from sklearn.metrics import roc_curve
    try:
        fpr_ext, tpr_ext, _ = roc_curve(y_external, probabilities)
        ext_auc = averaged_model_results.get('auc_score', None)
        
        ax6.plot(fpr_ext, tpr_ext, linewidth=2, label=f'External (AUC={ext_auc:.3f})', color='#ff7f0e')
        ax6.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        ax6.set_xlabel('False Positive Rate')
        ax6.set_ylabel('True Positive Rate')
        ax6.set_title('ROC Curve', fontweight='bold', fontsize=12)
        ax6.legend()
        ax6.grid(alpha=0.3)
    except:
        ax6.text(0.5, 0.5, 'ROC curve unavailable', ha='center', va='center')
        ax6.set_title('ROC Curve', fontweight='bold', fontsize=12)
    
    # 7. Individual Spectrum Scores by Patient
    ax7 = fig.add_subplot(gs[2, :])
    for i, (patient_id, group) in enumerate(patient_results_df.groupby('patient_id')):
        patient_indices = [j for j, info in enumerate(external_info) if info['patient_id'] == patient_id]
        patient_scores = decision_scores[patient_indices]
        true_label = group['true_label'].iloc[0]
        color = 'red' if true_label == 'cancer' else 'green'
        marker = 'o' if group['majority_vote_correct'].iloc[0] else 'x'
        ax7.scatter([i] * len(patient_scores), patient_scores,
                   c=color, marker=marker, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    ax7.axhline(threshold_score, color='black', linestyle='--', linewidth=2)
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
    
    plt.suptitle('External Dataset Prediction Results',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


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
    
    test_info = averaged_model_results['external_info']
    predictions = averaged_model_results['predictions']
    y_test = averaged_model_results['y_external']
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

# Example usage
if __name__ == "__main__":
    # Load dataset and training outputs
    data_folder = r'C:\Users\Yifei\Downloads\data_izabella_filtered\data_izabella_filtered'
    
    preprocessor = SpectrumPreprocessor(
        baseline_correction=True,
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
    models = output["models"]
    test_patients = output["test_patients"]
    raman_shift = output.get("raman_shift", None)
    
    # Create averaged top features model on training data
    averaged_model_results = create_averaged_top_features_model(
        dataset, patient_groups, results, spectrum_selections,
        models, test_patients, 
        train_accuracy_threshold=0.95,
        test_accuracy_threshold=0.80,
        train_accuracies_file="train_accuracies.pkl",
        top_n_features=10,
        raman_shift=raman_shift
    )
    
    if averaged_model_results is not None:
        # Now test on external dataset
        # Load external dataset (replace with your actual external data path)
        external_data_folder = r'C:\Users\Yifei\Downloads\data_bec_evs\data_bec_evs'
        
        preprocessor_with_cropping = SpectrumPreprocessor(
            cropping=True,
            baseline_correction=True,
            remove_cosmic_rays=False,
            normalization=True,
            smoothing=True
        )

        external_dataset = OC_Dataset(external_data_folder, preprocessor_with_cropping, augmentor=None)
        external_patient_groups = get_patient_groups(external_dataset)
        
        # Predict on external dataset
        external_results = predict_on_external_dataset(
            averaged_model_results, 
            external_dataset, 
            external_patient_groups,
            raman_shift=raman_shift,
            plot_results=True
        )
        
        
        # Optionally, plot individual patient spectra for external data
        external_patient_ids = list(external_patient_groups.keys())
        patient_figs = plot_patient_spectra_classification(
            external_results, 
            external_dataset, 
            external_patient_groups, 
            external_patient_ids,
            raman_shift=raman_shift,
            save_dir="external_patient_spectra_plots"
        )

                # Save external results
        with open("external_dataset_predictions.pkl", "wb") as f:
            pickle.dump(external_results, f)