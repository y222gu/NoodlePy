import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import pandas as pd
import scipy.stats as stats
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

def prepare_spectrum_level_data(dataset):
    """
    Extract all individual spectra without patient-level aggregation
    
    Returns:
    --------
    X : array, shape (n_spectra, n_features)
        Individual spectrum intensities
    y : array, shape (n_spectra,)
        Labels (0=control, 1=cancer)
    patient_ids : list
        Patient ID for each spectrum
    """
    X_list = []
    y_list = []
    patient_ids = []
    
    for i in range(len(dataset)):
        intensity, metadata = dataset[i]
        patient_id = metadata['patient_id']
        staging = metadata['staging']
        
        X_list.append(intensity)
        y_list.append(1 if staging.lower() == 'cancer' else 0)
        patient_ids.append(patient_id)
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"Spectrum-level data: {X.shape}")
    print(f"Cancer spectra: {np.sum(y)}, Control spectra: {len(y) - np.sum(y)}")
    print(f"Unique patients: {len(set(patient_ids))}")
    
    return X, y, patient_ids

def select_features_by_effect_size(X, y, top_k=15, min_effect_size=0.6):
    """
    Select features based on effect size between cancer and control spectra
    """
    print("Calculating effect sizes for feature selection...")
    
    cancer_mask = y == 1
    control_mask = y == 0
    X_cancer = X[cancer_mask]
    X_control = X[control_mask]
    
    effect_sizes = []
    p_values = []
    
    for feature_idx in range(X.shape[1]):
        cancer_vals = X_cancer[:, feature_idx]
        control_vals = X_control[:, feature_idx]
        
        # Cohen's d calculation
        mean_diff = np.mean(cancer_vals) - np.mean(control_vals)
        pooled_std = np.sqrt(((len(cancer_vals)-1) * np.var(cancer_vals) + 
                             (len(control_vals)-1) * np.var(control_vals)) / 
                            (len(cancer_vals) + len(control_vals) - 2))
        
        d = abs(mean_diff / pooled_std) if pooled_std > 0 else 0
        effect_sizes.append(d)
        
        # Statistical significance
        t_stat, p_val = stats.ttest_ind(cancer_vals, control_vals)
        p_values.append(p_val)
    
    effect_sizes = np.array(effect_sizes)
    p_values = np.array(p_values)
    
    # Select top features by effect size
    top_indices = np.argsort(effect_sizes)[::-1]
    good_indices = np.where(effect_sizes >= min_effect_size)[0]
    
    # Take top_k features that meet threshold, or just top_k if not enough
    if len(good_indices) >= top_k:
        selected_indices = top_indices[:top_k]
    else:
        selected_indices = good_indices if len(good_indices) > 0 else top_indices[:top_k]
    
    # Create feature info
    feature_info = pd.DataFrame({
        'Feature_Index': selected_indices,
        'Effect_Size': effect_sizes[selected_indices],
        'P_Value': p_values[selected_indices],
        'Interpretation': ['Large' if d > 0.8 else 'Medium' if d > 0.5 else 'Small' 
                          for d in effect_sizes[selected_indices]]
    }).sort_values('Effect_Size', ascending=False).reset_index(drop=True)
    
    print(f"Selected {len(selected_indices)} features")
    print(f"Effect size range: {effect_sizes[selected_indices].min():.3f} - {effect_sizes[selected_indices].max():.3f}")
    print(f"Features with large effect (d>0.8): {np.sum(effect_sizes[selected_indices] > 0.8)}")
    
    return selected_indices, effect_sizes, feature_info

def bootstrap_analysis(X, y, patient_ids, selected_indices, n_bootstrap=200):
    """
    Bootstrap analysis with patient-aware sampling to avoid data leakage
    """
    print(f"\nRunning bootstrap analysis with {n_bootstrap} iterations...")
    
    # Get unique patients and their labels
    unique_patients = list(set(patient_ids))
    patient_labels = {}
    for i, pid in enumerate(patient_ids):
        patient_labels[pid] = y[i]
    
    # Separate patients by class
    cancer_patients = [pid for pid in unique_patients if patient_labels[pid] == 1]
    control_patients = [pid for pid in unique_patients if patient_labels[pid] == 0]
    
    print(f"Unique patients: {len(unique_patients)} (Cancer: {len(cancer_patients)}, Control: {len(control_patients)})")
    
    # Model setup
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(kernel='linear', random_state=42))
    ])
    
    bootstrap_accuracies = []
    X_selected = X[:, selected_indices]
    
    for i in range(n_bootstrap):
        # Bootstrap sample patients (not spectra) to avoid data leakage
        boot_cancer_patients = np.random.choice(cancer_patients, len(cancer_patients), replace=True)
        boot_control_patients = np.random.choice(control_patients, len(control_patients), replace=True)
        boot_patients = list(boot_cancer_patients) + list(boot_control_patients)
        
        # Get all spectra from bootstrap-sampled patients
        boot_indices = [i for i, pid in enumerate(patient_ids) if pid in boot_patients]
        
        if len(boot_indices) < 10:  # Skip if too few spectra
            continue
            
        X_boot = X_selected[boot_indices]
        y_boot = y[boot_indices]
        
        # Skip if not both classes present
        if len(np.unique(y_boot)) < 2:
            continue
        
        try:
            # 3-fold CV on bootstrap sample
            cv_scores = cross_val_score(model, X_boot, y_boot, cv=3, scoring='accuracy')
            bootstrap_accuracies.append(np.mean(cv_scores))
        except:
            continue
    
    bootstrap_accuracies = np.array(bootstrap_accuracies)
    
    print(f"Bootstrap results: {np.mean(bootstrap_accuracies):.3f} ± {np.std(bootstrap_accuracies):.3f}")
    print(f"Range: {np.min(bootstrap_accuracies):.3f} - {np.max(bootstrap_accuracies):.3f}")
    
    return bootstrap_accuracies

def learning_curve_analysis(X, y, patient_ids, selected_indices):
    """
    Learning curve analysis with patient-aware cross-validation
    """
    print("\nRunning learning curve analysis...")
    
    # Get unique patients for proper CV splits
    unique_patients = list(set(patient_ids))
    patient_labels = {pid: y[patient_ids.index(pid)] for pid in unique_patients}
    
    X_selected = X[:, selected_indices]
    
    # Different training sizes (number of patients, not spectra)
    max_patients = len(unique_patients)
    train_sizes = np.linspace(200, max_patients, min(8, max_patients-2)).astype(int)
    train_sizes = train_sizes[train_sizes >= 200]
    
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(kernel='linear', random_state=42))
    ])
    
    results = {'train_sizes': [], 'val_scores': [], 'val_std': []}
    
    for train_size in train_sizes:
        if train_size >= max_patients:
            continue
            
        print(f"  Testing with {train_size} patients...")
        
        # Patient-level stratified sampling
        cancer_patients = [pid for pid in unique_patients if patient_labels[pid] == 1]
        control_patients = [pid for pid in unique_patients if patient_labels[pid] == 0]
        
        cv_scores = []
        n_folds = 5
        
        for fold in range(n_folds):
            # Random stratified split of patients
            np.random.seed(fold)
            
            n_cancer_train = max(1, int(train_size * len(cancer_patients) / len(unique_patients)))
            n_control_train = train_size - n_cancer_train
            
            if n_cancer_train > len(cancer_patients) or n_control_train > len(control_patients):
                continue
                
            train_cancer = np.random.choice(cancer_patients, n_cancer_train, replace=False)
            train_control = np.random.choice(control_patients, n_control_train, replace=False)
            train_patients = list(train_cancer) + list(train_control)
            
            # Remaining patients for validation
            val_patients = [pid for pid in unique_patients if pid not in train_patients]
            
            # Get spectra indices for train/val patients
            train_indices = [i for i, pid in enumerate(patient_ids) if pid in train_patients]
            val_indices = [i for i, pid in enumerate(patient_ids) if pid in val_patients]
            
            if len(train_indices) < 5 or len(val_indices) < 3:
                continue
                
            X_train, X_val = X_selected[train_indices], X_selected[val_indices]
            y_train, y_val = y[train_indices], y[val_indices]
            
            # Skip if not both classes in training
            if len(np.unique(y_train)) < 2:
                continue
                
            try:
                model.fit(X_train, y_train)
                val_pred = model.predict(X_val)
                cv_scores.append(accuracy_score(y_val, val_pred))
            except:
                continue
        
        if len(cv_scores) > 0:
            results['train_sizes'].append(train_size)
            results['val_scores'].append(np.mean(cv_scores))
            results['val_std'].append(np.std(cv_scores))
            print(f"    Validation accuracy: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
    
    return results

def plot_results(effect_sizes, selected_indices, bootstrap_accuracies, learning_results):
    """
    Create comprehensive plots of all results
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Effect size distribution
    ax1.hist(effect_sizes, bins=50, alpha=0.7, edgecolor='black')
    ax1.axvline(np.min(effect_sizes[selected_indices]), color='red', linestyle='--', 
               label=f'Selected threshold ({np.min(effect_sizes[selected_indices]):.2f})')
    ax1.set_xlabel('Effect Size (Cohen\'s d)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Effect Size Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Selected features effect sizes
    ax2.bar(range(len(selected_indices)), effect_sizes[selected_indices], alpha=0.7)
    ax2.set_xlabel('Selected Features (ranked)')
    ax2.set_ylabel('Effect Size')
    ax2.set_title(f'Top {len(selected_indices)} Selected Features')
    ax2.grid(True, alpha=0.3)
    
    # 3. Bootstrap distribution
    ax3.hist(bootstrap_accuracies, bins=20, alpha=0.7, edgecolor='black')
    ax3.axvline(np.mean(bootstrap_accuracies), color='red', linestyle='--', 
               label=f'Mean: {np.mean(bootstrap_accuracies):.3f}')
    ax3.set_xlabel('Bootstrap Accuracy')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Bootstrap Accuracy Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Learning curve
    if len(learning_results['train_sizes']) > 1:
        ax4.errorbar(learning_results['train_sizes'], learning_results['val_scores'], 
                    yerr=learning_results['val_std'], marker='o', capsize=5)
        ax4.set_xlabel('Training Set Size (Patients)')
        ax4.set_ylabel('Validation Accuracy')
        ax4.set_title('Learning Curve')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Insufficient data\nfor learning curve', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Learning Curve (Insufficient Data)')
    
    plt.tight_layout()
    plt.show()

def analyze_results(bootstrap_accuracies, learning_results, feature_info):
    """
    Provide analysis and recommendations
    """
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    
    # Bootstrap analysis
    bootstrap_mean = np.mean(bootstrap_accuracies)
    bootstrap_std = np.std(bootstrap_accuracies)
    bootstrap_range = np.max(bootstrap_accuracies) - np.min(bootstrap_accuracies)
    
    print(f"Bootstrap Performance:")
    print(f"  Mean accuracy: {bootstrap_mean:.3f} ± {bootstrap_std:.3f}")
    print(f"  Range: {bootstrap_range:.3f}")
    print(f"  95% CI width: ~±{2*bootstrap_std:.3f}")
    
    # Stability assessment
    if bootstrap_std < 0.05:
        stability = "EXCELLENT - Very stable performance"
    elif bootstrap_std < 0.08:
        stability = "GOOD - Reasonably stable performance"
    elif bootstrap_std < 0.12:
        stability = "MODERATE - Some variability"
    else:
        stability = "POOR - High variability"
    
    print(f"  Stability: {stability}")
    
    # Feature analysis
    print(f"\nFeature Analysis:")
    print(f"  Selected features: {len(feature_info)}")
    print(f"  Large effect features (d>0.8): {np.sum(feature_info['Effect_Size'] > 0.8)}")
    print(f"  Medium effect features (d>0.5): {np.sum(feature_info['Effect_Size'] > 0.5)}")
    
    # Learning curve analysis
    if len(learning_results['train_sizes']) > 2:
        final_performance = learning_results['val_scores'][-1]
        initial_performance = learning_results['val_scores'][0]
        improvement = final_performance - initial_performance
        
        print(f"\nLearning Curve Analysis:")
        print(f"  Performance at {learning_results['train_sizes'][0]} patients: {initial_performance:.3f}")
        print(f"  Performance at {learning_results['train_sizes'][-1]} patients: {final_performance:.3f}")
        print(f"  Overall improvement: {improvement:+.3f}")
        
        if abs(improvement) < 0.02:
            trend = "PLATEAU - Current size likely adequate"
        elif improvement > 0.05:
            trend = "IMPROVING - More patients would help"
        else:
            trend = "UNCLEAR - Need more analysis"
        
        print(f"  Trend: {trend}")
    
    # Recommendations
    print(f"\nRecommendations:")
    if bootstrap_mean > 0.80 and bootstrap_std < 0.06:
        print("  Status: Ready for validation study")
        print("  Next steps: Test on independent patient cohort")
    elif bootstrap_mean > 0.75 and bootstrap_std < 0.10:
        print("  Status: Promising but needs more data")
        print("  Next steps: Collect 20-30 more patients")
    else:
        print("  Status: Needs significant improvement")
        print("  Next steps: Collect 50+ more patients or revise approach")

def run_spectrum_level_analysis(dataset, top_k=15, min_effect_size=0.6, n_bootstrap=200):
    """
    Complete analysis pipeline treating each spectrum as independent
    
    WARNING: This approach may overestimate performance due to multiple 
    spectra per patient. Use with caution for clinical applications.
    """
    print("="*80)
    print("SPECTRUM-LEVEL ANALYSIS (NO PATIENT AGGREGATION)")
    print("WARNING: May overestimate performance due to within-patient correlation")
    print("="*80)
    
    # 1. Prepare data
    X, y, patient_ids = prepare_spectrum_level_data(dataset)
    
    # 2. Feature selection
    selected_indices, effect_sizes, feature_info = select_features_by_effect_size(
        X, y, top_k=top_k, min_effect_size=min_effect_size)
    
    print(f"\nTop 10 selected features:")
    print(feature_info.head(10).to_string(index=False))
    
    # 3. Bootstrap analysis (patient-aware)
    bootstrap_accuracies = bootstrap_analysis(X, y, patient_ids, selected_indices, n_bootstrap)
    
    # 4. Learning curve analysis (patient-aware)
    learning_results = learning_curve_analysis(X, y, patient_ids, selected_indices)
    
    # 5. Visualization
    plot_results(effect_sizes, selected_indices, bootstrap_accuracies, learning_results)
    
    # 6. Analysis and recommendations
    analyze_results(bootstrap_accuracies, learning_results, feature_info)
    
    return {
        'selected_indices': selected_indices,
        'feature_info': feature_info,
        'bootstrap_accuracies': bootstrap_accuracies,
        'learning_results': learning_results,
        'X_selected': X[:, selected_indices],
        'y': y,
        'patient_ids': patient_ids
    }

# Usage example
if __name__ == "__main__":
    # Load dataset
    from noodlepy.utils.spectrumpreprocessor import SpectrumPreprocessor
    from noodlepy.utils.izabelladataset import OC_Dataset

    data_folder = r'C:\Users\Yifei\Downloads\data_izabella_filtered\data_izabella_filtered'
    preprocessor = SpectrumPreprocessor(normalization=True)
    dataset = OC_Dataset(data_folder, preprocessor)

    # Run spectrum-level analysis
    results = run_spectrum_level_analysis(
        dataset, 
        top_k=15,           # Select top 15 features
        min_effect_size=0.6, # Minimum effect size threshold
        n_bootstrap=200     # Bootstrap iterations
    )