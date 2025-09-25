import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from noodlepy.utils.izabelladataset import OC_Dataset
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
from noodlepy.utils.spectrumpreprocessor import SpectrumPreprocessor

class LearningCurveAnalysis:
    def __init__(self, dataset):
        """
        Initialize with your OC_Dataset
        """
        self.dataset = dataset
        self.X, self.y, self.patient_ids = self._prepare_data()
        
    def _prepare_data(self):
        """
        Extract features and labels from dataset, aggregate by patient
        """
        print("Preparing patient-level aggregated data...")
        
        # Dictionary to store spectra by patient
        patient_spectra = defaultdict(list)
        patient_labels = {}
        
        # Collect all spectra per patient
        for i in range(len(self.dataset)):
            intensity, metadata = self.dataset[i]
            patient_id = metadata['patient_id']
            staging = metadata['staging']
            
            patient_spectra[patient_id].append(intensity)
            patient_labels[patient_id] = staging
        
        # Aggregate spectra per patient (mean + std)
        X_aggregated = []
        y_aggregated = []
        patient_ids = []
        
        for patient_id, spectra_list in patient_spectra.items():
            # Stack all spectra for this patient
            spectra_array = np.stack(spectra_list)
            
            # Calculate mean and std across spectra
            mean_spectrum = np.mean(spectra_array, axis=0)
            std_spectrum = np.std(spectra_array, axis=0)
            
            # Combine mean and std features
            combined_features = np.concatenate([mean_spectrum, std_spectrum])
            
            X_aggregated.append(combined_features)
            y_aggregated.append(1 if patient_labels[patient_id].lower() == 'cancer' else 0)
            patient_ids.append(patient_id)
        
        X = np.array(X_aggregated)
        y = np.array(y_aggregated)
        
        print(f"Patient-level data shape: {X.shape}")
        print(f"Class distribution - Cancer: {np.sum(y)}, Control: {len(y) - np.sum(y)}")
        
        return X, y, patient_ids
    
    def learning_curve_analysis(self, 
                               train_sizes=None,
                               cv_folds=5,
                               n_features_list=[5, 20, 100],
                               random_state=42):
        """
        Perform learning curve analysis with different models and feature counts
        """
        if train_sizes is None:
            # Create training sizes from 10 to total patients
            max_size = len(self.y)
            train_sizes = np.linspace(10, max_size, min(8, max_size-5)).astype(int)
            train_sizes = train_sizes[train_sizes >= 10]  # Minimum 10 patients
        
        # Models to test
        models = {
            'Linear SVM': Pipeline([
                ('scaler', StandardScaler()),
                ('feature_selection', SelectKBest(f_classif)),
                ('classifier', SVC(kernel='linear', random_state=random_state))
            ]),
            'Logistic Regression': Pipeline([
                ('scaler', StandardScaler()),
                ('feature_selection', SelectKBest(f_classif)),
                ('classifier', LogisticRegression(random_state=random_state, max_iter=1000))
            ]),
            'Random Forest': Pipeline([
                ('feature_selection', SelectKBest(f_classif)),
                ('classifier', RandomForestClassifier(n_estimators=100, random_state=random_state))
            ])
        }
        
        results = {}
        
        # Test different numbers of features
        for n_features in n_features_list:
            if n_features > self.X.shape[1]:
                continue
                
            results[n_features] = {}
            
            for model_name, model in models.items():
                print(f"\nTesting {model_name} with {n_features} features...")
                
                # Set number of features
                model.named_steps['feature_selection'].k = n_features
                
                train_scores = []
                val_scores = []
                train_std = []
                val_std = []
                
                for train_size in train_sizes:
                    if train_size >= len(self.y):
                        continue
                        
                    # Perform cross-validation with current training size
                    cv_train_scores = []
                    cv_val_scores = []
                    
                    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
                    
                    for train_idx, val_idx in skf.split(self.X, self.y):
                        # Limit training set size
                        if len(train_idx) > train_size:
                            # Stratified sampling to maintain class balance
                            train_cancer_idx = train_idx[self.y[train_idx] == 1]
                            train_control_idx = train_idx[self.y[train_idx] == 0]
                            
                            n_cancer = int(train_size * np.mean(self.y))
                            n_control = train_size - n_cancer
                            
                            if len(train_cancer_idx) >= n_cancer and len(train_control_idx) >= n_control:
                                selected_cancer = np.random.choice(train_cancer_idx, n_cancer, replace=False)
                                selected_control = np.random.choice(train_control_idx, n_control, replace=False)
                                train_idx = np.concatenate([selected_cancer, selected_control])
                        
                        X_train, X_val = self.X[train_idx], self.X[val_idx]
                        y_train, y_val = self.y[train_idx], self.y[val_idx]
                        
                        # Train and evaluate
                        model.fit(X_train, y_train)
                        
                        train_pred = model.predict(X_train)
                        val_pred = model.predict(X_val)
                        
                        cv_train_scores.append(accuracy_score(y_train, train_pred))
                        cv_val_scores.append(accuracy_score(y_val, val_pred))
                    
                    if len(cv_train_scores) > 0:  # Only append if we have valid scores
                        train_scores.append(np.mean(cv_train_scores))
                        val_scores.append(np.mean(cv_val_scores))
                        train_std.append(np.std(cv_train_scores))
                        val_std.append(np.std(cv_val_scores))
                
                results[n_features][model_name] = {
                    'train_sizes': train_sizes[:len(train_scores)],
                    'train_scores': np.array(train_scores),
                    'val_scores': np.array(val_scores),
                    'train_std': np.array(train_std),
                    'val_std': np.array(val_std)
                }
        
        return results
    
    def plot_learning_curves(self, results):
        """
        Plot learning curves for different models and feature counts,
        and add a baseline accuracy representing random guessing based on the class ratio.
        """
        # Calculate baseline accuracy from class ratio (majority class)
        baseline = max(np.mean(self.y == 1), np.mean(self.y == 0))
        
        n_features_list = list(results.keys())
        n_models = len(list(results[n_features_list[0]].keys()))
        
        fig, axes = plt.subplots(len(n_features_list), 1, figsize=(12, 4 * len(n_features_list)))
        if len(n_features_list) == 1:
            axes = [axes]
        
        colors = ['blue', 'red', 'green', 'orange']
        
        for i, n_features in enumerate(n_features_list):
            ax = axes[i]
            
            for j, (model_name, result) in enumerate(results[n_features].items()):
                train_sizes = result['train_sizes']
                train_scores = result['train_scores']
                val_scores = result['val_scores']
                train_std = result['train_std']
                val_std = result['val_std']
                
                color = colors[j % len(colors)]
                
                # Plot training scores
                ax.plot(train_sizes, train_scores, 'o-', color=color, 
                        label=f'{model_name} (Train)', alpha=0.7)
                ax.fill_between(train_sizes, train_scores - train_std, 
                                train_scores + train_std, alpha=0.1, color=color)
                
                # Plot validation scores
                ax.plot(train_sizes, val_scores, 's--', color=color, 
                        label=f'{model_name} (Val)', alpha=0.7)
                ax.fill_between(train_sizes, val_scores - val_std, 
                                val_scores + val_std, alpha=0.1, color=color)
            
            # Plot baseline accuracy
            ax.axhline(baseline, color='gray', linestyle='--', 
                       label=f'Baseline (Random Guess: {baseline*100:.1f}%)', lw=2)
            ax.set_xlabel('Training Set Size (Patients)')
            ax.set_ylabel('Accuracy')
            ax.set_title(f'Learning Curves with {n_features} Features')
            ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0.3, 1.05)
        
        plt.tight_layout(pad=3.0, h_pad=3.0, w_pad=3.0)
        plt.show()
    
    def bootstrap_stability_analysis(self, n_bootstrap=100, n_features=10):
        """
        Analyze model stability using bootstrap sampling
        """
        print(f"\nPerforming bootstrap stability analysis with {n_bootstrap} iterations...")
        
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('feature_selection', SelectKBest(f_classif, k=n_features)),
            ('classifier', SVC(kernel='linear', random_state=42))
        ])
        
        accuracies = []
        feature_selections = []
        
        for i in range(n_bootstrap):
            # Bootstrap sampling
            indices = np.random.choice(len(self.y), len(self.y), replace=True)
            X_boot = self.X[indices]
            y_boot = self.y[indices]
            
            # Cross-validation on bootstrap sample
            cv_scores = cross_val_score(model, X_boot, y_boot, cv=3, scoring='accuracy')
            accuracies.append(np.mean(cv_scores))
            
            # Track selected features
            model.fit(X_boot, y_boot)
            selected_features = model.named_steps['feature_selection'].get_support()
            feature_selections.append(selected_features)
        
        accuracies = np.array(accuracies)
        feature_selections = np.array(feature_selections)
        
        # Feature stability (how often each feature is selected)
        feature_stability = np.mean(feature_selections, axis=0)
        
        print(f"Bootstrap Results:")
        print(f"  Mean Accuracy: {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")
        print(f"  Accuracy Range: [{np.min(accuracies):.3f}, {np.max(accuracies):.3f}]")
        print(f"  Most stable features (selected >50% of time): {np.sum(feature_stability > 0.5)}")
        
        # Plot results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy distribution
        ax1.hist(accuracies, bins=20, alpha=0.7, edgecolor='black')
        ax1.axvline(np.mean(accuracies), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(accuracies):.3f}')
        ax1.set_xlabel('Bootstrap Accuracy')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Bootstrap Accuracy Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Feature stability
        feature_indices = np.arange(len(feature_stability))
        ax2.bar(feature_indices, feature_stability, alpha=0.7)
        ax2.axhline(0.5, color='red', linestyle='--', label='50% threshold')
        ax2.set_xlabel('Feature Index')
        ax2.set_ylabel('Selection Frequency')
        ax2.set_title('Feature Selection Stability')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return accuracies, feature_stability
    
    def effect_size_analysis(self):
        """
        Calculate effect sizes for features to estimate discriminative power
        """
        print("\nCalculating effect sizes...")
        
        cancer_indices = self.y == 1
        control_indices = self.y == 0
        
        X_cancer = self.X[cancer_indices]
        X_control = self.X[control_indices]
        
        # Calculate Cohen's d for each feature
        effect_sizes = []
        p_values = []
        
        from scipy.stats import ttest_ind
        
        for i in range(self.X.shape[1]):
            cancer_values = X_cancer[:, i]
            control_values = X_control[:, i]
            
            # T-test
            t_stat, p_val = ttest_ind(cancer_values, control_values)
            p_values.append(p_val)
            
            # Cohen's d
            mean_diff = np.mean(cancer_values) - np.mean(control_values)
            pooled_std = np.sqrt(((len(cancer_values)-1) * np.var(cancer_values) + 
                                 (len(control_values)-1) * np.var(control_values)) / 
                                (len(cancer_values) + len(control_values) - 2))
            
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
            effect_sizes.append(abs(cohens_d))
        
        effect_sizes = np.array(effect_sizes)
        p_values = np.array(p_values)
        
        # Find top discriminative features
        top_indices = np.argsort(effect_sizes)[::-1][:20]
        
        print(f"Effect Size Analysis:")
        print(f"  Mean effect size: {np.mean(effect_sizes):.3f}")
        print(f"  Max effect size: {np.max(effect_sizes):.3f}")
        print(f"  Features with large effect (d>0.8): {np.sum(effect_sizes > 0.8)}")
        print(f"  Features with medium effect (d>0.5): {np.sum(effect_sizes > 0.5)}")
        print(f"  Features with small effect (d>0.2): {np.sum(effect_sizes > 0.2)}")
        
        # Plot effect sizes
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.hist(effect_sizes, bins=30, alpha=0.7, edgecolor='black')
        plt.axvline(0.2, color='green', linestyle='--', label='Small (0.2)')
        plt.axvline(0.5, color='orange', linestyle='--', label='Medium (0.5)')
        plt.axvline(0.8, color='red', linestyle='--', label='Large (0.8)')
        plt.xlabel('Effect Size (Cohen\'s d)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Effect Sizes')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.plot(effect_sizes, 'o', alpha=0.6)
        plt.xlabel('Feature Index')
        plt.ylabel('Effect Size')
        plt.title('Effect Size by Feature')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        plt.scatter(effect_sizes, -np.log10(p_values + 1e-10), alpha=0.6)
        plt.xlabel('Effect Size (Cohen\'s d)')
        plt.ylabel('-log10(p-value)')
        plt.title('Volcano Plot: Effect Size vs Significance')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        top_20_effects = effect_sizes[top_indices]
        plt.bar(range(len(top_20_effects)), top_20_effects, alpha=0.7)
        plt.xlabel('Top 20 Features (ranked)')
        plt.ylabel('Effect Size')
        plt.title('Top 20 Most Discriminative Features')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return effect_sizes, p_values, top_indices
    
    def sample_size_estimation(self, target_accuracy=0.85, confidence_width=0.1):
        """
        Estimate required sample size for target accuracy with given confidence interval width
        """
        print(f"\nSample size estimation for {target_accuracy:.0%} accuracy ± {confidence_width/2:.0%}")
        
        # Current performance estimation
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('feature_selection', SelectKBest(f_classif, k=10)),
            ('classifier', SVC(kernel='linear'))
        ])
        
        current_scores = cross_val_score(model, self.X, self.y, cv=5)
        current_mean = np.mean(current_scores)
        current_std = np.std(current_scores)
        
        print(f"Current performance: {current_mean:.3f} ± {current_std:.3f}")
        
        # Sample size estimation for confidence interval
        from scipy.stats import norm
        z_score = norm.ppf(0.975)  # 95% confidence
        
        # Estimated required sample size for desired CI width
        estimated_variance = target_accuracy * (1 - target_accuracy)  # Binomial variance
        required_n = (z_score / (confidence_width/2))**2 * estimated_variance
        
        print(f"Estimated sample size needed:")
        print(f"  For ±{confidence_width/2:.0%} confidence interval: ~{int(required_n)} patients")
        
        # Rule of thumb estimations
        print(f"\nRule of thumb estimations:")
        print(f"  For stable model (10-20 samples per feature with 10 features): 100-200 patients")
        print(f"  For clinical deployment (narrow CI): 150-300 patients")
        print(f"  Your current dataset: {len(self.y)} patients")
        
        return required_n

# Example usage function
def run_complete_analysis(dataset):
    """
    Run all analyses on the dataset
    """
    print("="*60)
    print("COMPREHENSIVE LEARNING CURVE ANALYSIS")
    print("="*60)
    
    # Initialize analysis
    analysis = LearningCurveAnalysis(dataset)
    
    # 1. Learning curve analysis
    print("\n1. LEARNING CURVE ANALYSIS")
    print("-" * 40)
    results = analysis.learning_curve_analysis(
        n_features_list=[5, 30, 100],  # Reasonable for small dataset
        cv_folds=5
    )
    analysis.plot_learning_curves(results)
    
    # 2. Bootstrap stability analysis
    print("\n2. BOOTSTRAP STABILITY ANALYSIS")
    print("-" * 40)
    accuracies, feature_stability = analysis.bootstrap_stability_analysis(
        n_bootstrap=100, 
        n_features=15
    )
    
    # 3. Effect size analysis
    print("\n3. EFFECT SIZE ANALYSIS")
    print("-" * 40)
    effect_sizes, p_values, top_indices = analysis.effect_size_analysis()
    
    # 4. Sample size estimation
    print("\n4. SAMPLE SIZE ESTIMATION")
    print("-" * 40)
    required_n = analysis.sample_size_estimation(
        target_accuracy=0.85, 
        confidence_width=0.10
    )
    
    print("\n" + "="*60)
    print("SUMMARY AND RECOMMENDATIONS")
    print("="*60)
    print(f"Current dataset size: {len(analysis.y)} patients")
    print(f"Current estimated performance: {np.mean(accuracies):.1%} ± {np.std(accuracies):.1%}")
    print(f"Largest effect size found: {np.max(effect_sizes):.2f}")
    print(f"Number of highly discriminative features (d>0.5): {np.sum(effect_sizes > 0.5)}")
  
    return analysis, results

class LearningCurveWithFixedFeatures:
    def __init__(self, dataset, selected_indices):
        """
        Learning curve analysis using pre-selected features
        
        Parameters:
        -----------
        dataset : OC_Dataset
            Your loaded dataset
        selected_indices : array-like
            Feature indices selected by effect size analysis
        """
        self.dataset = dataset
        self.selected_indices = selected_indices
        self.X, self.y, self.patient_ids = self._prepare_data()
        
        print(f"Learning Curve Setup:")
        print(f"  Total patients: {len(self.y)}")
        print(f"  Features used: {len(selected_indices)} (fixed selection)")
        print(f"  Cancer: {np.sum(self.y)}, Control: {len(self.y) - np.sum(self.y)}")
        
    def _prepare_data(self):
        """
        Prepare patient-level data with selected features only
        """
        # Same aggregation as before
        patient_spectra = defaultdict(list)
        patient_labels = {}
        
        for i in range(len(self.dataset)):
            intensity, metadata = self.dataset[i]
            patient_id = metadata['patient_id']
            staging = metadata['staging']
            
            patient_spectra[patient_id].append(intensity)
            patient_labels[patient_id] = staging
        
        # Aggregate per patient
        X_full = []
        y_aggregated = []
        patient_ids = []
        
        for patient_id, spectra_list in patient_spectra.items():
            spectra_array = np.stack(spectra_list)
            mean_spectrum = np.mean(spectra_array, axis=0)
            
            X_full.append(mean_spectrum)
            y_aggregated.append(1 if patient_labels[patient_id].lower() == 'cancer' else 0)
            patient_ids.append(patient_id)
        
        X_full = np.array(X_full)
        y = np.array(y_aggregated)
        
        # Extract only selected features
        X_selected = X_full[:, self.selected_indices]
        
        return X_selected, y, patient_ids
    
    def learning_curve_analysis(self, 
                               train_sizes=None,
                               cv_folds=5,
                               models=None,
                               random_state=42):
        """
        Perform learning curve analysis with fixed feature selection
        """
        if train_sizes is None:
            max_size = len(self.y)
            train_sizes = np.linspace(10, max_size, min(8, max_size-2)).astype(int)
            train_sizes = train_sizes[train_sizes >= 8]  # Minimum 8 patients
        
        if models is None:
            models = {
                'Linear SVM': Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', SVC(kernel='linear', random_state=random_state))
                ]),
                'Logistic Regression': Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', LogisticRegression(random_state=random_state, max_iter=1000))
                ]),
                'Random Forest': Pipeline([
                    ('classifier', RandomForestClassifier(n_estimators=100, random_state=random_state))
                ])
            }
        
        results = {}
        
        print(f"\nRunning learning curves with {len(self.selected_indices)} fixed features...")
        
        for model_name, model in models.items():
            print(f"  Testing {model_name}...")
            
            train_scores = []
            val_scores = []
            train_std = []
            val_std = []
            
            for train_size in train_sizes:
                if train_size >= len(self.y):
                    continue
                
                cv_train_scores = []
                cv_val_scores = []
                
                skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
                
                for train_idx, val_idx in skf.split(self.X, self.y):
                    # Limit training set size while maintaining class balance
                    if len(train_idx) > train_size:
                        # Stratified subsampling
                        train_cancer_idx = train_idx[self.y[train_idx] == 1]
                        train_control_idx = train_idx[self.y[train_idx] == 0]
                        
                        # Maintain roughly the same class ratio
                        cancer_ratio = np.mean(self.y)
                        n_cancer = max(1, int(train_size * cancer_ratio))
                        n_control = train_size - n_cancer
                        
                        # Ensure we don't exceed available samples
                        n_cancer = min(n_cancer, len(train_cancer_idx))
                        n_control = min(n_control, len(train_control_idx))
                        
                        if n_cancer > 0 and n_control > 0:
                            np.random.seed(random_state)
                            selected_cancer = np.random.choice(train_cancer_idx, n_cancer, replace=False)
                            selected_control = np.random.choice(train_control_idx, n_control, replace=False)
                            train_idx = np.concatenate([selected_cancer, selected_control])
                        else:
                            continue  # Skip this fold if we can't maintain both classes
                    
                    X_train, X_val = self.X[train_idx], self.X[val_idx]
                    y_train, y_val = self.y[train_idx], self.y[val_idx]
                    
                    # Skip if training set doesn't have both classes
                    if len(np.unique(y_train)) < 2:
                        continue
                        
                    try:
                        # Train and evaluate
                        model.fit(X_train, y_train)
                        
                        train_pred = model.predict(X_train)
                        val_pred = model.predict(X_val)
                        
                        cv_train_scores.append(accuracy_score(y_train, train_pred))
                        cv_val_scores.append(accuracy_score(y_val, val_pred))
                        
                    except Exception as e:
                        print(f"    Warning: Error in fold for size {train_size}: {e}")
                        continue
                
                if len(cv_train_scores) > 0:
                    train_scores.append(np.mean(cv_train_scores))
                    val_scores.append(np.mean(cv_val_scores))
                    train_std.append(np.std(cv_train_scores))
                    val_std.append(np.std(cv_val_scores))
                    
                    print(f"    Size {train_size}: Train={np.mean(cv_train_scores):.3f}, Val={np.mean(cv_val_scores):.3f}")
            
            results[model_name] = {
                'train_sizes': np.array(train_sizes[:len(train_scores)]),
                'train_scores': np.array(train_scores),
                'val_scores': np.array(val_scores),
                'train_std': np.array(train_std),
                'val_std': np.array(val_std)
            }
        
        return results
    
    def plot_learning_curves(self, results, title_suffix=""):
        """
        Plot learning curves with improved visualization
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        # Plot 1: All models together
        for i, (model_name, result) in enumerate(results.items()):
            color = colors[i % len(colors)]
            
            train_sizes = result['train_sizes']
            train_scores = result['train_scores']
            val_scores = result['val_scores']
            train_std = result['train_std']
            val_std = result['val_std']
            
            # Training curves (solid lines)
            ax1.plot(train_sizes, train_scores, 'o-', color=color, 
                    label=f'{model_name} (Train)', alpha=0.8, linewidth=2)
            ax1.fill_between(train_sizes, train_scores - train_std, 
                           train_scores + train_std, alpha=0.1, color=color)
            
            # Validation curves (dashed lines)
            ax1.plot(train_sizes, val_scores, 's--', color=color, 
                    label=f'{model_name} (Val)', alpha=0.8, linewidth=2)
            ax1.fill_between(train_sizes, val_scores - val_std, 
                           val_scores + val_std, alpha=0.1, color=color)
        
        ax1.set_xlabel('Training Set Size (Patients)')
        ax1.set_ylabel('Accuracy')
        ax1.set_title(f'Learning Curves - Fixed Feature Selection{title_suffix}')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0.0, 1.05)
        
        # Add baseline (random guess)
        baseline_acc = max(np.mean(self.y), 1 - np.mean(self.y))  # Majority class accuracy
        ax1.axhline(baseline_acc, color='gray', linestyle=':', alpha=0.7, 
                   label=f'Baseline ({baseline_acc:.2f})')
        
        # Plot 2: Focus on validation curves only for clarity
        for i, (model_name, result) in enumerate(results.items()):
            color = colors[i % len(colors)]
            
            train_sizes = result['train_sizes']
            val_scores = result['val_scores']
            val_std = result['val_std']
            
            ax2.plot(train_sizes, val_scores, 'o-', color=color, 
                    label=model_name, alpha=0.8, linewidth=2, markersize=6)
            ax2.fill_between(train_sizes, val_scores - val_std, 
                           val_scores + val_std, alpha=0.15, color=color)
        
        ax2.set_xlabel('Training Set Size (Patients)')
        ax2.set_ylabel('Validation Accuracy')
        ax2.set_title(f'Validation Performance vs Training Size{title_suffix}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0.0, 1.05)
        ax2.axhline(baseline_acc, color='gray', linestyle=':', alpha=0.7)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def analyze_learning_patterns(self, results):
        """
        Analyze learning curve patterns to make recommendations
        """
        print("\n" + "="*60)
        print("LEARNING CURVE PATTERN ANALYSIS")
        print("="*60)
        
        for model_name, result in results.items():
            train_sizes = result['train_sizes']
            val_scores = result['val_scores']
            
            if len(val_scores) < 3:
                continue
                
            print(f"\n{model_name}:")
            print(f"  Performance at smallest size ({train_sizes[0]}): {val_scores[0]:.3f}")
            print(f"  Performance at largest size ({train_sizes[-1]}): {val_scores[-1]:.3f}")
            
            # Analyze trend
            performance_change = val_scores[-1] - val_scores[0]
            print(f"  Overall change: {performance_change:+.3f}")
            
            # Analyze recent trend (last half of curve)
            mid_point = len(val_scores) // 2
            recent_change = val_scores[-1] - val_scores[mid_point] if mid_point > 0 else 0
            print(f"  Recent trend: {recent_change:+.3f}")
            
            # Pattern classification
            if abs(recent_change) < 0.02 and performance_change > 0:
                pattern = "✅ PLATEAU REACHED - Current size likely adequate"
            elif recent_change > 0.03:
                pattern = "📈 STILL IMPROVING - More patients will help"
            elif recent_change < -0.03:
                pattern = "📉 DECLINING - Possible overfitting, check model complexity"
            elif abs(performance_change) < 0.05:
                pattern = "➡️ STABLE - Consistent performance across sizes"
            else:
                pattern = "❓ UNCLEAR PATTERN - Need more analysis"
            
            print(f"  Pattern: {pattern}")
        
        # Overall recommendation
        print(f"\n" + "="*60)
        print("SAMPLE SIZE RECOMMENDATIONS")
        print("="*60)
        
        best_model = max(results.keys(), key=lambda k: np.mean(results[k]['val_scores']))
        best_result = results[best_model]
        
        final_performance = best_result['val_scores'][-1]
        final_std = best_result['val_std'][-1]
        
        print(f"Best model: {best_model}")
        print(f"Current performance: {final_performance:.3f} ± {final_std:.3f}")
        
        if final_std < 0.06 and final_performance > 0.75:
            recommendation = "✅ CURRENT SIZE ADEQUATE for proof-of-concept"
            next_steps = "Focus on validation with independent cohort"
        elif final_std < 0.10:
            recommendation = "⚠️ CURRENT SIZE REASONABLE but could be improved"
            next_steps = "Collect 20-30 more patients for robust clinical model"
        else:
            recommendation = "❌ NEED MORE PATIENTS for stable clinical performance"
            next_steps = "Target 50-100 total patients"
        
        print(f"Recommendation: {recommendation}")
        print(f"Next steps: {next_steps}")

# Wrapper function for easy use
def run_learning_curves_with_selected_features(dataset, selected_indices):
    """
    Complete learning curve analysis using effect size-selected features
    """
    print("="*80)
    print("LEARNING CURVE ANALYSIS WITH FIXED FEATURE SELECTION")
    print("="*80)
    
    # Initialize analyzer
    lc_analyzer = LearningCurveWithFixedFeatures(dataset, selected_indices)
    
    # Run analysis
    results = lc_analyzer.learning_curve_analysis(
        train_sizes=np.array([8, 12, 16, 20, 24, 28, 32]),  # Customized for your dataset size
        cv_folds=5,
        random_state=42
    )
    
    # Plot results
    title_suffix = f" ({len(selected_indices)} features)"
    lc_analyzer.plot_learning_curves(results, title_suffix)
    
    # Analyze patterns
    lc_analyzer.analyze_learning_patterns(results)
    
    return lc_analyzer, results
    
# Usage example:
if __name__ == "__main__":


    # Load your dataset (modify path as needed)
    data_folder = r'C:\Users\Yifei\Downloads\data_izabella_filtered\data_izabella_filtered'

    preprocessor = SpectrumPreprocessor(cropping=False,
                                        baseline_correction=False,
                                        remove_cosmic_rays=False,
                                        normalization=True,
                                        smoothing=False)

    dataset = OC_Dataset(data_folder, preprocessor, augmentor=None)

    # Run the complete analysis
    print("Starting comprehensive analysis...")
    analysis, results = run_complete_analysis(dataset)

