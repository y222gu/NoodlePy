import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from noodlepy.utils.izabelladataset import OC_Dataset
import pandas as pd
from noodlepy.utils.spectrumpreprocessor import SpectrumPreprocessor
import scipy.stats as stats
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class BootstrapWithSelectedFeatures:
    def __init__(self, X_selected, y, selected_indices, feature_names=None):
        """
        Initialize bootstrap analysis with pre-selected features
        
        Parameters:
        -----------
        X_selected : array-like, shape (n_patients, n_selected_features)
            Feature matrix with only selected features
        y : array-like, shape (n_patients,)
            Binary labels (0=control, 1=cancer)
        selected_indices : array-like
            Original indices of selected features
        feature_names : list, optional
            Names/descriptions of selected features
        """
        self.X = X_selected
        self.y = y
        self.selected_indices = selected_indices
        self.feature_names = feature_names or [f"Feature_{i}" for i in selected_indices]
        self.n_patients = len(y)
        self.n_features = X_selected.shape[1]
        
        print(f"Bootstrap Analysis Setup:")
        print(f"  Patients: {self.n_patients} (Cancer: {np.sum(y)}, Control: {len(y)-np.sum(y)})")
        print(f"  Selected features: {self.n_features}")
        print(f"  Feature indices: {selected_indices[:10]}..." if len(selected_indices) > 10 else f"  Feature indices: {selected_indices}")
    
    def bootstrap_stability_analysis(self, n_bootstrap=200, models=None, random_state=42):
        """
        Perform bootstrap stability analysis with selected features only
        
        Parameters:
        -----------
        n_bootstrap : int
            Number of bootstrap iterations
        models : dict, optional
            Dictionary of models to test
        random_state : int
            Random seed for reproducibility
        """
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
        
        print(f"\nRunning bootstrap analysis with {n_bootstrap} iterations...")
        
        # Store results for each model
        results = {}
        
        for model_name, model in models.items():
            print(f"  Testing {model_name}...")
            
            bootstrap_accuracies = []
            feature_importances = []
            
            np.random.seed(random_state)
            
            for i in range(n_bootstrap):
                # Bootstrap sampling
                bootstrap_indices = np.random.choice(self.n_patients, self.n_patients, replace=True)
                X_boot = self.X[bootstrap_indices]
                y_boot = self.y[bootstrap_indices]
                
                # Skip if bootstrap sample is not representative
                if len(np.unique(y_boot)) < 2:
                    continue
                
                try:
                    # Cross-validation on bootstrap sample
                    cv_scores = cross_val_score(model, X_boot, y_boot, cv=3, scoring='accuracy')
                    bootstrap_accuracies.append(np.mean(cv_scores))
                    
                    # Feature importance (for Random Forest and Logistic Regression)
                    if 'Random Forest' in model_name:
                        model.fit(X_boot, y_boot)
                        importances = model.named_steps['classifier'].feature_importances_
                        feature_importances.append(importances)
                    elif 'Logistic' in model_name:
                        model.fit(X_boot, y_boot)
                        coefs = np.abs(model.named_steps['classifier'].coef_[0])
                        # Normalize coefficients to [0,1] range for comparison
                        normalized_coefs = coefs / np.sum(coefs)
                        feature_importances.append(normalized_coefs)
                    
                except Exception as e:
                    continue
            
            results[model_name] = {
                'accuracies': np.array(bootstrap_accuracies),
                'feature_importances': np.array(feature_importances) if feature_importances else None
            }
            
            # Print summary statistics
            acc = results[model_name]['accuracies']
            print(f"    {model_name}: {np.mean(acc):.3f} ± {np.std(acc):.3f} (range: {np.min(acc):.3f}-{np.max(acc):.3f})")
        
        return results
    
    def plot_bootstrap_results(self, results):
        """
        Create comprehensive plots of bootstrap results
        """
        n_models = len(results)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Accuracy distributions
        plt.subplot(3, 2, 1)
        colors = ['blue', 'red', 'green', 'orange']
        
        for i, (model_name, result) in enumerate(results.items()):
            accuracies = result['accuracies']
            plt.hist(accuracies, bins=20, alpha=0.6, label=model_name, 
                    color=colors[i % len(colors)], density=True)
            plt.axvline(np.mean(accuracies), color=colors[i % len(colors)], 
                       linestyle='--', alpha=0.8)
        
        plt.xlabel('Bootstrap Accuracy')
        plt.ylabel('Density')
        plt.title('Bootstrap Accuracy Distributions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Accuracy comparison boxplot
        plt.subplot(3, 2, 2)
        accuracy_data = []
        model_names = []
        
        for model_name, result in results.items():
            accuracy_data.extend(result['accuracies'])
            model_names.extend([model_name] * len(result['accuracies']))
        
        df_acc = pd.DataFrame({'Accuracy': accuracy_data, 'Model': model_names})
        
        # Create boxplot
        models_list = list(results.keys())
        box_data = [results[model]['accuracies'] for model in models_list]
        
        bp = plt.boxplot(box_data, labels=models_list, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors[:len(models_list)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        plt.ylabel('Accuracy')
        plt.title('Bootstrap Accuracy Comparison')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 3. Feature importance stability (if available)
        models_with_importance = [(name, result) for name, result in results.items() 
                                 if result['feature_importances'] is not None]
        
        if models_with_importance:
            for idx, (model_name, result) in enumerate(models_with_importance):
                plt.subplot(3, 2, 3 + idx)
                
                feature_importances = result['feature_importances']
                mean_importances = np.mean(feature_importances, axis=0)
                std_importances = np.std(feature_importances, axis=0)
                
                # Bar plot of mean importance with error bars
                x_pos = np.arange(len(mean_importances))
                plt.bar(x_pos, mean_importances, yerr=std_importances, 
                       capsize=3, alpha=0.7, color=colors[idx])
                
                plt.xlabel('Selected Features')
                plt.ylabel('Importance')
                plt.title(f'Feature Importance Stability - {model_name}')
                plt.xticks(x_pos, [f'F{i}' for i in range(len(mean_importances))], rotation=45)
                plt.grid(True, alpha=0.3)
        
        # 4. Feature importance heatmap (if available)
        if models_with_importance:
            plt.subplot(3, 2, 5)
            
            # Use first model with importance for heatmap
            model_name, result = models_with_importance[0]
            feature_importances = result['feature_importances']
            
            # Show heatmap of feature importances across bootstrap samples
            plt.imshow(feature_importances[:50].T, aspect='auto', cmap='viridis', 
                      interpolation='nearest')
            plt.colorbar(label='Importance')
            plt.xlabel('Bootstrap Sample')
            plt.ylabel('Feature Index')
            plt.title(f'Feature Importance Heatmap - {model_name}\n(First 50 bootstrap samples)')
        
        # 5. Summary statistics table
        plt.subplot(3, 2, 6)
        plt.axis('off')
        
        # Create summary table
        summary_data = []
        for model_name, result in results.items():
            acc = result['accuracies']
            summary_data.append([
                model_name,
                f"{np.mean(acc):.3f}",
                f"{np.std(acc):.3f}",
                f"{np.max(acc) - np.min(acc):.3f}",
                f"{len(acc)}"
            ])
        
        table_df = pd.DataFrame(summary_data, 
                               columns=['Model', 'Mean Acc', 'Std Dev', 'Range', 'N Samples'])
        
        # Create table
        table = plt.table(cellText=table_df.values,
                         colLabels=table_df.columns,
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        plt.title('Bootstrap Summary Statistics', pad=20)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def compare_with_previous_results(self, previous_mean_accuracy, previous_std_accuracy):
        """
        Compare new results with previous bootstrap analysis
        """
        print("\n" + "="*60)
        print("COMPARISON WITH PREVIOUS BOOTSTRAP RESULTS")
        print("="*60)
        
        # Get best model results
        best_model = None
        best_mean = 0
        
        # Assuming we have results from bootstrap_stability_analysis
        if hasattr(self, 'latest_results'):
            for model_name, result in self.latest_results.items():
                mean_acc = np.mean(result['accuracies'])
                if mean_acc > best_mean:
                    best_mean = mean_acc
                    best_model = model_name
            
            best_std = np.std(self.latest_results[best_model]['accuracies'])
            best_range = (np.max(self.latest_results[best_model]['accuracies']) - 
                         np.min(self.latest_results[best_model]['accuracies']))
            
            print(f"PREVIOUS RESULTS (All Features):")
            print(f"  Mean Accuracy: {previous_mean_accuracy:.3f} ± {previous_std_accuracy:.3f}")
            print(f"  Estimated Range: ~{previous_mean_accuracy - 2*previous_std_accuracy:.3f} to {previous_mean_accuracy + 2*previous_std_accuracy:.3f}")
            
            print(f"\nNEW RESULTS (Selected Features Only):")
            print(f"  Best Model: {best_model}")
            print(f"  Mean Accuracy: {best_mean:.3f} ± {best_std:.3f}")
            print(f"  Actual Range: {np.min(self.latest_results[best_model]['accuracies']):.3f} to {np.max(self.latest_results[best_model]['accuracies']):.3f}")
            print(f"  Range Width: {best_range:.3f}")
            
            print(f"\nIMPROVEMENT ANALYSIS:")
            stability_improvement = previous_std_accuracy - best_std
            print(f"  Stability Improvement: {stability_improvement:.3f} (lower is better)")
            print(f"  Range Reduction: {stability_improvement * 4:.3f} (approximate)")
            
            if stability_improvement > 0.05:
                print("  ✅ MAJOR IMPROVEMENT: Much more stable results!")
            elif stability_improvement > 0.02:
                print("  ✅ GOOD IMPROVEMENT: Noticeably more stable")
            elif stability_improvement > 0:
                print("  ✅ SLIGHT IMPROVEMENT: Somewhat more stable")
            else:
                print("  ⚠️  NO IMPROVEMENT: Consider different feature selection")
    
    def run_complete_analysis(self, n_bootstrap=200, previous_results=None):
        """
        Run complete bootstrap analysis and generate report
        """
        print("\n" + "="*60)
        print("BOOTSTRAP ANALYSIS WITH SELECTED FEATURES")
        print("="*60)
        
        # Run bootstrap analysis
        results = self.bootstrap_stability_analysis(n_bootstrap=n_bootstrap)
        self.latest_results = results
        
        # Create plots
        self.plot_bootstrap_results(results)
        
        # Generate recommendations
        self._generate_recommendations(results)
        
        return results
    
    def _generate_recommendations(self, results):
        """
        Generate recommendations based on bootstrap results
        """
        print("\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60)
        
        best_model = max(results.keys(), 
                        key=lambda k: np.mean(results[k]['accuracies']))
        best_acc = results[best_model]['accuracies']
        
        mean_acc = np.mean(best_acc)
        std_acc = np.std(best_acc)
        range_acc = np.max(best_acc) - np.min(best_acc)
        
        print(f"Best Model: {best_model}")
        print(f"Performance: {mean_acc:.3f} ± {std_acc:.3f}")
        print(f"Range: {range_acc:.3f}")
        
        # Stability assessment
        if std_acc < 0.05:
            print("\n✅ EXCELLENT STABILITY:")
            print("  - Model performance is very consistent")
            print("  - Ready for validation on new patients")
            print("  - Consider clinical pilot study")
        elif std_acc < 0.08:
            print("\n✅ GOOD STABILITY:")
            print("  - Model performance is reasonably consistent")
            print("  - Collect 10-20 more patients for validation")
            print("  - Good candidate for clinical testing")
        elif std_acc < 0.12:
            print("\n⚠️  MODERATE STABILITY:")
            print("  - Some variability in performance")
            print("  - Collect 30-50 more patients")
            print("  - Consider further feature selection")
        else:
            print("\n❌ POOR STABILITY:")
            print("  - High variability in performance")
            print("  - Need significantly more data")
            print("  - Consider different approach")
        
        # Feature recommendations
        if self.n_features > 15:
            print(f"\n🔧 FEATURE OPTIMIZATION:")
            print(f"  - Currently using {self.n_features} features")
            print(f"  - Try reducing to 5-10 top features")
            print(f"  - Focus on highest effect size features only")

# Wrapper function to easily use with your existing workflow
def run_bootstrap_with_selected_features(dataset, 
                                        selected_indices,
                                        effect_size_threshold=0.7,
                                        n_bootstrap=200,
                                        compare_with_previous=True,
                                        previous_mean_acc=0.707,
                                        previous_std_acc=0.10):
    """
    Complete workflow: feature selection + bootstrap analysis
    
    Parameters:
    -----------
    dataset : OC_Dataset
        Your loaded dataset
    selected_indices : array-like
        Indices of features selected by effect size
    effect_size_threshold : float
        Minimum effect size used for selection
    n_bootstrap : int
        Number of bootstrap iterations
    compare_with_previous : bool
        Whether to compare with previous results
    previous_mean_acc : float
        Mean accuracy from previous bootstrap analysis
    previous_std_acc : float
        Standard deviation from previous bootstrap analysis
    """
    
    # 1. Prepare data (same as in feature selection)
    print("Step 1: Preparing patient-level data...")
    
    from collections import defaultdict
    patient_spectra = defaultdict(list)
    patient_labels = {}
    
    for i in range(len(dataset)):
        intensity, metadata = dataset[i]
        patient_id = metadata['patient_id']
        staging = metadata['staging']
        
        patient_spectra[patient_id].append(intensity)
        patient_labels[patient_id] = staging
    
    # Aggregate per patient
    X_aggregated = []
    y_aggregated = []
    patient_ids = []
    
    for patient_id, spectra_list in patient_spectra.items():
        spectra_array = np.stack(spectra_list)
        mean_spectrum = np.mean(spectra_array, axis=0)
        
        X_aggregated.append(mean_spectrum)
        y_aggregated.append(1 if patient_labels[patient_id].lower() == 'cancer' else 0)
        patient_ids.append(patient_id)
    
    X_full = np.array(X_aggregated)
    y = np.array(y_aggregated)
    
    # 2. Extract selected features
    X_selected = X_full[:, selected_indices]
    
    print(f"Step 2: Using {len(selected_indices)} selected features")
    print(f"Original features: {X_full.shape[1]} → Selected: {X_selected.shape[1]}")
    
    # 3. Run bootstrap analysis
    bootstrap_analyzer = BootstrapWithSelectedFeatures(X_selected, y, selected_indices)
    results = bootstrap_analyzer.run_complete_analysis(n_bootstrap=n_bootstrap)
    
    # 4. Compare with previous results if requested
    if compare_with_previous:
        bootstrap_analyzer.compare_with_previous_results(previous_mean_acc, previous_std_acc)
    
    return bootstrap_analyzer, results



def select_features_for_raman_data(dataset, 
                                   top_k=15, 
                                   min_effect_size=0.6,
                                   visualize=True):
    """
    Select best features from your Raman dataset using effect size
    
    This function works directly with your OC_Dataset class
    """
    print("=" * 60)
    print("EFFECT SIZE-BASED FEATURE SELECTION FOR RAMAN DATA")
    print("=" * 60)
    
    # 1. Extract patient-level aggregated data (same as your learning curve analysis)
    print("Step 1: Extracting and aggregating patient data...")
    
    from collections import defaultdict
    patient_spectra = defaultdict(list)
    patient_labels = {}
    
    for i in range(len(dataset)):
        intensity, metadata = dataset[i]
        patient_id = metadata['patient_id']
        staging = metadata['staging']
        
        patient_spectra[patient_id].append(intensity)
        patient_labels[patient_id] = staging
    
    # Aggregate spectra per patient (mean only for simplicity)
    X_aggregated = []
    y_aggregated = []
    patient_ids = []
    
    for patient_id, spectra_list in patient_spectra.items():
        spectra_array = np.stack(spectra_list)
        mean_spectrum = np.mean(spectra_array, axis=0)  # Just mean this time
        
        X_aggregated.append(mean_spectrum)
        y_aggregated.append(1 if patient_labels[patient_id].lower() == 'cancer' else 0)
        patient_ids.append(patient_id)
    
    X = np.array(X_aggregated)
    y = np.array(y_aggregated)
    
    print(f"Patient-level data: {X.shape}")
    print(f"Cancer: {np.sum(y)}, Control: {len(y) - np.sum(y)}")
    
    # 2. Calculate effect sizes for all features
    print("\nStep 2: Calculating effect sizes...")
    
    cancer_mask = y == 1
    control_mask = y == 0
    X_cancer = X[cancer_mask]
    X_control = X[control_mask]
    
    effect_sizes = []
    p_values = []
    cancer_means = []
    control_means = []
    
    for feature_idx in range(X.shape[1]):
        cancer_vals = X_cancer[:, feature_idx]
        control_vals = X_control[:, feature_idx]
        
        # Effect size calculation
        mean_diff = np.mean(cancer_vals) - np.mean(control_vals)
        pooled_std = np.sqrt(((len(cancer_vals)-1) * np.var(cancer_vals) + 
                             (len(control_vals)-1) * np.var(control_vals)) / 
                            (len(cancer_vals) + len(control_vals) - 2))
        
        d = abs(mean_diff / pooled_std) if pooled_std > 0 else 0
        effect_sizes.append(d)
        
        cancer_means.append(np.mean(cancer_vals))
        control_means.append(np.mean(control_vals))
        
        # Statistical significance
        t_stat, p_val = stats.ttest_ind(cancer_vals, control_vals)
        p_values.append(p_val)
    
    effect_sizes = np.array(effect_sizes)
    p_values = np.array(p_values)
    
    # 3. Feature selection strategy
    print("\nStep 3: Selecting discriminative features...")
    
    # Strategy 1: Top K by effect size
    top_indices = np.argsort(effect_sizes)[::-1]
    
    # Strategy 2: Above threshold
    good_indices = np.where(effect_sizes >= min_effect_size)[0]
    
    # Combine: Take top_k from good features, or just top_k if not enough good ones
    if len(good_indices) >= top_k:
        selected_indices = top_indices[:top_k]  # Top K overall
    else:
        selected_indices = good_indices  # All good features
    
    print(f"Selected {len(selected_indices)} features")
    print(f"Effect size range: {effect_sizes[selected_indices].min():.3f} - {effect_sizes[selected_indices].max():.3f}")
    
    # 4. Create summary table
    feature_summary = pd.DataFrame({
        'Feature_Index': selected_indices,
        'Effect_Size': effect_sizes[selected_indices],
        'P_Value': p_values[selected_indices],
        'Cancer_Mean': np.array(cancer_means)[selected_indices],
        'Control_Mean': np.array(control_means)[selected_indices],
        'Mean_Difference': np.array(cancer_means)[selected_indices] - np.array(control_means)[selected_indices],
        'Interpretation': ['Large' if d > 0.8 else 'Medium' if d > 0.5 else 'Small' 
                          for d in effect_sizes[selected_indices]]
    })
    
    feature_summary = feature_summary.sort_values('Effect_Size', ascending=False).reset_index(drop=True)
    
    print("\nTop 10 Selected Features:")
    print(feature_summary.head(10).to_string(index=False))
    
    # 5. Validation: Test model performance with selected features
    print(f"\nStep 4: Validating selected features...")
    
    X_selected = X[:, selected_indices]
    
    # Simple linear SVM with selected features
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='linear', random_state=42))
    ])
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_selected, y, cv=5, scoring='accuracy')
    
    print(f"Cross-validation with {len(selected_indices)} selected features:")
    print(f"Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"Individual fold scores: {cv_scores}")
    
    # 6. Visualization
    if visualize:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Effect size distribution
        ax1.hist(effect_sizes, bins=50, alpha=0.7, edgecolor='black')
        ax1.axvline(min_effect_size, color='red', linestyle='--', 
                   label=f'Threshold ({min_effect_size})')
        ax1.axvline(effect_sizes[selected_indices].min(), color='green', linestyle='--', 
                   label=f'Selected min ({effect_sizes[selected_indices].min():.2f})')
        ax1.set_xlabel('Effect Size (Cohen\'s d)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Effect Sizes')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Selected features
        ax2.bar(range(len(selected_indices)), effect_sizes[selected_indices], alpha=0.7)
        ax2.set_xlabel('Selected Features (ranked)')
        ax2.set_ylabel('Effect Size')
        ax2.set_title(f'Top {len(selected_indices)} Selected Features')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Feature locations
        ax3.scatter(range(len(effect_sizes)), effect_sizes, alpha=0.5, s=10)
        ax3.scatter(selected_indices, effect_sizes[selected_indices], 
                   color='red', s=30, label='Selected')
        ax3.set_xlabel('Feature Index (Spectral Position)')
        ax3.set_ylabel('Effect Size')
        ax3.set_title('Effect Size vs Feature Index')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Validation scores
        ax4.bar(['CV Mean'], [cv_scores.mean()], yerr=[cv_scores.std()], 
                capsize=5, alpha=0.7, color='green')
        ax4.set_ylabel('Accuracy')
        ax4.set_title('Cross-Validation Performance')
        ax4.set_ylim(0, 1)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Plot individual feature distributions for top features
        n_show = min(8, len(selected_indices))
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i in range(n_show):
            feature_idx = selected_indices[i]
            
            cancer_vals = X_cancer[:, feature_idx]
            control_vals = X_control[:, feature_idx]
            
            axes[i].hist(control_vals, bins=8, alpha=0.7, label='Control', 
                        color='blue', density=True)
            axes[i].hist(cancer_vals, bins=8, alpha=0.7, label='Cancer', 
                        color='red', density=True)
            
            d = effect_sizes[feature_idx]
            axes[i].set_title(f'Feature {feature_idx}\nEffect Size = {d:.3f}')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle('Individual Feature Distributions (Top 8)', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    return selected_indices, feature_summary, X_selected, effect_sizes



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
        ax1.set_ylim(0, 1.05)
        
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
        ax2.set_ylim(0, 1.05)
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

# Usage example
if __name__ == "__main__":
    # Complete workflow combining effect size selection with improved bootstrap analysis

    # Step 1: Load your dataset (modify path as needed)
    data_folder = r'C:\Users\Yifei\Downloads\data_izabella_filtered\data_izabella_filtered'

    preprocessor = SpectrumPreprocessor(cropping=False,
                                        baseline_correction=False,
                                        remove_cosmic_rays=False,
                                        normalization=True,
                                        smoothing=False)

    dataset = OC_Dataset(data_folder, preprocessor, augmentor=None)

    print("="*80)
    print("COMPLETE WORKFLOW: EFFECT SIZE SELECTION + BOOTSTRAP ANALYSIS")
    print("="*80)

    # Step 2: Effect size-based feature selection
    print("\n🔍 STEP 1: EFFECT SIZE-BASED FEATURE SELECTION")
    print("-" * 60)

    selected_indices, feature_info, X_selected, all_effect_sizes = select_features_for_raman_data(
        dataset, 
        top_k=12,            # Select top 12 features
        min_effect_size=0.65, # Medium-large effect size threshold
        visualize=True       # Show plots
    )

    bootstrap_analyzer, results = run_bootstrap_with_selected_features(
        dataset=dataset,
        selected_indices=selected_indices,
        n_bootstrap=150,      # Reduced for faster execution, increase to 200+ for final analysis
        compare_with_previous=True,
        previous_mean_acc=0.707,  # Your previous bootstrap result
        previous_std_acc=0.095    # Estimated from your previous plot
    )

    lc_analyzer, lc_results = run_learning_curves_with_selected_features(
        dataset, selected_indices
    )