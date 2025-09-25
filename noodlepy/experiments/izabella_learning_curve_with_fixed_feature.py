import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from collections import defaultdict

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
        ax1.set_ylim(0.4, 1.05)
        
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
        ax2.set_ylim(0.5, 1.05)
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

# Usage example combining with feature selection
if __name__ == "__main__":
    
    # 1. Effect size selection (once)
    selected_indices, feature_info, X_selected, all_effect_sizes = select_features_for_raman_data(
        dataset, top_k=10, min_effect_size=0.7
    )

    # 2. Bootstrap analysis (with selected features)
    bootstrap_analyzer, bootstrap_results = run_bootstrap_with_selected_features(
        dataset, selected_indices, n_bootstrap=200
    )

    # 3. Learning curves (with SAME selected features)
    lc_analyzer, lc_results = run_learning_curves_with_selected_features(
        dataset, selected_indices
    )