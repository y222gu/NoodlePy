import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
from noodlepy.utils.izabelladataset import OC_Dataset
from noodlepy.utils.spectrumpreprocessor import SpectrumPreprocessor
from matplotlib.colors import LinearSegmentedColormap

# Set global plot font sizes (titles, labels, legend, ticks)
plt.rcParams.update({
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.titlesize': 18
})

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


def load_results(file_path):
    """Load the saved k-fold results."""
    with open(file_path, 'rb') as f:
        results = pickle.load(f)
    
    results_df = results['results_df']
    detailed_predictions_df = results['detailed_predictions_df']
    raman_shift = results['raman_shift']
    
    print(f"Loaded results from {file_path}")
    print(f"  - Results shape: {results_df.shape}")
    print(f"  - Detailed predictions shape: {detailed_predictions_df.shape}")
    
    return results_df, detailed_predictions_df, raman_shift

def plot_accuracy_histograms(results_df, save_dir=None, file_prefix=''):
    """
    Plot histograms of k-fold accuracies across iterations with a black theme.

    Args:
        results_df: DataFrame containing the results
        save_dir: Directory to save plots (optional)
        file_prefix: Prefix for saved file names
    """
    # Extract all k-fold accuracies across iterations
    train_acc_spectrum = []
    train_acc_patient = []
    val_acc_spectrum = []
    val_acc_patient = []

    for _, row in results_df.iterrows():
        train_acc_spectrum.append(row['average_train_accuracy_spectrum'])
        train_acc_patient.append(row['average_train_accuracy_patient'])
        val_acc_spectrum.append(row['average_val_accuracy_spectrum'])
        val_acc_patient.append(row['average_val_accuracy_patient'])

    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    # Black figure background
    fig.patch.set_facecolor('black')

    fig.suptitle(
        f'Distribution of K-Fold Accuracies Across {len(results_df)} Iterations',
        fontsize=20, fontweight='bold', color='white'
    )

    # Define data and titles for each subplot
    data_list = [
        (train_acc_spectrum, 'Train Accuracy (Spectrum Level)', axes[0, 0]),
        (train_acc_patient, 'Train Accuracy (Patient Level)', axes[0, 1]),
        (val_acc_spectrum, 'Validation Accuracy (Spectrum Level)', axes[1, 0]),
        (val_acc_patient, 'Validation Accuracy (Patient Level)', axes[1, 1])
    ]

    # Plot each histogram with black-themed styling
    for i, (data, title, ax) in enumerate(data_list):
        # Set axis (panel) background to black
        ax.set_facecolor('black')

        # Plot histogram; use a color visible on black and white edges
        n, bins, patches = ax.hist(data, bins=30, edgecolor='white', alpha=0.85, color='steelblue')

        # Add statistics
        mean_val = np.mean(data)
        std_val = np.std(data)
        median_val = np.median(data)

        # Add vertical lines for mean and median
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
        ax.axvline(median_val, color='lime', linestyle='--', linewidth=2, label=f'Median: {median_val:.3f}')

        # Set labels and title with white color
        ax.set_xlabel('Accuracy', fontsize=20, color='white')
        ax.set_ylabel('Frequency', fontsize=20, color='white')
        ax.set_title(f'{title}\n(μ={mean_val:.3f}, σ={std_val:.3f})', fontsize=20, color='white')

        # Style legend for black background
        leg = ax.legend(frameon=True, fontsize=20)
        leg.get_frame().set_facecolor('black')
        leg.get_frame().set_edgecolor('white')
        for text in leg.get_texts():
            text.set_color('white')

        # Ticks and spines white and larger
        ax.tick_params(colors='white', labelsize=20)
        plt.setp(ax.get_xticklabels(), color='white', fontsize=20)
        plt.setp(ax.get_yticklabels(), color='white', fontsize=20)
        for spine in ax.spines.values():
            spine.set_color('white')

        # Grid subtle gray
        ax.grid(True, alpha=0.25, color='gray')

        # Set x-axis limits: first two subplots tight near 1.0, last two wider
        if i < 2:
            ax.set_xlim([0.96, 1.0])
        else:
            ax.set_xlim([0.4, 1.0])

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for the suptitle

    # Save figure if directory provided (preserve facecolor)
    if save_dir:
        save_path = Path(save_dir) / f'{file_prefix}accuracy_histograms.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"Saved accuracy histograms to {save_path}")

    plt.show()

    # Print summary statistics
    print("\n=== Accuracy Statistics ===")
    print(f"Train Accuracy (Spectrum Level): {np.mean(train_acc_spectrum):.3f} ± {np.std(train_acc_spectrum):.3f}")
    print(f"Train Accuracy (Patient Level): {np.mean(train_acc_patient):.3f} ± {np.std(train_acc_patient):.3f}")
    print(f"Val Accuracy (Spectrum Level): {np.mean(val_acc_spectrum):.3f} ± {np.std(val_acc_spectrum):.3f}")
    print(f"Val Accuracy (Patient Level): {np.mean(val_acc_patient):.3f} ± {np.std(val_acc_patient):.3f}")

    return {
        'train_spectrum': train_acc_spectrum,
        'train_patient': train_acc_patient,
        'val_spectrum': val_acc_spectrum,
        'val_patient': val_acc_patient
    }


def filter_iterations_by_threshold(results_df, thresholds):
    """
    Filter iterations based on accuracy thresholds.
    
    Args:
        results_df: DataFrame containing the results
        thresholds: Dictionary with keys:
            - 'train_spectrum_min': minimum train accuracy at spectrum level
            - 'train_patient_min': minimum train accuracy at patient level
            - 'val_spectrum_min': minimum validation accuracy at spectrum level
            - 'val_patient_min': minimum validation accuracy at patient level
    
    Returns:
        filtered_df: DataFrame with iterations meeting all thresholds
        selected_iterations: List of iteration indices that meet thresholds
    """
    # Initialize mask with all True
    mask = pd.Series([True] * len(results_df))
    
    # Apply each threshold if provided
    if 'train_spectrum_min' in thresholds:
        mask &= results_df['average_train_accuracy_spectrum'].apply(
            lambda x: np.mean(x) >= thresholds['train_spectrum_min']
        )
    
    if 'train_patient_min' in thresholds:
        mask &= results_df['average_train_accuracy_patient'].apply(
            lambda x: np.mean(x) >= thresholds['train_patient_min']
        )
    
    if 'val_spectrum_min' in thresholds:
        mask &= results_df['average_val_accuracy_spectrum'].apply(
            lambda x: np.mean(x) >= thresholds['val_spectrum_min']
        )
    
    if 'val_patient_min' in thresholds:
        mask &= results_df['average_val_accuracy_patient'].apply(
            lambda x: np.mean(x) >= thresholds['val_patient_min']
        )
    
    filtered_df = results_df[mask].copy()
    selected_iterations = filtered_df['iteration'].tolist()
    
    print(f"\n=== Filtering Results ===")
    print(f"Applied thresholds: {thresholds}")
    print(f"Iterations meeting thresholds: {len(selected_iterations)} out of {len(results_df)}")
    print(f"Selected iterations: {selected_iterations[:10]}..." if len(selected_iterations) > 10 else f"Selected iterations: {selected_iterations}")
    
    return filtered_df, selected_iterations


def create_spectrum_frequency_heatmap(filtered_df, detailed_predictions_df, 
                                     dataset_path, save_dir=None, file_prefix=''):
    """
    Create a heatmap showing the frequency of spectrum selection for each patient.
    Only display spectrum IDs from 0 to 32 (inclusive) on the x-axis.
    
    Args:
        filtered_df: Filtered DataFrame with iterations meeting thresholds
        detailed_predictions_df: DataFrame with detailed predictions
        dataset_path: Path to the original dataset to get patient metadata
        save_dir: Directory to save the heatmap (optional)
        file_prefix: Prefix for saved file name
    """
    
    # Load dataset to get patient metadata
    print("Loading dataset to get patient metadata...")
    preprocessor = SpectrumPreprocessor(
        cropping=True,
        baseline_correction=True,
        remove_cosmic_rays=False,
        normalization=True,
        smoothing=True)
    
    dataset = OC_Dataset(dataset_path, preprocessor, augmentor=None)
    
    # Get patient groups and their labels
    patient_groups = defaultdict(list)
    patient_labels = {}
    spectrum_to_patient = {}
    
    for idx in range(len(dataset)):
        _, _, metadata = dataset[idx]
        patient_id = metadata['patient_id']
        spectrum_id = metadata['spectrum_id']
        staging = metadata['staging'].lower()
        
        patient_groups[patient_id].append(spectrum_id)
        spectrum_to_patient[spectrum_id] = patient_id
        
        if patient_id not in patient_labels:
            patient_labels[patient_id] = 'cancer' if ('cancer' in staging or 'ca' in staging) else 'control'
    
    # Count spectrum selection frequency
    spectrum_selection_count = defaultdict(lambda: defaultdict(int))
    
    for _, row in filtered_df.iterrows():
        selected_spectra = row['selected_spectra_index']
        
        # Count each selected spectrum (increment by 0.5 so totals are effectively divided by 2)
        for patient_id, spectrum_indices in selected_spectra.items():
            for spectrum_idx in spectrum_indices:
                _, _, metadata = dataset[spectrum_idx]
                spectrum_id = metadata['spectrum_id']
                spectrum_selection_count[patient_id][spectrum_id] += 1
    
    # Organize data for heatmap
    cancer_patients = sorted([pid for pid in patient_labels if patient_labels[pid] == 'cancer'])
    control_patients = sorted([pid for pid in patient_labels if patient_labels[pid] == 'control'])
    all_patients = cancer_patients + control_patients
    
    # Get all unique spectrum counts per patient and determine display columns (0..32)
    max_spectra = max([len(patient_groups[pid]) for pid in all_patients]) if all_patients else 0
    display_cols = min(max_spectra, 33)  # show only spectrum ids 0..32 (33 columns)
    
    # Create matrix for heatmap (only first display_cols spectra per patient)
    heatmap_data = []
    patient_names = []
    
    for patient_id in all_patients:
        # sort spectra deterministically
        patient_spectra = sorted(patient_groups[patient_id])
        row_data = []
        
        # take only up to display_cols spectra (if a patient has more, we only display the first 33)
        for spectrum_id in patient_spectra[:display_cols]:
            count = spectrum_selection_count[patient_id].get(spectrum_id, 0)
            row_data.append(count)
        
        # Pad with NaN if patient has fewer than display_cols spectra
        while len(row_data) < display_cols:
            row_data.append(np.nan)
        
        heatmap_data.append(row_data)
        patient_names.append(f"{patient_id} ({patient_labels[patient_id][:3].upper()})")
    
    # Convert to numpy array (this is the displayed heatmap array)
    heatmap_array = np.array(heatmap_data)
    
    # Mask NaNs so they can be colored specially
    masked_array = np.ma.masked_invalid(heatmap_array)
    
    cmap = sns.color_palette("copper", as_cmap=True)
    cmap.set_bad(color='black')  # NaN values will be black (matches background)
    
    # Create figure sized by number of displayed columns and patients and set black background/margins
    fig, ax = plt.subplots(figsize=(max(12, display_cols * 0.3), 
                                   max(8, len(all_patients) * 0.3)))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    
    # Create heatmap using masked array so NaNs appear in black
    im = ax.imshow(masked_array, cmap=cmap, aspect='auto', interpolation='nearest')
    
    # Set ticks and labels for 0..display_cols-1 and make them white
    ax.set_xticks(np.arange(display_cols))
    ax.set_yticks(np.arange(len(all_patients)))
    ax.set_xticklabels([str(i) for i in range(display_cols)], fontsize=20, color='white')  # show 0..32
    ax.set_yticklabels(patient_names, fontsize=20, color='white')
    
    # Rotate the tick labels for better readability and ensure tick colors are white
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", color='white', fontsize=20)
    plt.setp(ax.get_yticklabels(), color='white', fontsize=20)
    ax.tick_params(colors='white', labelsize=20)  # ticks and tick labels to white
    
    # Set spines to white
    for spine in ax.spines.values():
        spine.set_color('white')
    
    # Add colorbar and style it white
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Selection Frequency', rotation=270, labelpad=20, fontsize=20, color='white')
    # Colorbar tick labels to white and outline to white, background black
    plt.setp(cbar.ax.get_yticklabels(), color='white', fontsize=20)
    cbar.outline.set_edgecolor('white')
    cbar.ax.set_facecolor('black')
    
    # Add divider line between cancer and control patients (white, thick dashed)
    if len(cancer_patients) > 0:
        ax.axhline(y=len(cancer_patients) - 0.5, color='white', linewidth=5, linestyle='--', alpha=0.9)
    
    # Set title and labels in white
    total_iterations = len(filtered_df)
    ax.set_title(f'Spectrum Selection Frequency Heatmap\n'
                f'(Based on {total_iterations} iterations meeting thresholds)\n'
                f'Cancer Patients: {len(cancer_patients)}, Control Patients: {len(control_patients)}',
                fontsize=20, fontweight='bold', color='white')
    ax.set_xlabel('Spectrum ID', fontsize=20, color='white')
    ax.set_ylabel('Patient ID (Type)', fontsize=20, color='white')
    
    # Add grid (use subtle gray lines that show on black background)
    ax.set_xticks(np.arange(display_cols + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(all_patients) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)
    ax.tick_params(which='minor', length=0)
    
    plt.tight_layout()
    
    # Save figure if directory provided (preserve facecolor)
    if save_dir:
        save_path = Path(save_dir) / f'{file_prefix}spectrum_frequency_heatmap.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"Saved heatmap to {save_path}")
    
    plt.show()
    
    # Print summary statistics (keep prints white in console is not applicable; just print)
    print("\n=== Spectrum Selection Summary ===")
    print(f"Total patients: {len(all_patients)}")
    print(f"  - Cancer patients: {len(cancer_patients)}")
    print(f"  - Control patients: {len(control_patients)}")
    print(f"Displayed spectra per patient (0..{display_cols-1}): {display_cols}")
    
    # Calculate average selection frequency per patient type (only for displayed spectra)
    cancer_selections = []
    control_selections = []
    
    for patient_id in cancer_patients:
        for spectrum_id in sorted(patient_groups[patient_id])[:display_cols]:
            count = spectrum_selection_count[patient_id].get(spectrum_id, 0)
            cancer_selections.append(count)
    
    for patient_id in control_patients:
        for spectrum_id in sorted(patient_groups[patient_id])[:display_cols]:
            count = spectrum_selection_count[patient_id].get(spectrum_id, 0)
            control_selections.append(count)
    
    if cancer_selections:
        print(f"Cancer spectra - Avg selection frequency: {np.mean(cancer_selections):.2f} ± {np.std(cancer_selections):.2f}")
    if control_selections:
        print(f"Control spectra - Avg selection frequency: {np.mean(control_selections):.2f} ± {np.std(control_selections):.2f}")
    
    return heatmap_array, patient_names, spectrum_selection_count

def list_unique_patients(results_df):
    """
    Simple function to list all unique patient IDs used across all iterations.
    
    Args:
        results_df: DataFrame containing the results with 'selected_spectra_index' column
    
    Returns:
        unique_patients: Set of unique patient IDs
    """
    unique_patients = set()
    
    # Iterate through all iterations
    for _, row in results_df.iterrows():
        selected_spectra = row['selected_spectra_index']
        # Add all patient IDs from this iteration
        unique_patients.update(selected_spectra.keys())
    
    # Convert to sorted list for display
    unique_patients_list = sorted(unique_patients)
    
    print("\n=== Unique Patient IDs Used Across All Iterations ===")
    print(f"Total unique patients: {len(unique_patients_list)}")
    print("\nPatient IDs:")
    
    # Print in columns for better readability
    n_cols = 5  # Number of columns to display
    for i in range(0, len(unique_patients_list), n_cols):
        row = unique_patients_list[i:i+n_cols]
        print("  " + "\t".join(f"{pid:15}" for pid in row))
    
    return unique_patients

def analyze_spectrum_coverage(results_df, dataset_path, save_dir=None, file_prefix=''):
    """
    Analyze if every spectrum of every patient was selected at least once across all iterations.
    
    Args:
        results_df: DataFrame containing the results with 'selected_spectra_index' column
        dataset_path: Path to the original dataset to get all available spectra
        save_dir: Directory to save outputs (optional)
        file_prefix: Prefix for saved file names
    
    Returns:
        spectrum_coverage_stats: Dictionary with spectrum coverage statistics
    """
    from noodlepy.utils.izabelladataset import OC_Dataset
    from noodlepy.utils.spectrumpreprocessor import SpectrumPreprocessor
    
    # Load dataset to get all available spectra
    print("Loading dataset to analyze spectrum coverage...")
    preprocessor = SpectrumPreprocessor(
        cropping=True,
        baseline_correction=True,
        remove_cosmic_rays=False,
        normalization=True,
        smoothing=True)
    
    dataset = OC_Dataset(dataset_path, preprocessor, augmentor=None)
    
    # Build complete inventory of all spectra in dataset
    all_spectra_in_dataset = {}  # {patient_id: {spectrum_id: dataset_index}}
    patient_labels = {}
    spectrum_to_index = {}  # {(patient_id, spectrum_id): dataset_index}
    
    for idx in range(len(dataset)):
        _, _, metadata = dataset[idx]
        patient_id = metadata['patient_id']
        spectrum_id = metadata['spectrum_id']
        staging = metadata['staging'].lower()
        
        if patient_id not in all_spectra_in_dataset:
            all_spectra_in_dataset[patient_id] = {}
            patient_labels[patient_id] = 'cancer' if ('cancer' in staging or 'ca' in staging) else 'control'
        
        all_spectra_in_dataset[patient_id][spectrum_id] = idx
        spectrum_to_index[(patient_id, spectrum_id)] = idx
    
    # Track which spectra were actually selected across all iterations
    selected_spectra_indices = set()  # dataset indices that were selected
    spectrum_selection_count = defaultdict(int)  # {(patient_id, spectrum_id): count}
    
    # Analyze all iterations
    total_iterations = len(results_df)
    print(f"Analyzing {total_iterations} iterations for spectrum coverage...")
    
    for _, row in results_df.iterrows():
        selected_spectra = row['selected_spectra_index']
        
        # Track all selected spectrum indices
        for patient_id, spectrum_indices in selected_spectra.items():
            for idx in spectrum_indices:
                selected_spectra_indices.add(idx)
                
                # Get spectrum_id for this index
                _, _, metadata = dataset[idx]
                spectrum_id = metadata['spectrum_id']
                spectrum_selection_count[(patient_id, spectrum_id)] += 1
    
    # Find never-selected spectra
    never_selected_spectra = []
    patient_spectrum_coverage = defaultdict(lambda: {'total': 0, 'selected': 0, 'never_selected': []})
    
    for patient_id, spectra_dict in all_spectra_in_dataset.items():
        for spectrum_id, dataset_idx in spectra_dict.items():
            patient_spectrum_coverage[patient_id]['total'] += 1
            
            if dataset_idx in selected_spectra_indices:
                patient_spectrum_coverage[patient_id]['selected'] += 1
            else:
                patient_spectrum_coverage[patient_id]['never_selected'].append(spectrum_id)
                never_selected_spectra.append({
                    'patient_id': patient_id,
                    'spectrum_id': spectrum_id,
                    'dataset_index': dataset_idx,
                    'patient_type': patient_labels[patient_id]
                })
    
    # Calculate statistics
    total_spectra_in_dataset = sum(len(spectra) for spectra in all_spectra_in_dataset.values())
    total_spectra_selected = len(selected_spectra_indices)
    total_never_selected = len(never_selected_spectra)
    coverage_percentage = 100 * total_spectra_selected / total_spectra_in_dataset
    
    # Separate by patient type
    cancer_patients = [pid for pid in patient_labels if patient_labels[pid] == 'cancer']
    control_patients = [pid for pid in patient_labels if patient_labels[pid] == 'control']
    
    cancer_never_selected = [s for s in never_selected_spectra if s['patient_type'] == 'cancer']
    control_never_selected = [s for s in never_selected_spectra if s['patient_type'] == 'control']
    
    # Print comprehensive summary
    print("\n" + "="*60)
    print("SPECTRUM COVERAGE ANALYSIS RESULTS")
    print("="*60)
    
    print(f"\nOverall Statistics:")
    print(f"  Total spectra in dataset: {total_spectra_in_dataset}")
    print(f"  Total unique spectra selected: {total_spectra_selected}")
    print(f"  Total never-selected spectra: {total_never_selected}")
    print(f"  Overall coverage: {coverage_percentage:.2f}%")
    
    print(f"\nBy Patient Type:")
    print(f"  Cancer patients: {len(cancer_patients)}")
    print(f"    Never-selected cancer spectra: {len(cancer_never_selected)}")
    print(f"  Control patients: {len(control_patients)}")
    print(f"    Never-selected control spectra: {len(control_never_selected)}")
    
    if total_never_selected == 0:
        print("\n✓ EXCELLENT! All spectra were selected at least once across all iterations.")
    else:
        print(f"\n⚠ WARNING: {total_never_selected} spectra were NEVER selected in any iteration!")
        
        # Show details of never-selected spectra
        print("\nNever-selected spectra by patient:")
        for patient_id in sorted(patient_spectrum_coverage.keys()):
            coverage = patient_spectrum_coverage[patient_id]
            if coverage['never_selected']:
                patient_type = patient_labels[patient_id]
                print(f"\n  Patient {patient_id} ({patient_type}):")
                print(f"    Total spectra: {coverage['total']}")
                print(f"    Selected: {coverage['selected']}")
                print(f"    Never selected: {len(coverage['never_selected'])}")
                print(f"    Never-selected spectrum IDs: {sorted(coverage['never_selected'])}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Spectrum Coverage Analysis Across All Iterations', fontsize=20, fontweight='bold')
    
    # 1. Overall coverage pie chart
    ax1 = axes[0, 0]
    sizes = [total_spectra_selected, total_never_selected]
    labels = [f'Selected\n({total_spectra_selected})', f'Never Selected\n({total_never_selected})']
    colors = ['#2ecc71', '#e74c3c'] if total_never_selected > 0 else ['#2ecc71', '#95a5a6']
    explode = (0.05, 0.1) if total_never_selected > 0 else (0.05, 0)
    
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.set_title(f'Overall Spectrum Coverage\n(Total: {total_spectra_in_dataset} spectra)', fontsize=20)
    
    # 2. Coverage by patient type
    ax2 = axes[0, 1]
    
    # Calculate coverage for each patient type
    cancer_total_spectra = sum(patient_spectrum_coverage[pid]['total'] for pid in cancer_patients)
    control_total_spectra = sum(patient_spectrum_coverage[pid]['total'] for pid in control_patients)
    cancer_selected_spectra = sum(patient_spectrum_coverage[pid]['selected'] for pid in cancer_patients)
    control_selected_spectra = sum(patient_spectrum_coverage[pid]['selected'] for pid in control_patients)
    
    x = np.arange(2)
    width = 0.35
    
    total_bars = ax2.bar(x - width/2, [cancer_total_spectra, control_total_spectra], 
                         width, label='Total Spectra', color='lightgray', edgecolor='black')
    selected_bars = ax2.bar(x + width/2, [cancer_selected_spectra, control_selected_spectra], 
                            width, label='Selected Spectra', color='#2ecc71', edgecolor='black')
    
    ax2.set_xlabel('Patient Type', fontsize=20)
    ax2.set_ylabel('Number of Spectra', fontsize=20)
    ax2.set_title('Spectrum Coverage by Patient Type', fontsize=20)
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Cancer', 'Control'], fontsize=20)
    ax2.legend(fontsize=20)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(labelsize=20)
    
    # Add value labels on bars
    for bars in [total_bars, selected_bars]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=16)
    
    # 3. Patient-level coverage heatmap
    ax3 = axes[1, 0]
    
    # Prepare data for heatmap
    all_patients = cancer_patients + control_patients
    coverage_matrix = []
    patient_names = []
    
    for patient_id in all_patients:
        coverage = patient_spectrum_coverage[patient_id]
        coverage_pct = 100 * coverage['selected'] / coverage['total'] if coverage['total'] > 0 else 0
        coverage_matrix.append([coverage_pct])
        patient_type = patient_labels[patient_id][:3].upper()
        patient_names.append(f"{patient_id} ({patient_type})")
    
    # Create heatmap
    im = ax3.imshow(coverage_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    ax3.set_yticks(np.arange(len(all_patients)))
    ax3.set_yticklabels(patient_names, fontsize=20)
    ax3.set_xticks([0])
    ax3.set_xticklabels(['Coverage %'], fontsize=20)
    ax3.set_title('Spectrum Coverage by Patient', fontsize=20)
    ax3.tick_params(labelsize=20)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax3, orientation='horizontal', pad=0.1)
    cbar.set_label('Coverage Percentage', fontsize=20)
    
    # Add text annotations
    for i, patient_id in enumerate(all_patients):
        coverage = patient_spectrum_coverage[patient_id]
        coverage_pct = 100 * coverage['selected'] / coverage['total'] if coverage['total'] > 0 else 0
        text = ax3.text(0, i, f'{coverage_pct:.1f}%', ha="center", va="center",
                       color="white" if coverage_pct < 50 else "black", fontsize=16)
    
    # Add divider between cancer and control
    if len(cancer_patients) > 0:
        ax3.axhline(y=len(cancer_patients) - 0.5, color='black', linewidth=2)
    
    # 4. Distribution of selection counts
    ax4 = axes[1, 1]
    
    # Get selection counts for all selected spectra
    selection_counts = [count for count in spectrum_selection_count.values() if count > 0]
    
    if selection_counts:
        ax4.hist(selection_counts, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        ax4.axvline(x=np.mean(selection_counts), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(selection_counts):.1f}')
        ax4.axvline(x=np.median(selection_counts), color='green', linestyle='--',
                   label=f'Median: {np.median(selection_counts):.0f}')
        ax4.set_xlabel('Number of Times Selected', fontsize=20)
        ax4.set_ylabel('Number of Spectra', fontsize=20)
        ax4.set_title('Distribution of Spectrum Selection Frequency\n(for selected spectra only)', fontsize=20)
        ax4.legend(fontsize=20)
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(labelsize=20)
    else:
        ax4.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax4.transAxes, fontsize=20)
        ax4.set_title('Distribution of Spectrum Selection Frequency', fontsize=20)
    
    plt.tight_layout()
    
    # Save figure if directory provided
    if save_dir:
        save_path = Path(save_dir) / f'{file_prefix}spectrum_coverage_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved spectrum coverage analysis to {save_path}")
    
    plt.show()
    
    # Create detailed report of never-selected spectra
    if never_selected_spectra and save_dir:
        never_selected_df = pd.DataFrame(never_selected_spectra)
        never_selected_df = never_selected_df.sort_values(['patient_type', 'patient_id', 'spectrum_id'])
        
        report_path = Path(save_dir) / f'{file_prefix}never_selected_spectra.csv'
        never_selected_df.to_csv(report_path, index=False)
        print(f"Saved never-selected spectra report to {report_path}")
        
        print("\nNever-Selected Spectra Summary:")
        print(never_selected_df.groupby('patient_type').size())
    
    # Return comprehensive statistics
    spectrum_coverage_stats = {
        'total_spectra_in_dataset': total_spectra_in_dataset,
        'total_spectra_selected': total_spectra_selected,
        'total_never_selected': total_never_selected,
        'coverage_percentage': coverage_percentage,
        'never_selected_spectra': never_selected_spectra,
        'patient_spectrum_coverage': dict(patient_spectrum_coverage),
        'spectrum_selection_count': dict(spectrum_selection_count),
        'cancer_never_selected': cancer_never_selected,
        'control_never_selected': control_never_selected
    }
    
    return spectrum_coverage_stats


def plot_spectra_per_iteration_colored_by_staging(
    filtered_df,
    dataset,
    raman_shift,
    max_iterations=None,
    save_dir=None
):
    """
    Plot spectra for each iteration (colored by staging) and one combined plot for all.

    Args:
        filtered_df: Filtered DataFrame of selected iterations
        dataset: OC_Dataset object used for training
        raman_shift: 1D array of Raman shift values
        max_iterations: Optional limit on number of iterations to plot
        save_dir: Directory to save plots (optional)

    Returns:
        None
    """
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    import seaborn as sns

    # Use Set1 palette red and blue
    palette = sns.color_palette("Set1", n_colors=3)
    color_cancer = palette[0]  # typically red
    color_control = palette[1]  # typically blue

    # Track all spectra for the final combined plot
    all_spectra = []

    iterations_to_plot = filtered_df['iteration'].tolist()
    if max_iterations:
        iterations_to_plot = iterations_to_plot[:max_iterations]

    print(f"Plotting {len(iterations_to_plot)} iterations...")

    # Iterate only over the iterations we want to plot
    for _, row in filtered_df[filtered_df['iteration'].isin(iterations_to_plot)].iterrows():
        iteration = row['iteration']
        selected_spectra = row['selected_spectra_index']

        fig, ax = plt.subplots(figsize=(14, 6))
        # Black background and margins
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')

        ax.set_title(f"Iteration {iteration} - Selected Spectra", fontsize=20, color='white')
        ax.set_xlabel("Raman Shift", color='white', fontsize=20)
        ax.set_ylabel("Intensity", color='white', fontsize=20)

        for patient_id, indices in selected_spectra.items():
            for idx in indices:
                intensity, _, metadata = dataset[idx]
                label = metadata['staging'].lower()
                color = color_cancer if ('cancer' in label or 'ca' in label) else color_control

                # Ensure numpy array for plotting
                try:
                    y = intensity.squeeze().numpy()
                except Exception:
                    y = np.asarray(intensity).squeeze()

                # Plot for this iteration
                ax.plot(raman_shift, y, color=color, alpha=0.4)

                # Save for combined plot
                all_spectra.append((y, color))

        # Custom legend (invisible plots used to create legend entries)
        p_cancer, = ax.plot([], [], color=color_cancer, label='Cancer', linewidth=3)
        p_control, = ax.plot([], [], color=color_control, label='Control', linewidth=3)
        legend = ax.legend(frameon=True, fontsize=20)
        # Style legend for black bg and white text
        legend.get_frame().set_facecolor('black')
        legend.get_frame().set_edgecolor('white')
        for text in legend.get_texts():
            text.set_color('white')

        # Style ticks, spines, grid
        ax.tick_params(colors='white', labelsize=20)  # ticks and tick labels to white
        plt.setp(ax.get_xticklabels(), color='white', fontsize=20)
        plt.setp(ax.get_yticklabels(), color='white', fontsize=20)
        for spine in ax.spines.values():
            spine.set_color('white')

        ax.grid(True, alpha=0.2, color='gray')

        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f"iteration_{iteration}_spectra.png")
            plt.savefig(path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
            print(f"Saved iteration plot to {path}")

        plt.show()

    # === Final Combined Plot ===
    print("\nPlotting combined spectra from all filtered iterations...")
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    for intensity, color in all_spectra:
        ax.plot(raman_shift, intensity, color=color, alpha=0.3)

    ax.set_title(f"All Spectra from {len(iterations_to_plot)} Filtered Iterations", fontsize=20, color='white')
    ax.set_xlabel("Raman Shift", color='white', fontsize=20)
    ax.set_ylabel("Intensity", color='white', fontsize=20)
    ax.grid(True, alpha=0.2, color='gray')

    p_cancer, = ax.plot([], [], color=color_cancer, label='Cancer', linewidth=3)
    p_control, = ax.plot([], [], color=color_control, label='Control', linewidth=3)
    legend = ax.legend(frameon=True, fontsize=20)
    legend.get_frame().set_facecolor('black')
    legend.get_frame().set_edgecolor('white')
    for text in legend.get_texts():
        text.set_color('white')

    ax.tick_params(colors='white', labelsize=20)
    plt.setp(ax.get_xticklabels(), color='white', rotation=45, fontsize=20)
    plt.setp(ax.get_yticklabels(), color='white', fontsize=20)
    for spine in ax.spines.values():
        spine.set_color('white')

    plt.tight_layout()

    if save_dir:
        final_path = os.path.join(save_dir, "all_filtered_iterations_combined_plot.png")
        plt.savefig(final_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
        print(f"Saved combined plot to {final_path}")

    plt.show()

def analyze_logreg_feature_selection(
    filtered_df,
    dataset,
    raman_shift,
    C=1.0,
    random_seed=42,
    save_dir=None
):
    """
    Train L1-regularized logistic regression on each filtered iteration and analyze feature selection.

    Args:
        filtered_df: Filtered results_df containing selected spectra
        dataset: The OC_Dataset used for training
        raman_shift: 1D numpy array of Raman shift values
        C: Inverse regularization strength for LogisticRegression
        random_seed: Random seed for reproducibility
        save_dir: Optional directory to save the plots

    Returns:
        selection_counts: How many times each feature was selected
        average_coefficients: Average coefficient value per feature
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.linear_model import LogisticRegression
    import os
    import matplotlib.patches as mpatches

    # palette: use Set1 red and blue
    palette = sns.color_palette("Set1", n_colors=3)
    color_cancer = palette[0]  # red-ish
    color_control = palette[1]  # blue-ish
    gray_color = 'gray'

    n_features = len(raman_shift)
    selection_counts = np.zeros(n_features)
    coefficient_sums = np.zeros(n_features)
    n_iterations = 0

    for _, row in filtered_df.iterrows():
        selected_indices = row['selected_spectra_index']
        all_indices = [idx for indices in selected_indices.values() for idx in indices]

        if len(all_indices) == 0:
            continue

        # Prepare data
        X, y, _, _ = prepare_data(dataset, all_indices)

        # Train L1 Logistic Regression
        model = LogisticRegression(
            penalty='l1',
            solver='liblinear',
            C=C,
            random_state=random_seed
        )
        model.fit(X, y)

        coefs = model.coef_.squeeze()

        # Ensure proper length
        if coefs.shape[0] != n_features:
            # If for any reason features mismatch, pad or trim to n_features
            tmp = np.zeros(n_features)
            tmp[:min(len(coefs), n_features)] = coefs[:min(len(coefs), n_features)]
            coefs = tmp

        selection_counts += (coefs != 0).astype(int)
        coefficient_sums += coefs
        n_iterations += 1

    if n_iterations == 0:
        average_coefficients = np.zeros(n_features)
    else:
        average_coefficients = coefficient_sums / n_iterations

    # === Plotting ===
    fig, axes = plt.subplots(2, 1, figsize=(18, 10), sharex=True)
    fig.patch.set_facecolor('black')  # Set background to black

    # Top: selection counts histogram (grey)
    axes[0].bar(raman_shift, selection_counts, width=1.0, color=gray_color, label='Selection Count')

    # Bottom: average coefficients colored by sign (red -> cancer, blue -> control, grey -> zero)
    coef_colors = []
    for val in average_coefficients:
        if val > 0:
            coef_colors.append(color_cancer)
        elif val < 0:
            coef_colors.append(color_control)
        else:
            coef_colors.append(gray_color)

    axes[1].bar(raman_shift, average_coefficients, width=1.0, color=coef_colors, label='Avg Coefficient')

    # Customize both subplots
    for ax, title, ylabel in zip(
        axes,
        ["Feature Selection Frequency (L1 Logistic Regression)", "Average Coefficient Value per Feature"],
        ["Selection Count", "Average Coefficient"]
    ):
        ax.set_facecolor('black')
        ax.set_title(title, fontsize=24, color='white')
        ax.set_ylabel(ylabel, fontsize=24, color='white')
        ax.tick_params(colors='white', labelsize=20)
        ax.grid(True, alpha=0.3, color='gray')
        for spine in ax.spines.values():
            spine.set_color('white')

    # Legends: create separate legends for each subplot
    # Top legend (selection count)
    top_leg = axes[0].legend(frameon=True, fontsize=24)
    top_leg.get_frame().set_facecolor('black')
    top_leg.get_frame().set_edgecolor('white')
    for text in top_leg.get_texts():
        text.set_color('white')

    # Bottom legend: red/blue/grey patches meaning
    cancer_patch = mpatches.Patch(color=color_cancer, label='Contributes toward Cancer (coef > 0)')
    control_patch = mpatches.Patch(color=color_control, label='Contributes toward Control (coef < 0)')
    # zero_patch = mpatches.Patch(color=gray_color, label='No contribution (coef = 0)')

    bottom_leg = axes[1].legend(handles=[cancer_patch, control_patch], frameon=True, fontsize=20)
    bottom_leg.get_frame().set_facecolor('black')
    bottom_leg.get_frame().set_edgecolor('white')
    for text in bottom_leg.get_texts():
        text.set_color('white')

    axes[1].set_xlabel("Raman Shift", fontsize=24, color='white')
    axes[1].axhline(0, color='white', linestyle='--', linewidth=1)

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "logreg_l1_feature_selection.png")
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"Saved feature selection analysis to {path}")

    plt.show()

    return selection_counts, average_coefficients


def main():
    """Main function to run the analysis."""
    
    # Configuration
    # Update these paths to match your setup
    results_file_path = r'/Users/yifeigu/Library/CloudStorage/Box-Box/Carney Lab Shared/Biweekly Meetings/Yifei/izabella/ev_data_balanced/both_bec_and_izabella_cleaned2_kfold_logistic_regression_selected_12_spectra.pkl'
    dataset_path = r'/Users/yifeigu/Library/CloudStorage/Box-Box/Carney Lab Shared/Biweekly Meetings/Yifei/izabella/ev_data_balanced/train_cleaned2'
    save_dir = r'/Users/yifeigu/Library/CloudStorage/Box-Box/Carney Lab Shared/Biweekly Meetings/Yifei/izabella/ev_data_balanced/analysis_results'
    
    # Create save directory if it doesn't exist
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Extract model type and spectrum count from filename for labeling
    file_name = Path(results_file_path).stem
    file_prefix = f'{file_name}_'
    
    print(f"Analyzing results from: {results_file_path}")
    
    # Load results
    results_df, detailed_predictions_df, raman_shift = load_results(results_file_path)
    
    # # Part 1: Plot histograms of accuracies
    # print("\n" + "="*50)
    # print("PART 1: Plotting Accuracy Histograms")
    # print("="*50)
    # accuracy_data = plot_accuracy_histograms(results_df, save_dir, file_prefix)
    
    # # Part 2: Filter iterations and create heatmap
    # print("\n" + "="*50)
    # print("PART 2: Filtering Iterations and Creating Heatmap")
    # print("="*50)
    
    # Define thresholds (adjust these based on your requirements)
    thresholds = {
        'train_spectrum_min': 0.95,  # Minimum average train accuracy at spectrum level
        'train_patient_min': 0.95,   # Minimum average train accuracy at patient level
        'val_spectrum_min': 0.75,   # Minimum average validation accuracy at spectrum level
        'val_patient_min': 0.80     # Minimum average validation accuracy at patient level
    }
    
    # Filter iterations based on thresholds
    filtered_df, selected_iterations = filter_iterations_by_threshold(results_df, thresholds)
    
    if len(filtered_df) > 0:
        # Create heatmap
        heatmap_data, patient_names, selection_counts = create_spectrum_frequency_heatmap(
            filtered_df, detailed_predictions_df, dataset_path, save_dir, file_prefix
        )
    else:
        print("\nNo iterations met the specified thresholds. Try lowering the thresholds.")
    
    print("\n" + "="*50)
    print("Analysis Complete!")
    print("="*50)

    # spectrum_stats = analyze_spectrum_coverage(results_df, dataset_path, save_dir, file_prefix)

    # # Check if any spectra were never selected
    # if spectrum_stats['total_never_selected'] > 0:
    #     print("Warning: Some spectra were never selected!")
    #     never_selected_list = spectrum_stats['never_selected_spectra']

    preprocessor = SpectrumPreprocessor(
        cropping=True,
        baseline_correction=True,
        remove_cosmic_rays=False,
        normalization=True,
        smoothing=True)
    dataset = OC_Dataset(dataset_path, preprocessor)

    # plot_spectra_per_iteration_colored_by_staging(
    # filtered_df=filtered_df,
    # dataset=dataset,
    # raman_shift=raman_shift,
    # max_iterations=None,
    # save_dir=save_dir  # or None
    # )

    selection_counts, average_coefficients = analyze_logreg_feature_selection(
    filtered_df=filtered_df,
    dataset=dataset,
    raman_shift=raman_shift,
    C=15.0,
    random_seed=42,
    save_dir=save_dir  # or None
    )

if __name__ == "__main__":
    main()