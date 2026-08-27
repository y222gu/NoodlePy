from torch.utils.data import Dataset
from utils.spectrum import Spectrum
from utils.spectrumpreprocessor import SpectrumPreprocessor
from combat.pycombat import pycombat
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import copy
from sklearn.manifold import TSNE
import yaml
import matplotlib.pyplot as plt
import traceback
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import mahalanobis
from scipy import linalg
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

class SpectrumPreprocessorWrapper:
    """Wrapper to maintain compatibility with the UI while using actual SpectrumPreprocessor"""
    def __init__(self, 
                 cropping=True,
                 baseline_correction=False,
                 remove_cosmic_rays=True,
                 normalization=False,
                 smoothing=False,
                 # Cropping parameters
                 start_raman_shift_cm=662.697,
                 end_raman_shift_cm=1784.104,
                 # Baseline correction parameters (airPLS)
                 baseline_lam=1000,
                 baseline_diff_order=1,
                 baseline_max_iter=15,
                 baseline_tol=0.005,
                 # Cosmic ray removal parameters (modified z-scores)
                 cosmic_threshold=10,
                 # Smoothing parameters (Savitzky-Golay)
                 smooth_window_length=9,
                 smooth_polyorder=2,
                 # Normalization type
                 normalization_type='by_max',
                 # Config path for actual preprocessor
                 config_path=None):
        
        # Store UI parameters
        self.cropping = cropping
        self.baseline_correction = baseline_correction
        self.remove_cosmic_rays = remove_cosmic_rays
        self.normalization = normalization
        self.smoothing = smoothing
        
        self.start_raman_shift_cm = start_raman_shift_cm
        self.end_raman_shift_cm = end_raman_shift_cm
        self.baseline_lam = baseline_lam
        self.baseline_diff_order = baseline_diff_order
        self.baseline_max_iter = baseline_max_iter
        self.baseline_tol = baseline_tol
        self.cosmic_threshold = cosmic_threshold
        self.smooth_window_length = smooth_window_length
        self.smooth_polyorder = smooth_polyorder
        self.normalization_type = normalization_type
        
        # Create config file for the actual preprocessor
        self._create_temp_config()
        
        # Initialize the actual SpectrumPreprocessor
        self.actual_preprocessor = SpectrumPreprocessor(
            cropping=cropping,
            baseline_correction=baseline_correction,
            remove_cosmic_rays=remove_cosmic_rays,
            normalization=normalization,
            smoothing=smoothing,
            config_path=self.temp_config_path
        )

    def _create_temp_config(self):
        """Create a temporary config file for the actual preprocessor"""
        config = {
            'preprocessing': {
                'cropping': {
                    'start_raman_shift_cm': self.start_raman_shift_cm,
                    'end_raman_shift_cm': self.end_raman_shift_cm
                },
                'baseline_correction': {
                    'lam': self.baseline_lam,
                    'diff_order': self.baseline_diff_order,
                    'max_iter': self.baseline_max_iter,
                    'tol': self.baseline_tol
                },
                'cosmic_rays_removal': {
                    'threshold': self.cosmic_threshold
                },
                'smoothing': {
                    'window_length': self.smooth_window_length,
                    'polyorder': self.smooth_polyorder
                },
                'normalization_type': self.normalization_type
            }
        }
        
        # Save to temp file
        import tempfile


        self.temp_config_path = tempfile.mktemp(suffix='.yml')
        with open(self.temp_config_path, 'w') as f:
            yaml.dump(config, f)

    def preprocess(self, spectrum: Spectrum) -> Spectrum:
        """Preprocess using the actual SpectrumPreprocessor with robust cropping"""
        try:
            # Make a deep copy to avoid modifying the original
            preprocessed = copy.deepcopy(spectrum)
            
            # Manual cropping with approximate matching
            if self.cropping:
                mask = (preprocessed.raman_shift_cm >= self.start_raman_shift_cm) & \
                       (preprocessed.raman_shift_cm <= self.end_raman_shift_cm)
                
                if not np.any(mask):
                    print(f"Warning: Cropping range [{self.start_raman_shift_cm}, {self.end_raman_shift_cm}] "
                          f"outside spectrum range [{preprocessed.raman_shift_cm.min()}, {preprocessed.raman_shift_cm.max()}]")
                    # Skip cropping if range is invalid
                else:
                    preprocessed.raman_shift_cm = preprocessed.raman_shift_cm[mask]
                    preprocessed.intensity = preprocessed.intensity[mask]
            
            # Apply other preprocessing steps
            if self.remove_cosmic_rays:
                preprocessed.remove_cosmic_rays(threshold=self.cosmic_threshold)
            
            if self.baseline_correction:
                preprocessed.airPLS(
                    lam=self.baseline_lam,
                    diff_order=self.baseline_diff_order,
                    max_iter=self.baseline_max_iter,
                    tol=self.baseline_tol
                )
            
            if self.normalization:
                preprocessed.normalize_spectrum(self.normalization_type)
            
            if self.smoothing:
                preprocessed.savgol_filter(
                    window_length=self.smooth_window_length,
                    polyorder=self.smooth_polyorder
                )
            
            return preprocessed
            
        except Exception as e:
            print(f"Error preprocessing spectrum: {e}")
            traceback.print_exc()
            # Return a copy of original on error
            return copy.deepcopy(spectrum)
    
    def to_dict(self):
        """Export preprocessor parameters to dictionary matching YAML structure"""
        return {
            'preprocessing': {
                'cropping': {
                    'start_raman_shift_cm': self.start_raman_shift_cm,
                    'end_raman_shift_cm': self.end_raman_shift_cm
                },
                'baseline_correction': {
                    'lam': self.baseline_lam,
                    'diff_order': self.baseline_diff_order,
                    'max_iter': self.baseline_max_iter,
                    'tol': self.baseline_tol
                },
                'cosmic_rays_removal': {
                    'threshold': self.cosmic_threshold
                },
                'smoothing': {
                    'window_length': self.smooth_window_length,
                    'polyorder': self.smooth_polyorder
                },
                'normalization_type': self.normalization_type
            },
            'enabled': {
                'cropping': self.cropping,
                'baseline_correction': self.baseline_correction,
                'remove_cosmic_rays': self.remove_cosmic_rays,
                'normalization': self.normalization,
                'smoothing': self.smoothing
            }
        }
    
    @classmethod
    def from_dict(cls, config):
        """Create preprocessor from dictionary (YAML structure)"""
        preproc = config.get('preprocessing', {})
        enabled = config.get('enabled', {})
        
        cropping_params = preproc.get('cropping', {})
        baseline_params = preproc.get('baseline_correction', {})
        cosmic_params = preproc.get('cosmic_rays_removal', {})
        smoothing_params = preproc.get('smoothing', {})
        norm_type = preproc.get('normalization_type', 'by_max')
        
        return cls(
            cropping=enabled.get('cropping', True),
            baseline_correction=enabled.get('baseline_correction', False),
            remove_cosmic_rays=enabled.get('remove_cosmic_rays', True),
            normalization=enabled.get('normalization', False),
            smoothing=enabled.get('smoothing', False),
            start_raman_shift_cm=cropping_params.get('start_raman_shift_cm', 662.697),
            end_raman_shift_cm=cropping_params.get('end_raman_shift_cm', 1784.104),
            baseline_lam=baseline_params.get('lam', 1000),
            baseline_diff_order=baseline_params.get('diff_order', 1),
            baseline_max_iter=baseline_params.get('max_iter', 15),
            baseline_tol=baseline_params.get('tol', 0.005),
            cosmic_threshold=cosmic_params.get('threshold', 10),
            smooth_window_length=smoothing_params.get('window_length', 9),
            smooth_polyorder=smoothing_params.get('polyorder', 2),
            normalization_type=norm_type
        )

class RamanRobotDataset(Dataset):
    def __init__(self, data_folder=None,
                 preprocessor=None,
                 augmentor=None):
        """
        Load the database of spectra and attach rich metadata from
        filename + metadata files in:

            data_folder/
                metadata/*.txt ( *_metadata.txt )
                spectra/*.txt
        """
        print("Loading the Raman dataset")
        self.preprocessor = preprocessor
        self.augmentor = augmentor

        if data_folder is None:
            raise ValueError("data_folder must be provided")

        self.data_folder = data_folder
        metadata_dir = os.path.join(data_folder, "metadata")
        spectra_dir = os.path.join(data_folder, "spectra")

        if not os.path.isdir(spectra_dir):
            print(f"Spectra folder not found: {spectra_dir}")
            self.db = []
            return
        
        # Build a table of file-level metadata (one row per spectra file)
        meta_df = self._build_metadata_table(spectra_dir, metadata_dir)

        list_of_spectrum_objects = []
        # loop through each file in spectra_dir build metadata for it
        for spectrum_file in os.listdir(spectra_dir):
            if not spectrum_file.endswith(".txt"):
                print(f"Skipping non-txt file in spectra folder: {spectrum_file}")
                continue

            spectrum_path = os.path.join(spectra_dir, spectrum_file)
            # find the corresponding row in meta_df
            row = meta_df[meta_df["spectrum_file"] == spectrum_file]
            meta = row.iloc[0].to_dict()

            spectrum_objects = self._load_files_to_spectrum_objects(
                data_folder=self.data_folder,
                filename=spectrum_path,
                base_metadata=meta,
            )
            list_of_spectrum_objects += spectrum_objects

        self.db = list_of_spectrum_objects
        print(f"Loaded {len(self.db)} spectra")

    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx: int):
        """
        Return one preprocessed spectrum (you can extend to return two augmentations if needed).
        """
        preprocessor = self.preprocessor
        chosen_spectrum: Spectrum = copy.deepcopy(self.db[idx])
        # cropped_spectrum = chosen_spectrum.crop_spectrum(625.277, 1786.791)
        preprocessed_spectrum = preprocessor.preprocess(chosen_spectrum)
        return preprocessed_spectrum

    # ----------------- METADATA HELPERS (from surface-plot script) -----------------

    @staticmethod
    def _parse_filename(filename: str) -> dict:
        """
        Parse metadata encoded in the filename.
        
        Generalized pattern that extracts key-value pairs from filenames like:
        "patient_538_staging_cancer_sample_0_point_17_rep_20_line_1_ring_9_x72.93_y-130.50.txt"
        
        Extracts: patient, staging, sample, point, rep, line, ring, x, y
        """
        name = os.path.basename(filename).replace('.txt', '')
        
        # Use regex to extract all key-value pairs
        # Match a word followed by underscore and a value (stops at next underscore+word pattern or end)

        split = name.split('_')

        # # remove the last two
        # split = split[:-2]

        # remove the ones with x, y, or z coordinates
        split = [s for s in split if not re.match(r'^[xyz][-+]?\d', s)]

        # every two elements are a key-value pair
        # Guard against odd-length splits (non-standard filenames)
        matches = []
        for i in range(0, len(split) - 1, 2):
            matches.append((split[i], split[i + 1]))

        d = {}
        for key, value in matches:
            key_lower = key.lower()
            
            # Try to convert to appropriate type
            try:
                # Check if it's a float (contains decimal point or negative)
                if '.' in value or (value.startswith('-') and value[1:].replace('.', '').isdigit()):
                    d[key_lower] = float(value)
                # Check if it's an integer
                elif value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                    d[key_lower] = int(value)
                else:
                    # Keep as string
                    d[key_lower] = value
            except ValueError:
                d[key_lower] = value

        # parse x, y, z coordinates separately
        xy_match = re.search(r'x([-+]?\d*\.?\d+)_y([-+]?\d*\.?\d+)', name)
        if xy_match:
            d['x'] = float(xy_match.group(1))
            d['y'] = float(xy_match.group(2))

        z_match = re.search(r'_z([-+]?\d*\.?\d+)(?:_|$)', name)
        if z_match:
            d['z'] = float(z_match.group(1))

        return d

    @staticmethod
    def _normalize_key(key: str) -> str:
        """
        Normalize metadata keys to lowercase with underscores.
        
        Examples:
          'Prusa_position' -> 'prusa_position'
          'Integration time (ms)' -> 'integration_time_ms'
        """
        key = key.strip().lower()
        key = key.replace("(", "_").replace(")", "")
        key = key.replace(" ", "_")
        key = re.sub(r'_+', '_', key)  # Replace multiple underscores with single
        return key.strip('_')  # Remove leading/trailing underscores

    @staticmethod
    def _parse_metadata_file(path: str) -> dict:
        """
        Parse the contents of a metadata file like:
          Number_of_repetitions = 3
          Integration_time = 5000
          Laser_power = 450
          Prusa_position = 13.02
          Nanodrive_position = 64.0
          Timestamp = 2025-12-06 20:07:19
        or using ':' instead of '='.
        """
        meta = {}
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Handle both  "key: value"  and  "key = value"
                if ":" in line:
                    key, value = line.split(":", 1)
                elif "=" in line:
                    key, value = line.split("=", 1)
                else:
                    continue

                key = RamanRobotDataset._normalize_key(key)
                value = value.strip().strip(",")

                # Try to convert to int or float
                try:
                    if any(c in value for c in (".", "e", "E")):
                        num = float(value)
                        if num.is_integer():
                            num = int(num)
                        meta[key] = num
                    else:
                        meta[key] = int(value)
                except ValueError:
                    meta[key] = value  # keep as string if not numeric

        return meta

    @staticmethod
    def _build_metadata_table(spectrum_dir: str, metadata_dir: str) -> pd.DataFrame:
        """
        Build a DataFrame with one row per spectra file, containing:

        - spectrum_rel_path, metadata_rel_path (if available)
        - filename-derived metadata from spectra file (patient, staging, sample, point, line, ring, rep, x, y)
        - file metadata contents (number_of_repetitions, integration_time, laser_power,
            prusa_position, nanodrive_position, timestamp, ...) if metadata file exists
        - focus_position = prusa_position - 0.001 * nanodrive_position (if applicable)
        - relative timestamp in minutes from earliest timestamp (if metadata files exist)
        - distance_from_center_mm (if x and y available)
        """
        rows = []
        n_total = 0
        n_with_metadata = 0
        n_without_metadata = 0

        rows = []
        for spectrum_file in os.listdir(spectrum_dir):
            if not spectrum_file.endswith(".txt"):
                continue

            base_fname = os.path.basename(spectrum_file)
            n_total += 1

            # Parse filename from spectra file
            fn_meta = RamanRobotDataset._parse_filename(base_fname)
            # add filename to metadata row
            fn_meta["spectrum_file"] = spectrum_file
            row = fn_meta.copy()

            # If metadata file exists, add its contents
            metadata_fname = base_fname.replace(".txt", "_metadata.txt")
            metadata_path = os.path.join(metadata_dir, metadata_fname)

            if os.path.isfile(metadata_path):
                file_meta = RamanRobotDataset._parse_metadata_file(metadata_path)
                row.update(file_meta)
                
                # Add focus position
                RamanRobotDataset._add_focus_position(row)
                RamanRobotDataset._add_relative_timestamps(row)
                RamanRobotDataset._add_distance_from_center(row)

                n_with_metadata += 1
            else:
                n_without_metadata += 1

            rows.append(row)

        meta_df = pd.DataFrame(rows)
        # if "patient" in meta_df.columns:
        #     od433_rows = meta_df[meta_df["patient"] == "OD433"]
        #     if not od433_rows.empty:
        #         print("Rows with patient OD433:\n", od433_rows)

        return meta_df

    @staticmethod
    def _add_distance_from_center(row: dict) -> None:
        """
        Compute and add distance_from_center_mm to a metadata row.
        distance_from_center_mm = min(abs(x), abs(y)) * 0.1
        """
        x = row.get("x")
        y = row.get("y")
        if x is not None and y is not None: # x^2 + y^2
            row["distance_from_center_mm"] = np.sqrt(x**2 + y**2) * 0.001

    @staticmethod
    def _add_focus_position(row: dict) -> None:
        """
        Compute and add focus_position to a metadata row.
        focus_position = prusa_position - 0.001 * nanodrive_position
        """
        prusa = (
            row.get("prusa_position_mm")
            or row.get("prusa_position")
        )
        nano = (
            row.get("nanodrive_position_um")
            or row.get("nanodrive_position")
        )
        if prusa is not None and nano is not None:
            row["focus_position"] = prusa - 0.001 * nano

    @staticmethod
    def _add_relative_timestamps(row: dict) -> None:
        """
        Add a relative timestamp in minutes from the earliest timestamp.
        """
        if "timestamp" not in row:
            return

        ts = pd.to_datetime(row["timestamp"], errors="coerce")
        if ts is pd.NaT:
            raise ValueError("Timestamp could not be parsed.")

        # Store the baseline timestamp for relative calculation
        baseline = ts

        # Calculate relative timestamp in minutes
        row["relative_timestamp"] = (ts - baseline).total_seconds() / 60.0

    @staticmethod
    def _load_files_to_spectrum_objects(data_folder: str,
                                        filename: str,
                                        base_metadata: dict):
        """
        Load a spectra .txt file (relative to data_folder) into one or more Spectrum objects,
        splitting on repeated wavelength blocks, and attach a copy of base_metadata plus 'spectrum_id'.
        """
        full_path = os.path.join(data_folder, filename)

        # if the file is empty, skip the file
        if os.stat(full_path).st_size == 0:
            return []

        with open(full_path) as f:
            data = pd.read_csv(f, sep=",", header=None)

        repeated_wavelengths = data.iloc[:, 0].value_counts()
        first_repeated_wavelength = repeated_wavelengths.idxmax()
        start_indexes = data[data.iloc[:, 0] == first_repeated_wavelength].index.tolist()

        spectrum_objects = []
        wavelength_nm = None
        intensities = []

        # split the repeated measurements and collect intensities
        for i in range(len(start_indexes)):
            if i == len(start_indexes) - 1:
                wl = data.iloc[start_indexes[i]:, 0].values.round(3)
                intens = data.iloc[start_indexes[i]:, 1].values.round(3)
            else:
                wl = data.iloc[start_indexes[i]:start_indexes[i + 1], 0].values.round(3)
                intens = data.iloc[start_indexes[i]:start_indexes[i + 1], 1].values.round(3)

            if wavelength_nm is None:
                wavelength_nm = wl
            intensities.append(intens)

        if not intensities:
            return []

        median_intensity = np.median(np.vstack(intensities), axis=0).round(3)
        metadata = copy.deepcopy(base_metadata)

        spectrum_objects.append(
            Spectrum(
            wavelength_nm=wavelength_nm,
            intensity=median_intensity,
            metadata=metadata,
            )
        )
        return spectrum_objects

    # ----------------- COMBAT BATCH CORRECTION (unchanged) -----------------

    def combat_batch_correction(self):
        """
        Applies ComBat batch correction to the intensity values of all spectra in the dataset.
        This function modifies the dataset's intensities while keeping Raman shift and metadata intact.
        """
        if not self.db:
            print("Dataset is empty. No batch correction applied.")
            return

        # Extract intensity values and batch labels (using 'date' as batch label, if present).
        intensity_matrix = np.array([spectrum.intensity for spectrum in self.db])

        # You may want to change this to something like 'patient' or 'staging' depending on your use case.
        batch_labels = pd.Series([spectrum.metadata.get("date", "unknown") for spectrum in self.db])

        # Convert batch labels to categorical numeric labels.
        batch_categories = pd.factorize(batch_labels)[0]

        # Transpose data so that features are rows.
        data_transposed = pd.DataFrame(intensity_matrix.T)

        print("Data transposed shape:", data_transposed.shape)
        print("Batch categories shape:", len(batch_categories))

        # Apply ComBat for batch effect correction.
        corrected_data_transposed = pycombat(data_transposed, batch_categories)

        # Convert back to NumPy array and transpose to original shape.
        corrected_intensity_matrix = corrected_data_transposed.to_numpy().T

        # Replace original intensity values while keeping metadata and Raman shift intact.
        for i, spectrum in enumerate(self.db):
            self.db[i].intensity = corrected_intensity_matrix[i]

        print("Batch effect correction using ComBat has been applied successfully.")
        return self.db

    def plot_tsne(self, metadata_key: str, perplexity: int = 30, n_iter: int = 1000, 
                  random_state: int = 42, figsize: tuple = (10, 8)):
        """
        Plot t-SNE visualization of all spectra intensities, colored by a metadata field.
        
        Args:
            metadata_key: The metadata field to use for coloring (e.g., 'patient', 'staging')
            perplexity: The perplexity parameter for t-SNE (default: 30)
            n_iter: Number of iterations for t-SNE optimization (default: 1000)
            random_state: Random seed for reproducibility (default: 42)
            figsize: Figure size as (width, height) tuple (default: (10, 8))
        """
        import matplotlib.pyplot as plt
        
        if not self.db:
            print("Dataset is empty. Cannot plot t-SNE.")
            return
        
        # Extract intensity values
        intensity_matrix = np.array([spectrum.intensity for spectrum in self.db])
        
        # Extract metadata labels for coloring
        labels = [spectrum.metadata.get(metadata_key, "unknown") for spectrum in self.db]
        
        # Run t-SNE
        print(f"Running t-SNE with perplexity={perplexity}, n_iter={n_iter}...")
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, 
                    random_state=random_state)
        tsne_coords = tsne.fit_transform(intensity_matrix)
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get unique labels and assign colors
        unique_labels = list(set(labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = [l == label for l in labels]
            ax.scatter(tsne_coords[mask, 0], tsne_coords[mask, 1], 
                      label=str(label), alpha=0.6, s=50, color=colors[i])
        
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
        ax.set_title(f't-SNE Visualization colored by {metadata_key}')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

        savefig = os.path.join(self.data_folder, f'tsne_{metadata_key}.png')
        fig.savefig(savefig, dpi=300)
        print(f"t-SNE plot saved to {savefig}")
        
        return fig, tsne_coords
    
if __name__ == "__main__":
    # Example usage
    data_folder = r'/Users/yifeigu/Documents/Carney_Lab/Data/RamanRobot/2025_12_10'

        # Create default config
    default_config = {
        'preprocessing': {
            'cropping': {
                'start_raman_shift_cm': 662.697,
                'end_raman_shift_cm': 1784.104
            },
            'baseline_correction': {
                'lam': 100,
                'diff_order': 1,
                'max_iter': 15,
                'tol': 0.005
            },
            'cosmic_rays_removal': {
                'threshold': 10
            },
            'smoothing': {
                'window_length': 5,
                'polyorder': 3
            },
            'normalization_type': 'by_max'
        },
        'enabled': {
            'cropping': True,
            'baseline_correction': True,
            'remove_cosmic_rays': False,
            'normalization': True,
            'smoothing': True
        }
    }

    preprocessor = SpectrumPreprocessorWrapper.from_dict(default_config)
    dataset = RamanRobotDataset(data_folder=data_folder, preprocessor=preprocessor)

    # def plot_tsne_by_metadata(dataset, data_folder, metadata_key='ring', 
    #                           perplexity=30, n_iter=1000, random_state=42, 
    #                           figsize_2d=(10, 8), figsize_3d=(12, 10), 
    #                           point_size=200, point_alpha=1, selected_rings=None):
    #     """
    #     Plot t-SNE visualization of all spectra colored by a specified metadata field in 2D and 3D.
        
    #     Args:
    #         dataset: RamanRobotDataset instance
    #         data_folder: Path to save the plots
    #         metadata_key: Metadata field to use for coloring (e.g., 'ring', 'staging', 'patient')
    #         perplexity: t-SNE perplexity parameter (default: 30)
    #         n_iter: Number of t-SNE iterations (default: 1000)
    #         random_state: Random seed for reproducibility (default: 42)
    #         figsize_2d: Figure size for 2D plot (default: (10, 8))
    #         figsize_3d: Figure size for 3D plot (default: (12, 10))
    #         point_size: Size of scatter points (default: 100)
    #         point_alpha: Transparency of points (default: 0.6)
    #         selected_rings: List of rings to filter (default: None, which means no filtering)
        
    #     Returns:
    #         Tuple of (fig_2d, fig_3d, tsne_coords_2d, tsne_coords_3d)
    #     """
    #     if not dataset.db:
    #         print("Dataset is empty. Cannot plot t-SNE.")
    #         return None, None, None, None
        
    #     # Extract intensity values and metadata
    #     intensity_matrix = np.array([dataset.preprocessor.preprocess(s).intensity 
    #                                     for s in dataset.db])
    #     metadata_values = np.array([s.metadata.get(metadata_key, 'unknown') for s in dataset.db])
    #     ring_values = np.array([s.metadata.get('ring', None) for s in dataset.db])
        
    #     # Filter by selected rings if provided
    #     if selected_rings is not None:
    #         mask = np.isin(ring_values, selected_rings)
    #         intensity_matrix = intensity_matrix[mask]
    #         metadata_values = metadata_values[mask]
        
    #     # Check if we have enough samples
    #     n_samples = len(intensity_matrix)
    #     if n_samples < 2:
    #         print(f"Not enough samples ({n_samples}) after filtering. Cannot perform t-SNE.")
    #         return None, None, None, None
        
    #     # Adjust perplexity to be valid for the number of samples
    #     adjusted_perplexity = min(perplexity, n_samples - 1)
    #     if adjusted_perplexity != perplexity:
    #         print(f"Warning: Adjusted perplexity from {perplexity} to {adjusted_perplexity} due to small sample size ({n_samples})")
        
    #     # Get unique values and assign colors
    #     unique_values = sorted(list(set(metadata_values)))
    #     n_values = len(unique_values)
        
    #     # Use specific colors for staging, otherwise use turbo colormap
    #     if metadata_key == 'staging':
    #         staging_colors = {
    #             'control': '#437ED5',
    #             'healthy': "#437ED5",
    #             'cancer': "#D2504D"
    #         }
    #         value_to_color = {val: staging_colors.get(str(val).lower(), '#808080') 
    #                          for val in unique_values}
    #     else:
    #         colors = plt.cm.turbo(np.linspace(0, 1, n_values))
    #         value_to_color = {val: colors[i] for i, val in enumerate(unique_values)}
        
    #     fig_2d, tsne_coords_2d = None, None
    #     fig_3d, tsne_coords_3d = None, None
        
    #     # 2D t-SNE
    #     print(f"Running 2D t-SNE with perplexity={adjusted_perplexity}, n_iter={n_iter}...")
    #     tsne_2d = TSNE(n_components=2, perplexity=adjusted_perplexity, n_iter=n_iter, 
    #                     random_state=random_state)
    #     tsne_coords_2d = tsne_2d.fit_transform(intensity_matrix)
        
    #     fig_2d, ax = plt.subplots(figsize=figsize_2d)
        
    #     for val in unique_values:
    #         mask = metadata_values == val
    #         ax.scatter(tsne_coords_2d[mask, 0], tsne_coords_2d[mask, 1],
    #                   label=f'{metadata_key}: {val}', alpha=point_alpha, s=point_size,
    #                   color=value_to_color[val])
        
    #     ax.set_xlabel('t-SNE Component 1', fontsize=24, fontweight='bold')
    #     ax.set_ylabel('t-SNE Component 2', fontsize=24, fontweight='bold')
    #     ax.set_title(f'Probing distance from the edge',
    #                 fontsize=24, pad=20)
    #     ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    #     ax.grid(True, alpha=0.3)
    #     plt.tight_layout()
        
    #     savepath_2d = os.path.join(data_folder, f'tsne_2d_by_{metadata_key}_3_9.svg')
    #     fig_2d.savefig(savepath_2d, format='svg', bbox_inches='tight')
    #     print(f"2D t-SNE plot saved to {savepath_2d}")
        
    #     # 3D t-SNE
    #     print(f"Running 3D t-SNE with perplexity={adjusted_perplexity}, n_iter={n_iter}...")
    #     tsne_3d = TSNE(n_components=3, perplexity=adjusted_perplexity, n_iter=n_iter,
    #                     random_state=random_state)
    #     tsne_coords_3d = tsne_3d.fit_transform(intensity_matrix)
        
    #     fig_3d = plt.figure(figsize=figsize_3d)
    #     ax = fig_3d.add_subplot(111, projection='3d')

    #     # Map metadata values to numeric indices for colorbar
    #     value_to_idx = {val: i for i, val in enumerate(unique_values)}
    #     idx_array = np.array([value_to_idx[val] for val in metadata_values])
    #     scatter = ax.scatter(tsne_coords_3d[:, 0], tsne_coords_3d[:, 1], 
    #                          tsne_coords_3d[:, 2], c=idx_array,
    #                          cmap=plt.cm.turbo, alpha=point_alpha, s=point_size)

    #     # Colorbar with a few labeled ticks
    #     cbar = fig_3d.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    #     if n_values > 1:
    #         tick_positions = np.linspace(0, n_values - 1, min(n_values, 5)).astype(int)
    #         cbar.set_ticks(tick_positions)
    #         cbar.set_ticklabels([f"{np.round(unique_values[i]*0.1, 1)} mm" for i in tick_positions], fontsize=24)
    #     else:
    #         cbar.set_ticks([0])
    #         cbar.set_ticklabels([f"{np.round(unique_values[0]*0.1, 1)} mm"])
    #     cbar.set_label(metadata_key, fontsize=16, fontweight='bold')

    #     # Tighten view limits to reduce empty space
    #     ax.set_xlim(-8.5, 8.5)
    #     ax.set_ylim(-4, 4)
    #     ax.set_zlim(-5, 5)

    #     ax.set_xlabel('t-SNE Component 1', fontsize=24, fontweight='bold', labelpad=10)
    #     ax.set_ylabel('t-SNE Component 2', fontsize=24, fontweight='bold', labelpad=10)
    #     ax.set_zlabel('t-SNE Component 3', fontsize=24, fontweight='bold', labelpad=10)
    #     ax.set_title(f'3D t-SNE Raman spectra colored by cancer status',
    #             fontsize=24, pad=20)
    #     ax.view_init(elev=20, azim=45)
    #     plt.tight_layout()
        
    #     savepath_3d = os.path.join(data_folder, f'tsne_3d_by_{metadata_key}_3_9.svg')
    #     fig_3d.savefig(savepath_3d, format='svg', bbox_inches='tight')
    #     print(f"3D t-SNE plot saved to {savepath_3d}")
        
    #     plt.figure(fig_2d.number)
    #     plt.show()
    #     plt.figure(fig_3d.number)
    #     plt.show()
        
    #     return fig_2d, fig_3d, tsne_coords_2d, tsne_coords_3d

    # selected_rings = [3, 4, 5, 6, 7, 8, 9] # [1, 2]  # Define which rings to filter
    # plot_tsne_by_metadata(dataset, data_folder, metadata_key='ring') #, selected_rings=selected_rings
    
    # plot_tsne_by_metadata(dataset, data_folder, metadata_key='staging', selected_rings=selected_rings)


    # def plot_spectrum(spectrum, title="Spectrum", figsize=(10, 4)):
    #     """Plot a single spectrum with wavelength vs intensity."""
    #     import matplotlib.pyplot as plt
        
    #     fig, ax = plt.subplots(figsize=figsize)
    #     ax.plot(spectrum.raman_shift_cm, spectrum.intensity, linewidth=2)
    #     ax.set_xlabel('Raman Shift (cm⁻¹)', fontsize=12, fontweight='bold')
    #     ax.set_ylabel('Intensity', fontsize=12, fontweight='bold')
    #     ax.set_title(title, fontsize=13, fontweight='bold')
    #     ax.tick_params(axis='both', labelsize=10, width=1.5)
    #     ax.grid(True, alpha=0.3)
    #     for spine in ax.spines.values():
    #         spine.set_linewidth(1.5)
    #     plt.tight_layout()
    #     return fig

    # def add_colored_bands(ax, bands, *, y_top_frac=0.97, label_rotation=90, label_kwargs=None):
    #     """
    #     Add colored vertical bands and labels to an axis.

    #     bands: list of dicts, each like:
    #         {"xmin": 400, "xmax": 600, "label": "Cholesterol\nGlycogen", "color": "#F4D06F", "alpha": 0.35}

    #     y_top_frac: label y-position as a fraction of the y-axis (0..1). 0.97 puts text near the top.
    #     """
    #     if label_kwargs is None:
    #         label_kwargs = {}

    #     # Use axis-fraction coordinates for y so labels stay at the top even if y-limits change
    #     trans = ax.get_xaxis_transform()  # x in data coords, y in axes fraction (0..1)

    #     for b in bands:
    #         xmin = b["xmin"]
    #         xmax = b["xmax"]
    #         color = b.get("color", "lightgray")
    #         alpha = b.get("alpha", 0.25)
    #         label = b.get("label", "")

    #         # Band
    #         ax.axvspan(xmin, xmax, color=color, alpha=alpha, zorder=0)

    #         # Label (centered in the band)
    #         if label:
    #             xmid = (xmin + xmax) / 2
    #             ax.text(
    #                 xmid, y_top_frac, label,
    #                 transform=trans,
    #                 ha="center", va="top",
    #                 rotation=label_rotation,
    #                 fontsize=20, fontweight="bold",
    #                 color="black",
    #                 **label_kwargs
    #             )


    # # Collect first few cancer and control spectra
    # cancer_spectra = []
    # control_spectra = []
    # num_per_class = 3
    
    # for spectrum in dataset.db:
    #     staging = spectrum.metadata.get('staging', '')
    #     if staging == 'cancer' and len(cancer_spectra) < num_per_class:
    #         cancer_spectra.append(spectrum)
    #     elif staging == 'control' and len(control_spectra) < num_per_class:
    #         control_spectra.append(spectrum)
        
    #     if len(cancer_spectra) >= num_per_class and len(control_spectra) >= num_per_class:
    #         break
    
    # # Plot cancer and control spectra in 3D
    # if cancer_spectra and control_spectra:
        
    #     fig = plt.figure(figsize=(14, 6))
    #     ax = fig.add_subplot(111, projection='3d')
        
    #     # Plot cancer spectra
    #     for i, spectrum in enumerate(cancer_spectra):
    #         processed = dataset.preprocessor.preprocess(spectrum)
    #         x = processed.raman_shift_cm
    #         y = np.full_like(x, i)  # Stack in y-direction
    #         z = processed.intensity
    #         ax.plot(x, y, z, linewidth=2, color='#8C2928', label=f'Cancer {i+1}' if i == 0 else '')
        
    #     # Plot control spectra (offset in y)
    #     y_offset = num_per_class
    #     for i, spectrum in enumerate(control_spectra):
    #         processed = dataset.preprocessor.preprocess(spectrum)
    #         x = processed.raman_shift_cm
    #         y = np.full_like(x, i + y_offset)
    #         z = processed.intensity
    #         ax.plot(x, y, z, linewidth=2, color='#2C4E80', label=f'Control {i+1}' if i == 0 else '')
        
    #     ax.set_xlabel('Raman Shift (cm⁻¹)', fontsize=11, fontweight='bold', labelpad=10)
    #     ax.set_ylabel('Spectrum Index', fontsize=11, fontweight='bold', labelpad=10)
    #     ax.set_zlabel('Intensity', fontsize=11, fontweight='bold', labelpad=10)
    #     ax.set_title('Cancer vs Control Spectra (3D Stacked View)', fontsize=13, fontweight='bold', pad=20)
        
    #     # Set y-ticks to show cancer/control labels
    #     y_ticks = list(range(num_per_class)) + list(range(y_offset, y_offset + num_per_class))
    #     y_labels = [f'C{i+1}' for i in range(num_per_class)] + [f'Ctrl{i+1}' for i in range(num_per_class)]
    #     ax.set_yticks(y_ticks)
    #     ax.set_yticklabels(y_labels)
        
    #     ax.view_init(elev=20, azim=45)
    #     plt.tight_layout()
        
    #     # Save as SVG
    #     savepath = os.path.join(data_folder, 'cancer_vs_control_spectra_3d.svg')
    #     fig.savefig(savepath, format='svg', bbox_inches='tight')
    #     print(f"3D figure saved to {savepath}")
        
    #     plt.show()

    # 2D plot of a single spectrum without grid, frames, and with arrow head axes
    # if cancer_spectra:
    #     fig, ax = plt.subplots(figsize=(16, 6))
        
    #     # Plot first cancer spectrum
    #     spectrum = cancer_spectra[0]
    #     processed = dataset.preprocessor.preprocess(spectrum)
    #     ax.plot(processed.raman_shift_cm, processed.intensity, linewidth=2.5, 
    #             color="#000000", alpha=0.8)
        
    #     # Remove all spines
    #     for spine in ax.spines.values():
    #         spine.set_visible(False)
        
    #     # Remove grid
    #     ax.grid(False)
        
    #     # Add arrow-style axes at the origin
    #     ax.annotate('', xy=(ax.get_xlim()[1], 0), xytext=(ax.get_xlim()[0], 0),
    #                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    #     ax.annotate('', xy=(ax.get_xlim()[0], ax.get_ylim()[1]), 
    #                xytext=(ax.get_xlim()[0], ax.get_ylim()[0]),
    #                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
        
    #     ax.set_xlabel('Raman Shift (cm⁻¹)', fontsize=20, fontweight='bold', labelpad=10)
    #     ax.set_ylabel('Intensity', fontsize=20, fontweight='bold', labelpad=10)
    #     ax.set_title('Spectrum', fontsize=20, fontweight='bold', pad=20)
    #     ax.tick_params(axis='both', labelsize=20)


    #     # Example band definitions (edit ranges/labels/colors to match your figure)
    #     bands = [
    #         {"xmin": 690,  "xmax": 800,  "label": "Nucleic acids",                  "color": "#8FB3D9", "alpha": 0.35},
    #         {"xmin": 800,  "xmax": 980,  "label": "Proteins\nCarbohydrates",        "color": "#9ED3C7", "alpha": 0.35},
    #         {"xmin": 980,  "xmax": 1030, "label": "Phenylalanine",        "color": "#CFE6C8", "alpha": 0.35},
    #         {"xmin": 1100, "xmax": 1250, "label": "Amide III\nLipids", "color": "#E2EBCF", "alpha": 0.35},
    #         {"xmin": 1260, "xmax": 1310, "label": "Nucleic acids",                  "color": "#F6E39A", "alpha": 0.35},
    #         {"xmin": 1310, "xmax": 1360, "label": "Proteins",            "color": "#F0C6A4", "alpha": 0.35},
    #         {"xmin": 1360, "xmax": 1510, "label": "Proteins\nLipids",                "color": "#E4B6CF", "alpha": 0.35},
    #         {"xmin": 1520, "xmax": 1700, "label": "Amide I\nLipids",                 "color": "#A89AD8", "alpha": 0.45},
    #     ]

    #     add_colored_bands(ax, bands, y_top_frac=0.98, label_rotation=90)
  
    #     plt.tight_layout()
        
    #     # Save figure
    #     savepath = os.path.join(data_folder, 'spectrum_2d.svg')
    #     fig.savefig(savepath, format='svg', bbox_inches='tight')
    #     print(f"Spectrum saved to {savepath}")
        
    #     plt.show()

    # def plot_cancer_vs_healthy_multiple_rings(
    #     dataset,
    #     data_folder,
    #     selected_rings,
    #     *,
    #     figsize_per_row=(20, 5),
    #     tsne_kwargs=None,
    #     point_size=50,
    #     point_alpha=0.6,
    #     cancer_color="#8C2928",
    #     healthy_color="#2C4E80",
    # ):
    #     """
    #     Plot mean±std Raman spectra and t-SNE scatter for multiple rings.
    #     Ensures each t-SNE subplot has the same *physical* dimensions across rings
    #     (without forcing identical axis limits).
    #     """
    #     tsne_kwargs = tsne_kwargs or {}
    #     tsne_defaults = dict(n_components=2, random_state=42, init="pca", learning_rate="auto")
    #     tsne_defaults.update(tsne_kwargs)

    #     # ---- Collect + preprocess once per ring ----
    #     ring_data = {}  # ring -> dict with arrays and tsne results
    #     for ring in selected_rings:
    #         ring_spectra = [s for s in dataset.db if s.metadata.get("ring") == ring]
    #         if not ring_spectra:
    #             print(f"No spectra found with ring={ring}")
    #             continue

    #         cancer_proc = []
    #         healthy_proc = []
    #         raman_shift = None

    #         for spectrum in ring_spectra:
    #             staging = (spectrum.metadata.get("staging", "") or "").lower()
    #             if staging not in {"cancer", "control", "healthy"}:
    #                 continue

    #             processed = dataset.preprocessor.preprocess(spectrum)
    #             if raman_shift is None:
    #                 raman_shift = processed.raman_shift_cm

    #             if staging == "cancer":
    #                 cancer_proc.append(processed.intensity)
    #             else:
    #                 healthy_proc.append(processed.intensity)

    #         print(f"Ring {ring}: Found {len(cancer_proc)} cancer spectra and {len(healthy_proc)} healthy spectra")

    #         if len(cancer_proc) == 0 or len(healthy_proc) == 0:
    #             print(f"Need both cancer and healthy spectra to plot for ring {ring}")
    #             continue

    #         cancer_arr = np.asarray(cancer_proc)
    #         healthy_arr = np.asarray(healthy_proc)

    #         # t-SNE (fit on all samples for this ring)
    #         all_intensities = np.vstack([cancer_arr, healthy_arr])
    #         tsne = TSNE(**tsne_defaults)
    #         tsne_results = tsne.fit_transform(all_intensities)

    #         ring_data[ring] = dict(
    #             raman_shift=raman_shift,
    #             cancer_arr=cancer_arr,
    #             healthy_arr=healthy_arr,
    #             tsne=tsne_results,
    #             n_cancer=len(cancer_arr),
    #             n_healthy=len(healthy_arr),
    #         )

    #     rings_to_plot = [r for r in selected_rings if r in ring_data]
    #     if not rings_to_plot:
    #         print("No rings had both cancer and healthy data; nothing to plot.")
    #         return None

    #     # ---- Figure / axes ----
    #     nrows = len(rings_to_plot)
    #     fig_w, fig_h_per_row = figsize_per_row
    #     fig = plt.figure(figsize=(fig_w, fig_h_per_row * nrows), constrained_layout=True)
    #     gs = fig.add_gridspec(nrows=nrows, ncols=2, width_ratios=[1.35, 1.0])  # tune to taste

    #     for i, ring in enumerate(rings_to_plot):
    #         d = ring_data[ring]
    #         ax_spec = fig.add_subplot(gs[i, 0])
    #         ax_tsne = fig.add_subplot(gs[i, 1])

    #         # ---- Mean ± std spectra ----
    #         cancer_mean = d["cancer_arr"].mean(axis=0)
    #         cancer_std = d["cancer_arr"].std(axis=0)
    #         healthy_mean = d["healthy_arr"].mean(axis=0)
    #         healthy_std = d["healthy_arr"].std(axis=0)

    #         rs = d["raman_shift"]
    #         ax_spec.plot(rs, cancer_mean, linewidth=2.5, color=cancer_color, label="Cancer Mean")
    #         ax_spec.fill_between(rs, cancer_mean - cancer_std, cancer_mean + cancer_std,
    #                             color=cancer_color, alpha=0.3, label="Cancer ±1 SD")
    #         ax_spec.plot(rs, healthy_mean, linewidth=2.5, color=healthy_color, label="Healthy Mean")
    #         ax_spec.fill_between(rs, healthy_mean - healthy_std, healthy_mean + healthy_std,
    #                             color=healthy_color, alpha=0.3, label="Healthy ±1 SD")

    #         ax_spec.set_title(f"Ring {ring} - Mean Intensity", fontsize=20, fontweight="bold")
    #         ax_spec.set_xlabel("Raman Shift (cm⁻¹)", fontsize=16)
    #         ax_spec.set_ylabel("Intensity", fontsize=16)
    #         ax_spec.tick_params(axis="both", labelsize=14)
    #         ax_spec.grid(True, alpha=0.3)
    #         ax_spec.legend(fontsize=14)

    #         # ---- t-SNE scatter ----
    #         ts = d["tsne"]
    #         n_c = d["n_cancer"]
    #         colors = np.array([cancer_color] * n_c + [healthy_color] * d["n_healthy"])

    #         ax_tsne.scatter(ts[:, 0], ts[:, 1], c=colors, alpha=point_alpha, s=point_size)
    #         ax_tsne.set_title(f"Ring {ring} - t-SNE Visualization", fontsize=20, fontweight="bold")
    #         ax_tsne.set_xlabel("t-SNE Component 1", fontsize=16)
    #         ax_tsne.set_ylabel("t-SNE Component 2", fontsize=16)
    #         ax_tsne.tick_params(axis="both", labelsize=14)
    #         ax_tsne.grid(True, alpha=0.3)

    #         # ✅ THIS is what you want:
    #         # Fix the *physical* axes box to a square, without forcing shared limits.
    #         ax_tsne.set_box_aspect(1)

    #         # Optional: add a small padding so points don't touch the frame (keeps clusters centered)
    #         ax_tsne.margins(0.08)

    #     # ---- Save ----
    #     os.makedirs(data_folder, exist_ok=True)
    #     savepath = os.path.join(data_folder, "cancer_vs_healthy_multiple_rings.svg")
    #     fig.savefig(savepath, format="svg", bbox_inches="tight")
    #     print(f"Figure saved to {savepath}")

    #     plt.show()
    #     return fig

    # # Call the function with selected rings
    # selected_rings = [1, 5]  # Example of selected rings
    # plot_cancer_vs_healthy_multiple_rings(dataset, data_folder, selected_rings)


    def train_classifiers_by_ring(dataset, data_folder, selected_rings=None):
        """
        Train simple classifiers (Logistic Regression, Decision Tree, Random Forest, SVM) 
        to classify healthy vs cancer spectra, filtered by ring number.
        Ensures no patient appears in both train and test sets.
        Includes baseline accuracy (majority class).
        
        Args:
            dataset: RamanRobotDataset instance
            data_folder: Path to save results
            selected_rings: List of ring numbers (default: None for all rings)
        
        Returns:
            DataFrame with accuracies for each classifier by ring
        """
        
        # Get all unique ring numbers
        ring_numbers = sorted(list(set(
            s.metadata.get('ring') for s in dataset.db 
            if s.metadata.get('ring') is not None
        )))
        
        if selected_rings is not None:
            ring_numbers = [ring for ring in ring_numbers if ring in selected_rings]
        
        print(f"Training classifiers for rings: {ring_numbers}")
        
        results = []
        
        for ring_num in ring_numbers:
            # Filter spectra by ring number
            ring_spectra = [s for s in dataset.db if s.metadata.get('ring') == ring_num]
            
            # Separate cancer and healthy
            cancer_spectra = [s for s in ring_spectra if s.metadata.get('staging', '').lower() == 'cancer']
            healthy_spectra = [s for s in ring_spectra if s.metadata.get('staging', '').lower() in ['control', 'healthy']]
            
            print(f"\nRing {ring_num}: {len(cancer_spectra)} cancer, {len(healthy_spectra)} healthy")
            
            if len(cancer_spectra) < 2 or len(healthy_spectra) < 2:
                print(f"Skipping ring {ring_num}: not enough samples")
                continue
            
            # Get patient IDs
            cancer_patients = [s.metadata.get('patient') for s in cancer_spectra]
            healthy_patients = [s.metadata.get('patient') for s in healthy_spectra]
            
            # Split by patient to avoid data leakage
            unique_cancer_patients = list(set(cancer_patients))
            unique_healthy_patients = list(set(healthy_patients))
            
            # Train-test split at patient level
            train_cancer_patients, test_cancer_patients = train_test_split(
                unique_cancer_patients, test_size=0.3, random_state=42
            )
            train_healthy_patients, test_healthy_patients = train_test_split(
                unique_healthy_patients, test_size=0.3, random_state=42
            )
            
            # Filter spectra by patient split
            X_train_cancer = np.array([dataset.preprocessor.preprocess(s).intensity 
                                       for s in cancer_spectra if s.metadata.get('patient') in train_cancer_patients])
            X_test_cancer = np.array([dataset.preprocessor.preprocess(s).intensity 
                                      for s in cancer_spectra if s.metadata.get('patient') in test_cancer_patients])
            
            X_train_healthy = np.array([dataset.preprocessor.preprocess(s).intensity 
                                        for s in healthy_spectra if s.metadata.get('patient') in train_healthy_patients])
            X_test_healthy = np.array([dataset.preprocessor.preprocess(s).intensity 
                                       for s in healthy_spectra if s.metadata.get('patient') in test_healthy_patients])
            
            if len(X_train_cancer) == 0 or len(X_train_healthy) == 0 or len(X_test_cancer) == 0 or len(X_test_healthy) == 0:
                print(f"Skipping ring {ring_num}: insufficient data after patient-level split")
                continue
            
            # Combine and create labels
            X_train = np.vstack([X_train_cancer, X_train_healthy])
            y_train = np.hstack([np.ones(len(X_train_cancer)), np.zeros(len(X_train_healthy))])
            
            X_test = np.vstack([X_test_cancer, X_test_healthy])
            y_test = np.hstack([np.ones(len(X_test_cancer)), np.zeros(len(X_test_healthy))])
            
            # Calculate baseline accuracy (majority class)
            baseline_accuracy = max(np.sum(y_test == 1), np.sum(y_test == 0)) / len(y_test)
            print(f"  Baseline (Majority Class): {baseline_accuracy:.4f}")
            
            # Standardize features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            # Train classifiers
            classifiers = {
                'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
                'Decision Tree': DecisionTreeClassifier(random_state=42),
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'SVM': SVC(kernel='rbf', random_state=42),
                'MLP (FCN)': MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
            }
            
            ring_results = {'ring': ring_num, 'Baseline (Majority)': baseline_accuracy}
            
            for clf_name, clf in classifiers.items():
                try:
                    clf.fit(X_train, y_train)
                    accuracy = clf.score(X_test, y_test)
                    ring_results[clf_name] = accuracy
                    print(f"  {clf_name}: {accuracy:.4f}")
                except Exception as e:
                    print(f"  {clf_name}: Failed - {str(e)}")
                    ring_results[clf_name] = np.nan
            
            results.append(ring_results)
        
        # Create results DataFrame
        if not results:
            print("No results to plot")
            return None
        
        results_df = pd.DataFrame(results)
        
        # Plot results
        fig, ax = plt.subplots(figsize=(12, 10))
        
        classifiers_to_plot = ['Baseline (Majority)', 'Decision Tree', 'SVM','Random Forest',  'MLP (FCN)', 'Logistic Regression'] # 
        colors = ['#999999',  '#4CAF50', '#FF9800', '#9C27B0', '#8C2928', '#2C4E80',]
        linestyles = ['--', '-', '-', '-', '-', '-']
        
        for clf_name, color, linestyle in zip(classifiers_to_plot, colors, linestyles):
            if clf_name in results_df.columns:
                ax.plot(np.round(results_df['ring']*0.1, 1), results_df[clf_name], 
                    marker='o', linewidth=1.5, markersize=10, label=clf_name, 
                    color=color, linestyle=linestyle)
        
        ax.set_xlabel('Probing distance from the edge (mm)', fontsize=16, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=8, fontweight='bold')
        ax.set_title('Classification Accuracy vs Distance from Edge', fontsize=18, fontweight='bold')
        ax.set_ylim([0, 1.05])
        
        # Set x-ticks only where data points exist
        ax.set_xticks(np.round(results_df['ring']*0.1, 1))
        
        ax.tick_params(axis='both', labelsize=40)
        # ax.legend(fontsize=30, loc='best')
        ax.grid(True, alpha=0.3)
        
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        
        plt.tight_layout()
        
        # Save figure
        savepath = os.path.join(data_folder, 'classifier_accuracy_by_ring.svg')
        fig.savefig(savepath, format='svg', bbox_inches='tight')
        print(f"\nAccuracy plot saved to {savepath}")
        
        plt.show()
        
        return results_df

    # Call the function
    selected_rings = [1, 3, 5, 7]
    classifier_results = train_classifiers_by_ring(dataset, data_folder, selected_rings)
    if classifier_results is not None:
        print("\nClassification Results:")
        print(classifier_results)


    # def calculate_mahalanobis_distance_by_ring(dataset, data_folder, selected_rings=None):
    #     """
    #     Calculate Mahalanobis distance between cancer and healthy classes for each ring.
    #     Uses all available data points without train/test split.
        
    #     Args:
    #         dataset: RamanRobotDataset instance
    #         data_folder: Path to save the plot
    #         selected_rings: List of ring numbers to calculate distances for (default: None for all rings)
        
    #     Returns:
    #         DataFrame with ring numbers and corresponding Mahalanobis distances
    #     """
        
    #     # Get all unique ring numbers
    #     ring_numbers = sorted(list(set(
    #         s.metadata.get('ring') for s in dataset.db 
    #         if s.metadata.get('ring') is not None
    #     )))
        
    #     if selected_rings is not None:
    #         ring_numbers = [ring for ring in ring_numbers if ring in selected_rings]
        
    #     print(f"Calculating Mahalanobis distances for rings: {ring_numbers}")
        
    #     results = []
        
    #     for ring_num in ring_numbers:
    #         # Filter spectra by ring number
    #         ring_spectra = [s for s in dataset.db if s.metadata.get('ring') == ring_num]
            
    #         # Separate cancer and healthy
    #         cancer_spectra = [s for s in ring_spectra if s.metadata.get('staging', '').lower() == 'cancer']
    #         healthy_spectra = [s for s in ring_spectra if s.metadata.get('staging', '').lower() in ['control', 'healthy']]
            
    #         print(f"\nRing {ring_num}: {len(cancer_spectra)} cancer, {len(healthy_spectra)} healthy")
            
    #         if len(cancer_spectra) < 2 or len(healthy_spectra) < 2:
    #             print(f"Skipping ring {ring_num}: not enough samples")
    #             continue
            
    #         # Preprocess all spectra
    #         cancer_intensities = np.array([dataset.preprocessor.preprocess(s).intensity for s in cancer_spectra])
    #         healthy_intensities = np.array([dataset.preprocessor.preprocess(s).intensity for s in healthy_spectra])
            
    #         # Standardize features
    #         scaler = StandardScaler()
    #         all_data = np.vstack([cancer_intensities, healthy_intensities])
    #         scaler.fit(all_data)
            
    #         cancer_intensities = scaler.transform(cancer_intensities)
    #         healthy_intensities = scaler.transform(healthy_intensities)
            
    #         # Calculate means
    #         cancer_mean = np.mean(cancer_intensities, axis=0)
    #         healthy_mean = np.mean(healthy_intensities, axis=0)
            
    #         # Calculate pooled covariance matrix
    #         cancer_cov = np.cov(cancer_intensities, rowvar=False)
    #         healthy_cov = np.cov(healthy_intensities, rowvar=False)
            
    #         n_cancer = len(cancer_intensities)
    #         n_healthy = len(healthy_intensities)
            
    #         pooled_cov = ((n_cancer - 1) * cancer_cov + (n_healthy - 1) * healthy_cov) / (n_cancer + n_healthy - 2)
            
    #         # Add small regularization to avoid singular matrix
    #         pooled_cov += np.eye(pooled_cov.shape[0]) * 1e-6
            
    #         # Calculate inverse of covariance matrix
    #         try:
    #             pooled_cov_inv = linalg.inv(pooled_cov)
    #         except linalg.LinAlgError:
    #             print(f"Warning: Singular covariance matrix for ring {ring_num}, using pseudo-inverse")
    #             pooled_cov_inv = linalg.pinv(pooled_cov)
            
    #         # Calculate Mahalanobis distance between class means
    #         diff = cancer_mean - healthy_mean
    #         mahal_dist = np.sqrt(diff.T @ pooled_cov_inv @ diff)
            
    #         print(f"Ring {ring_num} Mahalanobis Distance: {mahal_dist:.4f}")
            
    #         results.append({
    #             'ring': ring_num,
    #             'mahalanobis_distance': mahal_dist,
    #             'n_cancer': n_cancer,
    #             'n_healthy': n_healthy
    #         })
        
    #     # Create results DataFrame
    #     if not results:
    #         print("No results to plot")
    #         return None
        
    #     results_df = pd.DataFrame(results)
        
    #     # Plot results
    #     fig, ax = plt.subplots(figsize=(12, 6))
    #     ax.plot(results_df['ring'], results_df['mahalanobis_distance'], 
    #             marker='o', linewidth=2.5, markersize=10, color='#8C2928')
        
    #     ax.set_xlabel('Ring Number', fontsize=16, fontweight='bold')
    #     ax.set_ylabel('Mahalanobis Distance', fontsize=16, fontweight='bold')
    #     ax.set_title('Mahalanobis Distance Between Cancer and Healthy Classes', 
    #                     fontsize=18, fontweight='bold')
    #     ax.tick_params(axis='both', labelsize=14)
    #     ax.grid(True, alpha=0.3)
        
    #     for spine in ax.spines.values():
    #         spine.set_linewidth(1.5)
        
    #     plt.tight_layout()
        
    #     # Save figure
    #     savepath = os.path.join(data_folder, 'mahalanobis_distance_by_ring.svg')
    #     fig.savefig(savepath, format='svg', bbox_inches='tight')
    #     print(f"\nMahalanobis distance plot saved to {savepath}")
        
    #     plt.show()
        
    #     return results_df

    # # Call the function with selected rings
    # selected_rings = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # Example of selected rings
    # mahalanobis_results = calculate_mahalanobis_distance_by_ring(dataset, data_folder, selected_rings)
    # if mahalanobis_results is not None:
    #     print("\nMahalanobis Distance Results:")
    #     print(mahalanobis_results)


    # def calculate_mahalanobis_distance_by_ring(dataset, data_folder, selected_rings=None):
    #     """
    #     Calculate Mahalanobis distance between cancer and healthy classes for each ring.
    #     Uses t-SNE to reduce dimensionality to 2 components before calculating distance.
        
    #     Args:
    #         dataset: RamanRobotDataset instance
    #         data_folder: Path to save the plot
    #         selected_rings: List of ring numbers to calculate distances for (default: None for all rings)
        
    #     Returns:
    #         DataFrame with ring numbers and corresponding Mahalanobis distances
    #     """
        
    #     # Get all unique ring numbers
    #     ring_numbers = sorted(list(set(
    #         s.metadata.get('ring') for s in dataset.db 
    #         if s.metadata.get('ring') is not None
    #     )))
        
    #     if selected_rings is not None:
    #         ring_numbers = [ring for ring in ring_numbers if ring in selected_rings]
        
    #     print(f"Calculating Mahalanobis distances for rings: {ring_numbers}")
        
    #     results = []
        
    #     for ring_num in ring_numbers:
    #         # Filter spectra by ring number
    #         ring_spectra = [s for s in dataset.db if s.metadata.get('ring') == ring_num]
            
    #         # Separate cancer and healthy
    #         cancer_spectra = [s for s in ring_spectra if s.metadata.get('staging', '').lower() == 'cancer']
    #         healthy_spectra = [s for s in ring_spectra if s.metadata.get('staging', '').lower() in ['control', 'healthy']]
            
    #         print(f"\nRing {ring_num}: {len(cancer_spectra)} cancer, {len(healthy_spectra)} healthy")
            
    #         if len(cancer_spectra) < 2 or len(healthy_spectra) < 2:
    #             print(f"Skipping ring {ring_num}: not enough samples")
    #             continue
            
    #         # Preprocess all spectra
    #         cancer_intensities = np.array([dataset.preprocessor.preprocess(s).intensity for s in cancer_spectra])
    #         healthy_intensities = np.array([dataset.preprocessor.preprocess(s).intensity for s in healthy_spectra])
            
    #         # Combine data for t-SNE
    #         all_data = np.vstack([cancer_intensities, healthy_intensities])
            
    #         # Apply t-SNE to reduce to 2 components
    #         print(f"Applying t-SNE for ring {ring_num}...")
    #         tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_data) - 1))
    #         tsne_data = tsne.fit_transform(all_data)
            
    #         # Split back into cancer and healthy
    #         cancer_tsne = tsne_data[:len(cancer_intensities)]
    #         healthy_tsne = tsne_data[len(cancer_intensities):]
            
    #         # Calculate means
    #         cancer_mean = np.mean(cancer_tsne, axis=0)
    #         healthy_mean = np.mean(healthy_tsne, axis=0)
            
    #         # Calculate pooled covariance matrix
    #         cancer_cov = np.cov(cancer_tsne, rowvar=False)
    #         healthy_cov = np.cov(healthy_tsne, rowvar=False)
            
    #         n_cancer = len(cancer_tsne)
    #         n_healthy = len(healthy_tsne)
            
    #         pooled_cov = ((n_cancer - 1) * cancer_cov + (n_healthy - 1) * healthy_cov) / (n_cancer + n_healthy - 2)
            
    #         # Add small regularization to avoid singular matrix
    #         pooled_cov += np.eye(pooled_cov.shape[0]) * 1e-6
            
    #         # Calculate inverse of covariance matrix
    #         try:
    #             pooled_cov_inv = linalg.inv(pooled_cov)
    #         except linalg.LinAlgError:
    #             print(f"Warning: Singular covariance matrix for ring {ring_num}, using pseudo-inverse")
    #             pooled_cov_inv = linalg.pinv(pooled_cov)
            
    #         # Calculate Mahalanobis distance between class means
    #         diff = cancer_mean - healthy_mean
    #         mahal_dist = np.sqrt(diff.T @ pooled_cov_inv @ diff)
            
    #         print(f"Ring {ring_num} Mahalanobis Distance (t-SNE space): {mahal_dist:.4f}")
            
    #         results.append({
    #             'ring': ring_num,
    #             'mahalanobis_distance': mahal_dist,
    #             'n_cancer': n_cancer,
    #             'n_healthy': n_healthy
    #         })
        
    #     # Create results DataFrame
    #     if not results:
    #         print("No results to plot")
    #         return None
        
    #     results_df = pd.DataFrame(results)
        
    #     # Plot results
    #     fig, ax = plt.subplots(figsize=(8, 6))
    #     ax.plot(results_df['ring'], results_df['mahalanobis_distance'], 
    #             marker='o', linewidth=2.5, markersize=10, color='#8C2928')
        
    #     ax.set_xlabel('Ring Number', fontsize=16, fontweight='bold')
    #     ax.set_ylabel('Mahalanobis Distance (t-SNE space)', fontsize=16, fontweight='bold')
    #     ax.set_title('Mahalanobis Distance Between Cancer and Healthy Classes (t-SNE)', 
    #                     fontsize=18, fontweight='bold')
    #     ax.tick_params(axis='both', labelsize=14)
    #     ax.grid(True, alpha=0.3)
        
    #     for spine in ax.spines.values():
    #         spine.set_linewidth(1.5)
        
    #     plt.tight_layout()
        
    #     # Save figure
    #     savepath = os.path.join(data_folder, 'mahalanobis_distance_by_ring_tsne.svg')
    #     fig.savefig(savepath, format='svg', bbox_inches='tight')
    #     print(f"\nMahalanobis distance plot saved to {savepath}")
        
    #     plt.show()
        
    #     return results_df

    # # Call the function with selected rings
    # selected_rings = [1, 2, 3, 4, 5]  # Example of selected rings
    # mahalanobis_results = calculate_mahalanobis_distance_by_ring(dataset, data_folder, selected_rings)
    # if mahalanobis_results is not None:
    #     print("\nMahalanobis Distance Results:")
    #     print(mahalanobis_results)


    # def plot_spectra_before_after_processing(dataset, data_folder, selected_rings=None, 
    #                                         n_samples_per_ring=6, figsize=(8, 12)):
    #     """
    #     Plot spectra before and after preprocessing for selected rings.
    #     'Before' applies only the cropping step using the preprocessor's range.
    #     Each ring is colored using the turbo colormap.
        
    #     Args:
    #         dataset: RamanRobotDataset instance
    #         data_folder: Path to save the plot
    #         selected_rings: List of ring numbers to plot (default: None for all rings)
    #         n_samples_per_ring: Number of spectra to plot per ring (default: 3)
    #         figsize: Figure size as (width, height) tuple (default: (16, 10))
    #     """
    #     # Get all unique ring numbers
    #     ring_numbers = sorted(list(set(
    #         s.metadata.get('ring') for s in dataset.db 
    #         if s.metadata.get('ring') is not None
    #     )))
        
    #     if selected_rings is not None:
    #         ring_numbers = [ring for ring in ring_numbers if ring in selected_rings]
        
    #     if not ring_numbers:
    #         print("No rings found in dataset")
    #         return
        
    #     # Cropping parameters from preprocessor
    #     start_rs = dataset.preprocessor.start_raman_shift_cm
    #     end_rs = dataset.preprocessor.end_raman_shift_cm
        
    #     # Create color mapping for rings
    #     colors = plt.cm.turbo(np.linspace(0, 1, len(ring_numbers)))
    #     ring_to_color = {ring: colors[i] for i, ring in enumerate(ring_numbers)}
        
    #     fig, (ax_before, ax_after) = plt.subplots(2, 1, figsize=figsize)
        
    #     # Plot spectra before (cropped only) and after processing
    #     for ring_num in ring_numbers:
    #         ring_spectra = [s for s in dataset.db if s.metadata.get('ring') == ring_num]
            
    #         if not ring_spectra:
    #             continue
            
    #         # Select a few samples from this ring
    #         sample_spectra = ring_spectra[:n_samples_per_ring] if len(ring_spectra) >= n_samples_per_ring else ring_spectra
            
    #         color = ring_to_color[ring_num]
            
    #         # Before (cropping only)
    #         for i, spectrum in enumerate(sample_spectra):
    #             rs = spectrum.raman_shift_cm
    #             mask = (rs >= start_rs) & (rs <= end_rs)
    #             x = rs[mask] if np.any(mask) else rs
    #             y = spectrum.intensity[mask] if np.any(mask) else spectrum.intensity
    #             label = f'Ring {ring_num}' if i == 0 else None
    #             ax_before.plot(x, y, linewidth=1.5, color=color, alpha=0.8, label=label)
            
    #         # After processing (full preprocessor)
    #         for i, spectrum in enumerate(sample_spectra):
    #             processed = dataset.preprocessor.preprocess(spectrum)
    #             label = f'Ring {ring_num}' if i == 0 else None
    #             ax_after.plot(processed.raman_shift_cm, processed.intensity, 
    #                            linewidth=1.5, color=color, alpha=0.8, label=label)
        
    #     # Format before processing plot
    #     ax_before.set_xlabel('Raman Shift (cm⁻¹)', fontsize=14, fontweight='bold')
    #     ax_before.set_ylabel('Intensity', fontsize=14, fontweight='bold')
    #     ax_before.set_title('Before Processing (Cropping Only)', fontsize=16, fontweight='bold')
    #     ax_before.legend(fontsize=10, loc='best')
    #     ax_before.grid(True, alpha=0.3)
    #     ax_before.tick_params(axis='both', labelsize=12)
    #     ax_before.set_ylim([700 , 2300])
        
    #     # Format after processing plot
    #     ax_after.set_xlabel('Raman Shift (cm⁻¹)', fontsize=14, fontweight='bold')
    #     ax_after.set_ylabel('Intensity', fontsize=14, fontweight='bold')
    #     ax_after.set_title('After Processing', fontsize=16, fontweight='bold')
    #     ax_after.legend(fontsize=10, loc='best')
    #     ax_after.grid(True, alpha=0.3)
    #     ax_after.tick_params(axis='both', labelsize=12)
        
    #     plt.tight_layout()
        
    #     # Save figure
    #     savepath = os.path.join(data_folder, 'spectra_before_after_processing.svg')
    #     fig.savefig(savepath, format='svg', bbox_inches='tight')
    #     print(f"Before/after processing plot saved to {savepath}")
        
    #     plt.show()
    #     return fig

    # # # Call the function with selected rings
    # selected_rings = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # Example of selected rings
    # plot_spectra_before_after_processing(dataset, data_folder, selected_rings, n_samples_per_ring=10)