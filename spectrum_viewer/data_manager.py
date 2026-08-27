"""Data manager for the spectrum viewer that wraps RamanRobotDataset."""

import copy
import os
import numpy as np
from typing import Dict, List, Optional, Any
from collections import defaultdict

from utils.spectrum import Spectrum
from utils.raman_robot_dataset import RamanRobotDataset, SpectrumPreprocessorWrapper


class ViewerDataManager:
    """
    Wrapper around RamanRobotDataset for viewer-specific operations.

    This class wraps the PyTorch-compatible RamanRobotDataset and provides
    additional functionality for the spectrum viewer:
    - Grouping spectra by sample
    - Extracting intensity at specific wavenumbers for hyperspectral maps
    - Managing preprocessed data objects
    - Multi-experiment loading and management
    """

    def __init__(self, data_folder: str = None, preprocessor: SpectrumPreprocessorWrapper = None):
        """
        Initialize the data manager.

        Args:
            data_folder: Path to the data folder containing spectra/ and metadata/
            preprocessor: SpectrumPreprocessorWrapper instance for preprocessing
        """
        self.data_folder = data_folder
        self.preprocessor = preprocessor
        self.dataset: Optional[RamanRobotDataset] = None
        self._original_data_objects: List[Spectrum] = []
        self._data_objects: List[Spectrum] = []
        self._samples_cache: Optional[Dict[Any, List[int]]] = None
        self._common_grid: Optional[np.ndarray] = None
        self._intensity_matrix: Optional[np.ndarray] = None

        # Multi-experiment tracking
        self.experiments: Dict[int, Dict[str, Any]] = {}  # {experiment_id: {'path', 'folder_name', 'spectrum_count', 'start_idx'}}
        self._next_experiment_id: int = 0
        self._next_droplet_id: int = 0  # Global counter across experiments

        if data_folder is not None:
            self.load_folder(data_folder, preprocessor)

    # --- PyTorch Dataset passthrough ---
    def __len__(self) -> int:
        return len(self._data_objects)

    def __getitem__(self, idx: int) -> Spectrum:
        """Return preprocessed spectrum at index."""
        return self._data_objects[idx]

    @property
    def data_objects(self) -> List[Spectrum]:
        """Return list of preprocessed Spectrum objects."""
        return self._data_objects

    @property
    def original_data_objects(self) -> List[Spectrum]:
        """Return list of original (unprocessed) Spectrum objects."""
        return self._original_data_objects

    def load_folder(self, path: str, preprocessor: SpectrumPreprocessorWrapper = None):
        """
        Load experiment data from a folder (replaces all existing data).

        Args:
            path: Path to the data folder
            preprocessor: Optional preprocessor to use
        """
        # Clear existing data and add as first experiment
        self.clear_all()
        self.add_experiment(path, preprocessor)

    def add_experiment(self, path: str, preprocessor: SpectrumPreprocessorWrapper = None) -> int:
        """
        Load and append experiment data from a folder.

        Args:
            path: Path to the data folder
            preprocessor: Optional preprocessor to use

        Returns:
            experiment_id: Integer ID for the added experiment
        """
        self.data_folder = path
        if preprocessor is not None:
            self.preprocessor = preprocessor

        # Create dataset for the new experiment
        dataset = RamanRobotDataset(
            data_folder=path,
            preprocessor=self.preprocessor
        )

        if len(dataset.db) == 0:
            print(f"No spectra found in {path}")
            return -1

        # Get folder name for this experiment
        folder_name = os.path.basename(path.rstrip('/\\'))

        # Assign experiment ID
        experiment_id = self._next_experiment_id
        self._next_experiment_id += 1

        # Record starting index for this experiment's spectra
        start_idx = len(self._original_data_objects)

        # Add experiment metadata to each spectrum and assign droplet IDs
        new_spectra = copy.deepcopy(dataset.db)
        droplet_map = {}

        for obj in new_spectra:
            # Add experiment metadata
            obj.metadata['experiment_id'] = experiment_id
            obj.metadata['experiment_folder'] = folder_name

            # Assign droplet IDs using global counter
            patient = str(obj.metadata.get('patient', 'unknown'))
            sample = str(obj.metadata.get('sample', 'unknown'))
            key = (folder_name, patient, sample)

            if key not in droplet_map:
                droplet_map[key] = self._next_droplet_id
                self._next_droplet_id += 1

            obj.metadata['date'] = folder_name
            obj.metadata['droplet_id'] = droplet_map[key]

        # Append to original data objects
        self._original_data_objects.extend(new_spectra)

        # Store experiment info
        self.experiments[experiment_id] = {
            'path': path,
            'folder_name': folder_name,
            'spectrum_count': len(new_spectra),
            'start_idx': start_idx
        }

        print(f"Added experiment '{folder_name}' (ID: {experiment_id}) with {len(new_spectra)} spectra")
        print(f"Assigned {len(droplet_map)} new droplet IDs")

        # Reprocess all data to ensure consistent common grid
        self._preprocess_data()

        # Clear caches
        self._samples_cache = None

        print(f"Total spectra: {len(self._data_objects)}")
        return experiment_id

    def remove_experiment(self, experiment_id: int) -> bool:
        """
        Remove an experiment's spectra from the dataset.

        Args:
            experiment_id: Integer ID of the experiment to remove

        Returns:
            True if experiment was removed, False if not found
        """
        if experiment_id not in self.experiments:
            print(f"Experiment ID {experiment_id} not found")
            return False

        exp_info = self.experiments[experiment_id]
        folder_name = exp_info['folder_name']

        # Remove spectra with this experiment_id from original data
        self._original_data_objects = [
            obj for obj in self._original_data_objects
            if obj.metadata.get('experiment_id') != experiment_id
        ]

        # Remove from experiments dict
        del self.experiments[experiment_id]

        print(f"Removed experiment '{folder_name}' (ID: {experiment_id})")

        # Reprocess remaining data
        if self._original_data_objects:
            self._preprocess_data()
        else:
            self._data_objects = []
            self._common_grid = None
            self._intensity_matrix = None

        # Clear caches
        self._samples_cache = None

        print(f"Remaining spectra: {len(self._data_objects)}")
        return True

    def clear_all(self):
        """Clear all loaded experiment data."""
        self._original_data_objects = []
        self._data_objects = []
        self._samples_cache = None
        self._common_grid = None
        self._intensity_matrix = None
        self.experiments = {}
        self._next_experiment_id = 0
        self._next_droplet_id = 0
        self.dataset = None
        print("Cleared all experiment data")

    def get_experiments(self) -> List[Dict[str, Any]]:
        """
        Return list of loaded experiments with info.

        Returns:
            List of dicts with 'id', 'path', 'folder_name', 'spectrum_count'
        """
        return [
            {
                'id': exp_id,
                'path': info['path'],
                'folder_name': info['folder_name'],
                'spectrum_count': info['spectrum_count']
            }
            for exp_id, info in self.experiments.items()
        ]

    def _assign_droplet_ids(self):
        """Assign droplet_id to each spectrum based on (date, patient, sample).

        Note: This method is now only used for backward compatibility.
        The add_experiment method handles droplet ID assignment for new experiments.
        """
        if not self._original_data_objects:
            return

        folder_name = os.path.basename(self.data_folder.rstrip('/\\')) if self.data_folder else 'unknown'

        droplet_map = {}

        for obj in self._original_data_objects:
            # Skip if already has experiment_id (handled by add_experiment)
            if 'experiment_id' in obj.metadata:
                continue

            patient = str(obj.metadata.get('patient', 'unknown'))
            sample = str(obj.metadata.get('sample', 'unknown'))
            key = (folder_name, patient, sample)

            if key not in droplet_map:
                droplet_map[key] = self._next_droplet_id
                self._next_droplet_id += 1

            obj.metadata['date'] = folder_name
            obj.metadata['droplet_id'] = droplet_map[key]

        if droplet_map:
            print(f"Assigned {len(droplet_map)} unique droplet IDs from folder '{folder_name}'")

    def _preprocess_data(self):
        """Preprocess all data objects and align to common grid."""
        self._intensity_matrix = None
        if self.preprocessor is None:
            self._data_objects = copy.deepcopy(self._original_data_objects)
            return

        print(f"Preprocessing {len(self._original_data_objects)} spectra...")
        self._data_objects = []
        for i, obj in enumerate(self._original_data_objects):
            try:
                processed = self.preprocessor.preprocess(copy.deepcopy(obj))
                self._data_objects.append(processed)
            except Exception as e:
                print(f"Error preprocessing spectrum {i}: {e}")
                # Keep original if preprocessing fails
                self._data_objects.append(copy.deepcopy(obj))

        # Align spectra to common grid
        self._align_spectra_to_common_grid()
        print(f"Successfully preprocessed {len(self._data_objects)} spectra")

    def _align_spectra_to_common_grid(self):
        """Align all spectra to a common wavenumber grid using interpolation."""
        if not self._data_objects:
            return

        # Find the common wavenumber range across all spectra
        min_raman = max(obj.raman_shift_cm.min() for obj in self._data_objects)
        max_raman = min(obj.raman_shift_cm.max() for obj in self._data_objects)

        # Find the spectrum with the most points to use as reference
        reference_spectrum = max(self._data_objects, key=lambda obj: len(obj.raman_shift_cm))

        # Create a common grid within the overlapping range
        reference_mask = (reference_spectrum.raman_shift_cm >= min_raman) & \
                        (reference_spectrum.raman_shift_cm <= max_raman)
        self._common_grid = reference_spectrum.raman_shift_cm[reference_mask]

        print(f"Aligning spectra to common grid: {len(self._common_grid)} points, "
              f"range [{min_raman:.2f}, {max_raman:.2f}] cm^-1")

        # Interpolate all spectra to the common grid
        for obj in self._data_objects:
            if len(obj.raman_shift_cm) != len(self._common_grid) or \
               not np.allclose(obj.raman_shift_cm, self._common_grid):
                # Need to interpolate
                interpolated_intensity = np.interp(self._common_grid, obj.raman_shift_cm, obj.intensity)
                obj.raman_shift_cm = self._common_grid.copy()
                obj.intensity = interpolated_intensity

    def reprocess(self, preprocessor: SpectrumPreprocessorWrapper = None):
        """
        Re-preprocess all data with updated preprocessor settings.

        Args:
            preprocessor: New preprocessor to use, or None to use current
        """
        if preprocessor is not None:
            self.preprocessor = preprocessor

        self._preprocess_data()
        self._samples_cache = None
        self._intensity_matrix = None

    def get_experiment_count(self) -> int:
        """Return number of loaded experiments."""
        return len(self.experiments)

    # --- Viewer-specific methods ---

    def get_samples(self) -> Dict[Any, List[int]]:
        """
        Group spectra by droplet_id (unique per date+patient+sample).

        Falls back to sample metadata if droplet_id is not available.

        Returns:
            Dictionary mapping droplet_id to list of spectrum indices
        """
        if self._samples_cache is not None:
            return self._samples_cache

        samples = defaultdict(list)
        for idx, obj in enumerate(self._data_objects):
            sample_id = obj.metadata.get('droplet_id', obj.metadata.get('sample', 'unknown'))
            samples[sample_id].append(idx)

        self._samples_cache = dict(samples)
        return self._samples_cache

    def get_common_wavenumber_grid(self) -> np.ndarray:
        """
        Get the common wavenumber grid that all spectra are aligned to.

        Returns:
            1D numpy array of wavenumber values (cm^-1)
        """
        if self._common_grid is not None:
            return self._common_grid

        if self._data_objects:
            return self._data_objects[0].raman_shift_cm

        return np.array([])

    def _build_intensity_matrix(self):
        """Build cached 2D intensity matrix (n_spectra x n_wavenumbers) from all spectra."""
        if not self._data_objects:
            self._intensity_matrix = np.array([])
            return
        self._intensity_matrix = np.array([obj.intensity for obj in self._data_objects])

    def get_intensity_at_wavenumber(self, wavenumber: float, indices: List[int] = None) -> np.ndarray:
        """
        Get intensity values at a specific wavenumber for all or selected spectra.

        Uses vectorized numpy indexing with the cached intensity matrix when spectra
        are aligned to a common grid. Falls back to per-spectrum interpolation otherwise.

        Args:
            wavenumber: Target wavenumber in cm^-1
            indices: List of spectrum indices, or None for all

        Returns:
            1D numpy array of intensity values
        """
        if indices is None:
            indices = list(range(len(self._data_objects)))

        if not self._data_objects:
            return np.array([])

        # Build intensity matrix lazily
        if self._intensity_matrix is None:
            self._build_intensity_matrix()

        grid = self.get_common_wavenumber_grid()

        if len(grid) > 0 and self._intensity_matrix is not None and self._intensity_matrix.size > 0:
            # Vectorized: find interpolation position in common grid
            idx_float = np.interp(wavenumber, grid, np.arange(len(grid)))
            idx_lo = int(np.floor(idx_float))
            idx_hi = min(idx_lo + 1, len(grid) - 1)
            frac = idx_float - idx_lo

            rows = np.asarray(indices)
            if idx_lo == idx_hi:
                return self._intensity_matrix[rows, idx_lo]
            else:
                return self._intensity_matrix[rows, idx_lo] * (1 - frac) + \
                       self._intensity_matrix[rows, idx_hi] * frac

        # Fallback: per-spectrum interpolation
        intensities = []
        for idx in indices:
            obj = self._data_objects[idx]
            intensity = np.interp(wavenumber, obj.raman_shift_cm, obj.intensity)
            intensities.append(intensity)

        return np.array(intensities)

    def get_spatial_coordinates(self, indices: List[int] = None) -> tuple:
        """
        Get x, y coordinates for spectra from metadata.

        Args:
            indices: List of spectrum indices, or None for all

        Returns:
            Tuple of (x_coords, y_coords) as numpy arrays
        """
        if indices is None:
            indices = list(range(len(self._data_objects)))

        x_coords = []
        y_coords = []
        for idx in indices:
            obj = self._data_objects[idx]
            x = obj.metadata.get('x', 0)
            y = obj.metadata.get('y', 0)
            x_coords.append(x if x is not None else 0)
            y_coords.append(y if y is not None else 0)

        return np.array(x_coords), np.array(y_coords)

    def get_metadata_keys(self) -> List[str]:
        """Return sorted list of metadata keys across all spectra."""
        metadata_keys = set()
        for obj in self._data_objects:
            metadata_keys.update(obj.metadata.keys())
        return sorted(metadata_keys)

    def get_metadata_values(self, key: str) -> List[Any]:
        """
        Get all unique values for a metadata key.

        Args:
            key: Metadata key to query

        Returns:
            Sorted list of unique values
        """
        values = set()
        for obj in self._data_objects:
            if key in obj.metadata:
                val = obj.metadata[key]
                if val is not None:
                    values.add(str(val))

        # Try to sort numerically, fall back to string
        try:
            return sorted(values, key=lambda x: float(x) if x.replace('.', '', 1).replace('-', '', 1).isdigit() else float('inf'))
        except (ValueError, TypeError):
            return sorted(values)

    def get_wavenumber_range(self) -> tuple:
        """
        Get the min and max wavenumber values.

        Returns:
            Tuple of (min_wavenumber, max_wavenumber)
        """
        grid = self.get_common_wavenumber_grid()
        if len(grid) == 0:
            return (0, 0)
        return (float(grid.min()), float(grid.max()))

    def get_spectrum_count(self) -> int:
        """Return total number of spectra."""
        return len(self._data_objects)

    def get_sample_count(self) -> int:
        """Return number of unique samples."""
        samples = self.get_samples()
        return len(samples)
