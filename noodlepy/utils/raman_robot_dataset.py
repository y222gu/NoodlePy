from torch.utils.data import Dataset
from noodlepy.utils.spectrum import Spectrum
from noodlepy.utils.spectrumpreprocessor import SpectrumPreprocessor
from combat.pycombat import pycombat
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import copy


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

        if os.path.isdir(metadata_dir):
            # 1) Build a table of file-level metadata (one row per metadata file)
            meta_df = self._build_metadata_table(metadata_dir, spectra_dir)

            # 2) Compute relative timestamps exactly as in the surface-plot script
            meta_df = self._add_relative_timestamps(meta_df)

            self.meta_df = meta_df  # keep for later inspection if needed

            # 3) Create Spectrum objects for each spectra file, attaching metadata
            list_of_spectrum_objects = []

            for row in meta_df.to_dict(orient="records"):
                rel_spec_path = row["spectrum_rel_path"]
                base_metadata = {
                    k: v for k, v in row.items()
                    if k not in ("spectrum_rel_path", "metadata_rel_path")
                }

                spectrum_objects = self._load_files_to_spectrum_objects(
                    data_folder=self.data_folder,
                    filename=rel_spec_path,
                    base_metadata=base_metadata,
                )
                list_of_spectrum_objects += spectrum_objects

        else:
            print(f"Metadata folder not found: {metadata_dir}")
            list_of_spectrum_objects = []
            list_of_files = os.listdir(spectra_dir)
            for file in list_of_files:
                # parse the file name to get metadata
                base_metadata = self._parse_filename(file)

                if file.endswith('.txt'):
                    spectrum_objects = self._load_files_to_spectrum_objects(
                        data_folder=spectra_dir,
                        filename=file,
                        base_metadata=base_metadata,
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
        cropped_spectrum = chosen_spectrum.crop_spectrum(625.277, 1786.791)
        preprocessed_spectrum = preprocessor.preprocess(cropped_spectrum)
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

        # remove the one with x and y
        split = [s for s in split if not ('x' in s or 'y' in s)]

        # every two elements are a key-value pair
        matches = [(split[i], split[i + 1]) for i in range(0, len(split), 2)]

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
        
        # Ensure required fields exist
        # required_fields = ['patient', 'staging', 'sample', 'point', 'rep', 'ring']
        # for field in required_fields:
        #     if field not in d:
        #         raise ValueError(f"Required field '{field}' not found in filename: {filename}")
        
        # # If line is not present, set it equal to point
        # if 'line' not in d:
        #     d['line'] = d['point']
        
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
    def _add_relative_timestamps(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add a single relative time column in minutes from the earliest timestamp.
        """
        if "timestamp" not in df.columns:
            raise KeyError("No 'timestamp' column found in metadata DataFrame.")

        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        if ts.isna().any():
            bad = df.loc[ts.isna(), "timestamp"].unique()
            raise ValueError(
                "Some timestamps could not be parsed. Examples:\n"
                + "\n".join(map(str, bad[:10]))
            )

        df = df.copy()
        baseline = ts.min()
        df["timestamp"] = (ts - baseline).dt.total_seconds() / 60.0
        return df

    @staticmethod
    def _build_metadata_table(metadata_dir: str, spectra_dir: str) -> pd.DataFrame:
        """
        Build a DataFrame with one row per metadata file, containing:

          - spectrum_rel_path, metadata_rel_path
          - filename-derived metadata (patient, staging, sample, point, line, ring, rep, x, y)
          - file metadata contents (number_of_repetitions, integration_time, laser_power,
            prusa_position, nanodrive_position, timestamp, ...)
          - focus_position = prusa_position - 0.001 * nanodrive_position
        """
        rows = []
        metadata_dir = Path(metadata_dir)
        spectra_dir = Path(spectra_dir)

        n_total = 0
        n_used = 0

        for path in metadata_dir.glob("*.txt"):
            fname = path.name
            if not fname.endswith("_metadata.txt"):
                continue

            n_total += 1
            try:
                # Base name without "_metadata"
                base_name = fname.replace("_metadata.txt", ".txt")

                spectrum_path = spectra_dir / base_name
                if not spectrum_path.is_file():
                    print(f"Warning: spectrum file not found for metadata '{fname}'. Expected '{spectrum_path.name}'. Skipping.")
                    continue

                # Paths relative to dataset root (one level above spectra/metadata)
                data_root = spectra_dir.parent
                spectrum_rel_path = os.path.join("spectra", base_name)
                metadata_rel_path = os.path.join("metadata", fname)

                # Parse filename (use base_name – same pattern as spectra)
                fn_meta = RamanRobotDataset._parse_filename(base_name)

                # Parse metadata file contents
                file_meta = RamanRobotDataset._parse_metadata_file(str(path))

                # Combine
                row = {
                    "spectrum_rel_path": spectrum_rel_path,
                    "metadata_rel_path": metadata_rel_path,
                }
                row.update(fn_meta)
                row.update(file_meta)

                # Compute focus_position if prusa/nanodrive are present
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

                rows.append(row)
                n_used += 1

            except Exception as e:
                print(f"Warning: failed to parse metadata file '{fname}': {e}")

        if not rows:
            raise RuntimeError(f"No valid *_metadata.txt files found in {metadata_dir}")

        print(f"Successfully used {n_used} metadata files out of {n_total} found.")
        return pd.DataFrame(rows)

    # ----------------- SPECTRA LOADING (unchanged logic, but with new metadata) -----------------

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
