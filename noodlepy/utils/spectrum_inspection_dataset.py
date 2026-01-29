from torch.utils.data import Dataset
import pandas as pd
import os
from noodlepy.utils.spectrum import Spectrum
from noodlepy.utils.spectrumpreprocessor import SpectrumPreprocessor
from noodlepy.utils.spectrumaugmentor import SpectrumAugmentor
import torch
import copy
import numpy as np
import re
import glob
import matplotlib.pyplot as plt


class Dataset(Dataset):
    def __init__(self, data_folder = None,
                 preprocessor = None,
                 augmentor = None):
        """
        Load the database of spectra from the txt file 

        Args:
        data_folder (str): The folder where the spectra are stored
        preprocessor: Spectrum preprocessor object
        augmentor: Spectrum augmentor object

        Returns:
        list[Spectrum]: A list of Spectrum objects
        """
        print("Loading the Raman dataset")
        self.preprocessor = preprocessor
        self.augmentor = augmentor

        list_of_spectrum_objects = []
        list_of_file_paths = []
        
        # Get all txt files in the data folder
        for root, dirs, files in os.walk(data_folder):
            for file in files:
                if file.endswith('.txt'):
                    list_of_file_paths.append(os.path.join(root, file))

        print(f"Found {len(list_of_file_paths)} spectrum files to process...")

        # Process each file
        successful_files = 0
        failed_files = 0
        
        for file_idx, file_path in enumerate(list_of_file_paths):
            if (file_idx + 1) % 100 == 0:
                print(f"Processing file {file_idx + 1}/{len(list_of_file_paths)}")
                
            filename = os.path.basename(file_path)
            
            try:
                # Extract metadata from filename
                patient_annotations = Dataset._extract_patient_labels(filename)
                
                # Load spectrum objects from file
                spectrum_objects = Dataset._load_files_to_spectrum_objects(file_path, patient_annotations)
                
                if spectrum_objects:
                    list_of_spectrum_objects += spectrum_objects
                    successful_files += 1
                else:
                    failed_files += 1
                    
            except Exception as e:
                print(f"Error processing file '{filename}': {e}")
                failed_files += 1
                continue

        self.db = list_of_spectrum_objects
        print(f"Successfully loaded {len(self.db)} spectra from {successful_files} files")
        if failed_files > 0:
            print(f"Failed to load {failed_files} files")

    def __len__(self):
        return len(self.db)
    
    def __getitem__(self, 
                    idx: int
                    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Return 2 augmented spectra from the chosen spectrum along with its wavenumber

        Args:
        idx (int): The index of the spectrum to augment

        Returns:
        A tuple containing:
          - preprocessed spectrum intensity tensor (torch.Tensor)
          - preprocessed spectrum wavenumber tensor (torch.Tensor)
          - metadata dictionary (dict)
        """
        chosen_spectrum: Spectrum = self.db[idx]
        
        if self.preprocessor is not None:
            preprocessed_spectrum = self.preprocessor.preprocess(chosen_spectrum)
        else:
            preprocessed_spectrum = chosen_spectrum
        
        if self.augmentor is not None:
            augmented_spectrum_1, augmented_spectrum_2 = self.augmentor.augment(preprocessed_spectrum, 2)
        else:
            augmented_spectrum_1 = preprocessed_spectrum
            augmented_spectrum_2 = preprocessed_spectrum

        intensity_tensor = torch.tensor(augmented_spectrum_1.intensity, dtype=torch.float32).unsqueeze(0)
        # Note: Using wavelength_nm attribute - if your Spectrum class uses raman_shift_cm, change this accordingly
        wavelength_tensor = torch.tensor(augmented_spectrum_1.wavelength_nm, dtype=torch.float32).unsqueeze(0)

        return intensity_tensor, wavelength_tensor, chosen_spectrum.metadata

    @staticmethod
    def _extract_patient_labels(spectrum_file_name: str) -> dict:
        """
        Extract metadata from filename.

        Assumes pattern:
            "<meta_type>_<value>_<meta_type>_<value>_..._x<...>_y<...>.txt"

        Example:
        "patient_538_staging_cancer_sample_0_point_2_ring_3_rep_19_x-799.82_y-4.91.txt"

        This becomes:
            {
                "patient": 538,
                "staging": "cancer",
                "sample": 0,
                "point": 2,
                "ring": 3,
                "rep": 19,
                "patient_id": 538,   # convenience alias
            }

        Notes:
        - Does NOT store x/y in the metadata (they are always the last 2 items).
        - Does NOT hard-code metadata types; it just interprets name/value pairs.
        """
        # Remove file extension
        basename = os.path.splitext(spectrum_file_name)[0]

        # Split on underscores
        parts = basename.split("_")

        # Ignore the last two parts if they are x... and y...
        n = len(parts)
        if n >= 4 and parts[-2].startswith("x") and parts[-1].startswith("y"):
            effective_len = n - 2
        else:
            effective_len = n

        metadata = {}

        # Walk through parts in (key, value) pairs
        i = 0
        while i + 1 < effective_len:
            key = parts[i]
            value_str = parts[i + 1]

            # Try to cast value to int or float, otherwise leave as string
            value = value_str
            try:
                value = int(value_str)
            except ValueError:
                try:
                    value = float(value_str)
                except ValueError:
                    value = value_str  # leave as string

            metadata[key] = value
            i += 2

        # Small debug print for first few files
        if hasattr(Dataset, '_debug_count'):
            Dataset._debug_count += 1
        else:
            Dataset._debug_count = 1

        if Dataset._debug_count <= 3:
            print(f"Parsed metadata from '{spectrum_file_name}':")
            for k, v in metadata.items():
                print(f"  {k}: {v}")

        return metadata

    
    @staticmethod
    def _load_files_to_spectrum_objects(file_path: str,
                                       patient_annotations: dict) -> list:
        """
        Load spectrum data from CSV file and create ONE median Spectrum object.

        For files with multiple spectra stacked vertically (same wavelengths
        repeated multiple times), this function:
          - groups by wavelength (column 0)
          - takes the median of intensity (column 1) for each wavelength
          - returns a single "median" spectrum for the file

        Args:
        file_path (str): Path to the spectrum data file
        patient_annotations (dict): Metadata extracted from filename

        Returns:
        list: List with a single Spectrum object (or empty list on error)
        """
        try:
            # Load CSV data with comma delimiter
            data = pd.read_csv(file_path, sep=",", header=None)

            # Check if we have the expected format
            if data.shape[1] < 2:
                print(f"Warning: File '{file_path}' has unexpected format. Expected at least 2 columns.")
                return []

            # data.iloc[:, 0] = wavelength, data.iloc[:, 1] = intensity

            # Group by wavelength and take median intensity
            # This automatically handles both:
            #   - single spectrum (no repeats)
            #   - multiple spectra with repeated wavelengths
            median_series = data.groupby(0)[1].median()

            # Extract wavelength and median intensity
            wavelength = median_series.index.to_numpy()
            intensity = median_series.to_numpy()

            # Ensure wavelengths are sorted
            order = np.argsort(wavelength)
            wavelength = wavelength[order]
            intensity = intensity[order]

            metadata = copy.deepcopy(patient_annotations)

            # Create a single Spectrum object with the median spectrum
            spectrum_obj = Spectrum(
                wavelength_nm=wavelength.round(3),
                intensity=intensity.round(3),
                metadata=metadata
            )

            return [spectrum_obj]

        except Exception as e:
            print(f"Error loading data from '{file_path}': {e}")
            return []



if __name__ == "__main__":
    # Example usage
    data_folder = r'C:\Users\Yifei\Downloads\data_izabella_filtered\data_izabella_filtered'

    preprocessor = SpectrumPreprocessor(cropping=False,
                                        baseline_correction=False,
                                        remove_cosmic_rays=False,
                                        normalization=True,
                                        smoothing=False)

    # preprocessor = None
    augmentor = None

    dataset = Dataset(data_folder, preprocessor, augmentor)

    print(f"Dataset length: {len(dataset)}")
    
    # Show sample metadata
    if len(dataset) > 0:
        sample_spectrum = dataset.db[0]
        print(f"\nSample spectrum metadata:")
        for key, value in sample_spectrum.metadata.items():
            print(f"  {key}: {value}")

    # Create visualization
    fig, (ax_cancer, ax_control) = plt.subplots(1, 2, figsize=(12, 6))
    
    cancer_count = 0
    control_count = 0
    
    for i in range(min(len(dataset), 100)):  # Limit to first 100 for performance
        intensity_tensor, wavelength_tensor, metadata = dataset[i]
        
        # Convert tensors back to numpy for plotting
        intensity = intensity_tensor.squeeze().numpy()
        wavelength = wavelength_tensor.squeeze().numpy()
        
        # Determine which subplot to use based on patient staging (case insensitive)
        stage = metadata.get('staging', '').lower()
        if stage == "cancer" and cancer_count < 20:  # Limit number of plots
            ax_cancer.plot(wavelength, intensity, color="red", alpha=0.7,
                           label="Cancer" if cancer_count == 0 else "")
            cancer_count += 1
        elif stage in ["control", "benign", "normal"] and control_count < 20:
            ax_control.plot(wavelength, intensity, color="blue", alpha=0.7,
                            label="Control/Benign" if control_count == 0 else "")
            control_count += 1
    
    # Set axis labels, titles, and limits for cancer subplot
    ax_cancer.set_title(f"Cancer Spectra (n={cancer_count})")
    ax_cancer.set_xlabel("Wavelength (nm) / Raman Shift (cm⁻¹)")
    ax_cancer.set_ylabel("Intensity")
    ax_cancer.legend(loc="upper right", fontsize='small')
    
    # Set axis labels, titles, and limits for control subplot
    ax_control.set_title(f"Control/Benign Spectra (n={control_count})")
    ax_control.set_xlabel("Wavelength (nm) / Raman Shift (cm⁻¹)")
    ax_control.set_ylabel("Intensity")
    ax_control.legend(loc="upper right", fontsize='small')

    plt.tight_layout()
    plt.show()