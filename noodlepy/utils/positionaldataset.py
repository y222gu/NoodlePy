import torch
from torch.utils.data import Dataset
import os
import pandas as pd
from noodlepy.utils.spectrum import Spectrum
from noodlepy.utils.spectrumpreprocessor import SpectrumPreprocessor
from noodlepy.utils.spectrumaugmentor import SpectrumAugmentor


class PositionalDataset(Dataset):
    def __init__(self, data_folder, annotation_file_path, preprocessor):
        """
        Args:
            spectra (ndarray): 2D array where each row is a Raman spectrum.
            positions (ndarray): 2D array where each row is a position (x, y) or (r, θ).
            labels (ndarray): 1D array of labels for each spectrum.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        print("Loading the Raman dataset")
        self.preprocessor = preprocessor

        list_of_spectrum_objects = []
        list_of_file_names = sorted([f for f in os.listdir(data_folder) if f.endswith(('.txt'))])

        annotation_all = pd.read_excel(annotation_file_path)

        for filename in list_of_file_names:
            patient_annotations = PositionalDataset._extract_patient_labels(filename, annotation_all)
            spectrum_objects = PositionalDataset._load_files_to_spectrum_objects(data_folder, filename, patient_annotations)
            list_of_spectrum_objects+=spectrum_objects

        self.db = list_of_spectrum_objects
        print(f"Loaded {len(self.db)} spectra")

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.db)

    def __getitem__(self, idx):
        """
        Retrieve a single sample from the dataset.

        Args:
            idx (int): The index of the sample.

        Returns:
            dict: A dictionary containing the spectrum, position, and label.
        """
        # Get the spectrum, position, and label for the given index
        raw_spectrum, position, label = self.db[idx].spectrum, self.db[idx].position, self.db[idx].label
        
        # preprocess the spectrum
        preprocessed_spectrum = self.preprocessor.preprocess(raw_spectrum)

        # Convert the spectrum, position, and label to tensors
        spectrum = torch.tensor(preprocessed_spectrum, dtype=torch.float32).unsqueeze(0)  # Add channel dimension (1, n_wavelengths)
        position = torch.tensor(preprocessed_spectrum, dtype=torch.float32)  # Positional encoding (x, y) or (r, θ)
        label = torch.tensor(label, dtype=torch.long)  # Multi-class classification labels
            
        return spectrum, position, label

    @staticmethod
    def _extract_patient_labels(filename, annotation_all):
        # patient number
        patient_number = filename.split('_')[0]
        patient_annotations = annotation_all[annotation_all['OD Number'] == int(patient_number)]
        position_annotations = patient_annotations[['Start', 'End']]
        cancer_stage = patient_annotations['Staging'].values
        return patient_annotations, position_annotations, cancer_stage
    
    @staticmethod
    def _load_files_to_spectrum_objects(data_folder, filename, patient_annotations):
        list_of_spectrum_objects = []
        for index, row in patient_annotations.iterrows():
            spectrum = Spectrum(data_folder, filename, row['Start'], row['End'])
            list_of_spectrum_objects.append(spectrum)
        return list_of_spectrum_objects
