import torch
from torch.utils.data import Dataset
import os
import pandas as pd
from noodlepy.utils.spectrum import Spectrum
from noodlepy.utils.spectrumpreprocessor import SpectrumPreprocessor
from noodlepy.utils.spectrumaugmentor import SpectrumAugmentor
import copy
import numpy as np


class PositionalDataset(Dataset):
    def __init__(self, data_folder, annotation_file_path, preprocessor):
        """
        Args:
            spectra (ndarray): 2D array where each row is a Raman spectrum.
            positions (ndarray): 2D array where each row is a position (r, θ).
            labels (ndarray): 1D array of labels for each spectrum.
        """
        print("Loading the Raman dataset")
        self.preprocessor = preprocessor

        list_of_spectrum_objects = []
        list_of_file_names = sorted([f for f in os.listdir(data_folder) if f.endswith(('.txt'))])

        annotation_all = pd.read_excel(annotation_file_path)

        for filename in list_of_file_names:
            metadata = PositionalDataset._extract_patient_labels(filename, annotation_all)
            spectrum_objects = PositionalDataset._load_files_to_spectrum_objects(data_folder, filename, metadata)
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
        chosen_spectrum:Spectrum = self.db[idx]
        metadata = self.db[idx].metadata
        # preprocess the spectrum
        preprocessed_spectrum = self.preprocessor.preprocess(chosen_spectrum)

        preprocessed_spectrum_intensity = torch.tensor(preprocessed_spectrum.intensity, dtype=torch.float32).unsqueeze(0)
 
        return preprocessed_spectrum_intensity, metadata
    
    @staticmethod
    def _load_files_to_spectrum_objects(data_folder:str,
                               filename:str, 
                               metadata:dict):

        with open(os.path.join(data_folder, filename)) as f:
            data = pd.read_csv(f, sep=",", header=None)
            repeated_wavelengths = data.iloc[:,0].value_counts()
            first_repeated_wavelength = repeated_wavelengths.idxmax()
            start_indexes = data[data.iloc[:,0] == first_repeated_wavelength].index.tolist()
            spectrum_objects = []
            # split the repeated measurements into individual spectra
            for i in range(len(start_indexes)):
                spectrum_id = i + 1
                if i == len(start_indexes) - 1:
                    wavelength_nm = data.iloc[start_indexes[i]:, 0].values.round(3)
                    intensity = data.iloc[start_indexes[i]:, 1].values.round(3)
                else:
                    wavelength_nm = data.iloc[start_indexes[i]:start_indexes[i + 1], 0].values.round(3)
                    intensity = data.iloc[start_indexes[i]:start_indexes[i + 1], 1].values.round(3)
                
                metadata = copy.deepcopy(metadata)
                metadata['spectrum_id'] = spectrum_id
                spectrum = Spectrum(wavelength_nm=wavelength_nm,
                                    intensity=intensity,
                                    metadata=metadata,)
                spectrum_objects.append(spectrum)
        return spectrum_objects
    
    def _extract_patient_labels(filename:str, 
                          annotation_all:pd.DataFrame):
        metadata = {}

        patient_id = int(filename.split('_')[1])
        r = int(filename.split('_')[10].split('.')[0])
        theta = int(filename.split('_')[9].split('.')[0])
        sample_type = filename.split('_')[2]

        metadata['patient_id'] = patient_id
        metadata['sample_type'] = sample_type
        metadata['r'] = r
        metadata['theta'] = theta

        if patient_id in annotation_all['OD Number'].values:
            patient_metadata_row = annotation_all[annotation_all['OD Number'] == patient_id]
            metadata['staging'] = patient_metadata_row['Staging'].values[0]
            metadata['gender'] = patient_metadata_row['Gender'].values[0]
            metadata['race'] = patient_metadata_row['Race'].values[0]
        else:
            metadata['staging'] = np.nan
            metadata['gender'] = ''
            metadata['race'] = ''
        return metadata

if __name__ == "__main__":
    data_folder = os.path.join(os.getcwd(), "noodlepy", "data", "bec_hnc", "test")
    annotation_file_path =  os.path.join(os.getcwd(), "noodlepy", "data", "python_test_patient_staging.xlsx")
    preprocessor = SpectrumPreprocessor(cropping=True,
                                        baseline_correction=True,
                                        remove_cosmic_rays= True,
                                        normalization=True,
                                        smoothing=True)
    dataset = PositionalDataset(data_folder, annotation_file_path, preprocessor)
    for i in range(5):
        spectrum, metadata = dataset[i]
        print(f"Spectrum shape: {spectrum.shape}")
        print(f"Metadata: {metadata}")
        print("\n")