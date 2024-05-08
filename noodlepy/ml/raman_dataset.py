# https://pytorch.org/get-started/locally/
from torch.utils.data import Dataset
import pandas as pd
import os
from noodlepy.utils.class_Spectrum import Spectrum
from noodlepy.utils.class_SpectrumPreprocessor import SpectrumPreprocessor
from noodlepy.utils.class_SpectrumAugmentor import SpectrumAugmentor
import torch
import copy
import numpy as np

class RamanDataset(Dataset):
    def __init__(self, data_folder = None,
                 annotation_file_path = None,
                 preprocessor = None,
                 augmentor = None):
        """
        Load the database of spectra from the txt file 

        Args:
        data_folder (str): The folder where the spectra are stored

        Returns:
        list[Spectrum]: A list of Spectrum objects
        """
        print("Loading the Raman dataset")
        self.preprocessor = preprocessor
        self.augmentor = augmentor

        list_of_spectrum_objects = []
        list_of_file_names = sorted([f for f in os.listdir(data_folder) if f.endswith(('.txt'))])

        annotation_all = pd.read_excel(annotation_file_path)

        for filename in list_of_file_names:
            patient_annotations = RamanDataset._extract_patient_labels(filename, annotation_all)
            spectrum_objects = RamanDataset._load_files_to_spectrum_objects(data_folder, filename, patient_annotations)
            list_of_spectrum_objects+=spectrum_objects

        self.db = list_of_spectrum_objects
        print(f"Loaded {len(self.db)} spectra")

    def __len__(self):
        return len(self.db)
    
    def __getitem__(self, 
                    idx:int
                    )-> tuple[Spectrum, Spectrum]:
        """
        Return 2 augmented spectra from the chosen spectrum

        Args:
        idx (int): The index of the spectrum to augment
        preprocessing_flag (bool): Whether to apply preprocessing to the chosen_spectrum
        augmentation_step_option_list (list[str]): The list of augmentation steps to choose and apply randomly

        returns:
        augmented_spectrum_list (list[Spectrum]): a tuple of 2 augmented Spectrum objects
        """
        chosen_spectrum:Spectrum = self.db[idx]

        if self.preprocessor is not None:
            preprocessor = self.preprocessor
            preprocessed_spectrum = preprocessor.pre_process(chosen_spectrum)
        else:
            preprocessed_spectrum = chosen_spectrum

        if self.augmentor is not None:
            augmentor = self.augmentor
            augmented_spectrum_1,augmented_spectrum_2 = augmentor.augment(preprocessed_spectrum, 2) # REQ: Only need 2 children of the chosen_spectrum
        else:
            augmented_spectrum_1 = preprocessed_spectrum
            augmented_spectrum_2 = preprocessed_spectrum

        augmented_spectrum_intensity_1 = torch.tensor(augmented_spectrum_1.intensity, dtype=torch.float32).unsqueeze(0)
        augmented_spectrum_intensity_2 = torch.tensor(augmented_spectrum_2.intensity, dtype=torch.float32).unsqueeze(0)

        return augmented_spectrum_intensity_1, augmented_spectrum_intensity_2, chosen_spectrum.metadata
    
    def _extract_patient_labels(spectrum_file_name:str, 
                          all_patient_labels:pd.DataFrame):

        # Patient metatdata extraction
        patient_labels = {}
        f_split = spectrum_file_name.split('_')
        patient_id = int(f_split[0])
        sample_type = f_split[1]
        patient_labels['patient_id'] = patient_id
        patient_labels['sample_type'] = sample_type
        # Extract the metadata for the given patient_id
        if patient_id in all_patient_labels['OD Number'].values:
            patient_metadata_row = all_patient_labels[all_patient_labels['OD Number'] == patient_id]

            if len(patient_metadata_row) > 1:
                patient_metadata_row = patient_metadata_row.iloc[[0]]
                print(f"Patient ID {patient_id} has multiple entries in the metadata file")
                print("Only the first entry will be used")
                
            patient_labels['staging'] = patient_metadata_row['Staging'].values[0]
            patient_labels['gender'] = patient_metadata_row['Gender'].values[0]
            patient_labels['race'] = patient_metadata_row['Race'].values[0]
            #patient_labels['bmi'] = patient_metadata_row['BMI'].values[0]
            #patient_labels['age'] = patient_metadata_row['Age'].values[0]

        else:
            # If the patient_id is not found in the metadata file, set the every metadata to empty string and number
            patient_labels['staging'] = np.nan
            patient_labels['gender'] = ''
            patient_labels['race'] = ''
            #patient_labels['age'] = np.nan
            #patient_labels['bmi'] = np.nan
        
            print(f"Patient ID {patient_id} not found in the metadata file")
            print("Metadata set to empty strings and numbers")
        return patient_labels
    
    def _load_files_to_spectrum_objects(data_folder:str,
                               filename:str, 
                               patient_annotations:dict):
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
                
                metadata = copy.deepcopy(patient_annotations)
                metadata['spectrum_id'] = spectrum_id
                spectrum = Spectrum(wavelength_nm=wavelength_nm,
                                    intensity=intensity,
                                    metadata=metadata,)
                spectrum_objects.append(spectrum)
        return spectrum_objects

if __name__ == "__main__":
    data_folder = os.path.join(os.getcwd(), "noodlepy", "data", "Raman_DB", "plasma_saliva_mixed","all")
    metadata_file = os.path.join(os.getcwd(), "noodlepy", "data", "Biofluid_list_annotated_v4.xlsx")

    preprocessor = SpectrumPreprocessor(cropping=True,
                                        baseline_correction=False,
                                        despike= False,
                                        normalization=False,
                                        smoothing=False)
    
    augmentor = SpectrumAugmentor(ramdom_augmentations=True,
                                  augmentation_step_list = None,
                                  config_path= None)

    dataset = RamanDataset(data_folder, metadata_file, preprocessor, augmentor)

    for i in range(10):
        augmented_spectrum1,augmented_spectrum2, labels = dataset.__getitem__(idx= i)
        print(augmented_spectrum1.shape)
        print(augmented_spectrum2.shape)
        print(labels)