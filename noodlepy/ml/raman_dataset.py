# https://pytorch.org/get-started/locally/
from torch.utils.data import Dataset
import pandas as pd
import noodlepy.utils.create_augment as create_augment
import os
from noodlepy.utils.spectrum_class import Spectrum
import torch
import numpy as np

class RamanDataset(Dataset):
    def __init__(self, data_folder = None,
                 metadata_file_path = None,
                 preprocessing_flag: bool = True,
                 augmentation_step_option_list: list[str]= ['baseline','shot_noise','dark_current_noise','photo_response_non_uniformity','cosmic_ray']
                 ):
        """
        Load the database of spectra from the txt file 

        Args:
        data_folder (str): The folder where the spectra are stored

        Returns:
        list[Spectrum]: A list of Spectrum objects
        """
        self.preprocessing_flag = preprocessing_flag
        self.augmentation_step_option_list = augmentation_step_option_list

        # FIXME: This is a temporary function to create a list of spectrum objects on the fly from the text files
        # Once we have a database, we will need to change this to a database query
        list_of_spectrum_objects = []
        list_of_file_names = sorted([f for f in os.listdir(data_folder) if f.endswith(('.txt'))])

        # load the metadata file
        metadata_all = pd.read_excel(metadata_file_path)

        for filename in list_of_file_names:
            metadata_of_corresponding_file = RamanDataset._extract_metadata(filename, metadata_all)
            spectrum_objects = RamanDataset._make_spectrum_objects(data_folder,filename=filename, metadata=metadata_of_corresponding_file)
            list_of_spectrum_objects+=spectrum_objects

        self.db = list_of_spectrum_objects

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

        if self.preprocessing_flag == True:
            preorocessed_spectrum = create_augment.pre_process(chosen_spectrum)

        augmented_spectrum_1,augmented_spectrum_2 = create_augment.apply_augmentations(preorocessed_spectrum, self.augmentation_step_option_list, 2) # REQ: Only need 2 children of the chosen_spectrum
        augmented_spectrum_intensity_1 = torch.tensor(augmented_spectrum_1.intensity, dtype=torch.float32).unsqueeze(0)
        augmented_spectrum_intensity_2 = torch.tensor(augmented_spectrum_2.intensity, dtype=torch.float32).unsqueeze(0)

        return augmented_spectrum_intensity_1, augmented_spectrum_intensity_2, chosen_spectrum.staging
    
    def _extract_metadata(spectrum_file_name:str, 
                          metadata_all:pd.DataFrame):

        # Patient metatdata extraction
        metadata_of_corresponding_file = {}

        f_split = spectrum_file_name.split('_')
        patient_id = int(f_split[0])
        sample_type = f_split[1]
        metadata_of_corresponding_file['patient_id'] = patient_id
        metadata_of_corresponding_file['sample_type'] = sample_type

        # Extract the metadata for the given patient_id
        if patient_id in metadata_all['OD Number'].values:
            patient_metadata_row = metadata_all[metadata_all['OD Number'] == patient_id]
            metadata_of_corresponding_file['staging'] = patient_metadata_row['Staging'].values[0]
            metadata_of_corresponding_file['age'] = patient_metadata_row['Age'].values[0]
            metadata_of_corresponding_file['gender'] = patient_metadata_row['Gender'].values[0]
            metadata_of_corresponding_file['race'] = patient_metadata_row['Race/Ethnicity'].values[0]
            metadata_of_corresponding_file['bmi'] = patient_metadata_row['BMI'].values[0]
        else:
            # If the patient_id is not found in the metadata file, set the every metadata to empty string and number
            metadata_of_corresponding_file['staging'] = ''
            metadata_of_corresponding_file['age'] = []
            metadata_of_corresponding_file['gender'] = ''
            metadata_of_corresponding_file['race'] = ''
            metadata_of_corresponding_file['bmi'] = []
        
            print(f"Patient ID {patient_id} not found in the metadata file")
            print("Metadata set to empty string and number")

        return metadata_of_corresponding_file
    
    def _make_spectrum_objects(data_folder:str,
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

                spectrum = Spectrum(patient_id=metadata['patient_id'], 
                                    sample_type=metadata['sample_type'], 
                                    spectrum_id=spectrum_id, 
                                    wavelength_nm=wavelength_nm,
                                    intensity=intensity,
                                    staging=metadata['staging'],
                                    age=metadata['age'],
                                    gender=metadata['gender'],
                                    race=metadata['race'],
                                    bim=metadata['bmi'])
                spectrum_objects.append(spectrum)
        return spectrum_objects

if __name__ == "__main__":
    data_folder = os.path.join(os.getcwd(), "noodlepy", "data", "Raman_DB", "all")
    metadata_file = os.path.join(os.getcwd(), "noodlepy", "data", "Biofluid_list_annotated_v4.xlsx")

    dataset = RamanDataset(data_folder, metadata_file)

    for i in range(50):
        augmented_spectrum1,augmented_spectrum2, label_staging = dataset.__getitem__(idx= i)