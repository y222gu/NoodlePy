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


class OC_Dataset(Dataset):
    def __init__(self, data_folder = None,
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
        list_of_file_paths = []
        for root, dirs, files in os.walk(data_folder):
            for file in files:
                if file.endswith('.txt'):
                    list_of_file_paths.append(os.path.join(root, file))

        for file_path in list_of_file_paths:
            filename = os.path.basename(file_path)
            patient_annotations = OC_Dataset._extract_patient_labels(filename)
            spectrum_object = OC_Dataset._load_files_to_spectrum_objects(file_path, patient_annotations)
            list_of_spectrum_objects += spectrum_object

        self.db = list_of_spectrum_objects
        print(f"Loaded {len(self.db)} spectra")

    def __len__(self):
        return len(self.db)
    
    def __getitem__(self, 
                    idx:int
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
        chosen_spectrum:Spectrum = self.db[idx]
        
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
        raman_shift_tensor = torch.tensor(augmented_spectrum_1.raman_shift_cm, dtype=torch.float32).unsqueeze(0)

        return intensity_tensor, raman_shift_tensor, chosen_spectrum.metadata

    def _extract_patient_labels(spectrum_file_name:str):

        # Patient metatdata extraction
        patient_labels = {}
        f_split = spectrum_file_name.split('_')
        patient_id = f_split[1]
        staging = f_split[2]

        patient_labels['patient_id'] = patient_id
        patient_labels['staging'] = staging

        print(patient_labels)
        
        return patient_labels
    
    def _load_files_to_spectrum_objects(file_path: str,
                               patient_annotations:dict):
        with open(file_path) as f:
            data = pd.read_csv(f, sep=",", header=None) # sep="\t"

            repeated_wavelengths = data.iloc[:,0].value_counts()
            first_repeated_wavelength = repeated_wavelengths.idxmax()
            start_indexes = data[data.iloc[:,0] == first_repeated_wavelength].index.tolist()

            spectrum_objects = []
            # split the repeated measurements into individual spectra
            intensities = []
            wavelengths = None
            for i in range(len(start_indexes)):
                if i == len(start_indexes) - 1:
                    cur_wavelength = data.iloc[start_indexes[i]:, 0].values.round(3)
                    cur_intensity = data.iloc[start_indexes[i]:, 1].values.round(3)
                else:
                    cur_wavelength = data.iloc[start_indexes[i]:start_indexes[i + 1], 0].values.round(3)
                    cur_intensity = data.iloc[start_indexes[i]:start_indexes[i + 1], 1].values.round(3)
                spectrum_list = []
                for i in range(len(start_indexes)):
                    if i == len(start_indexes) - 1:
                        cur_wavelength = data.iloc[start_indexes[i]:, 0].values.round(3)
                        cur_intensity = data.iloc[start_indexes[i]:, 1].values.round(3)
                    else:
                        cur_wavelength = data.iloc[start_indexes[i]:start_indexes[i + 1], 0].values.round(3)
                        cur_intensity = data.iloc[start_indexes[i]:start_indexes[i + 1], 1].values.round(3)
                    
                    metadata = copy.deepcopy(patient_annotations)
                    metadata['spectrum_id'] = i + 1
                    spectrum_obj = Spectrum(wavelength_nm=cur_wavelength,
                                            intensity=cur_intensity,
                                            metadata=metadata)
                    spectrum_list.append(spectrum_obj)
                
                return spectrum_list

if __name__ == "__main__":
    data_folder = r'C:\Users\Yifei\Downloads\data_izabella_filtered\data_izabella_filtered'

    preprocessor = SpectrumPreprocessor(cropping=False,
                                        baseline_correction=False,
                                        remove_cosmic_rays=False,
                                        normalization=True,
                                        smoothing=False)

    # preprocessor = None

    augmentor = None

    dataset = OC_Dataset(data_folder, preprocessor, augmentor)

    print(len(dataset))

    fig, (ax_cancer, ax_control) = plt.subplots(1, 2, figsize=(12, 6))
    
    for i in range(len(dataset)):
        preprocessed_spectrum, labels = dataset.__getitem__(i)
        x_axis = range(len(preprocessed_spectrum))
        
        # Determine which subplot to use based on patient staging (case insensitive)
        stage = labels.get('staging', '').lower()
        if stage == "cancer":
            ax_cancer.plot(x_axis, preprocessed_spectrum, color="red", alpha=0.7,
                           label=f"Index {i} (staging: {labels['staging']})" if i == 0 else "")
        elif stage == "control":
            ax_control.plot(x_axis, preprocessed_spectrum, color="blue", alpha=0.7,
                            label=f"Index {i} (staging: {labels['staging']})" if i == 0 else "")
    
    # Set axis labels, titles, and limits for cancer subplot
    ax_cancer.set_title("Cancer")
    ax_cancer.set_xlabel("Measurement Index")
    ax_cancer.set_ylabel("Intensity")
    ax_cancer.set_ylim(-0.3, 1.3)
    ax_cancer.legend(loc="upper right", fontsize='small')
    
    # Set axis labels, titles, and limits for control subplot
    ax_control.set_title("Control")
    ax_control.set_xlabel("Measurement Index")
    ax_control.set_ylabel("Intensity")
    ax_control.set_ylim(-0.3, 1.3)
    ax_control.legend(loc="upper right", fontsize='small')

    plt.title("Overlay of Preprocessed Spectra (Red: Cancer, Blue: Control)")
    plt.xlabel("Measurement Index")
    plt.ylabel("Intensity")
    plt.ylim(-0.3, 1.3)
    plt.legend(loc="upper right", fontsize='small')
    plt.show()