# https://pytorch.org/get-started/locally/
from calendar import c
import dis
from numpy import number
from sympy import plot
from torch.utils.data import Dataset
import pandas as pd
import noodlepy.utils.create_augment as create_augment
import os
from noodlepy.utils.spectrum_class import Spectrum

class SyntheticRamanFileDataset(Dataset):
    def __init__(self):
        """
            Assumes that your spectra are saved as files
            Advantages:  You can distribute your training data to others
            Disadvantages:  You have to keep track of your training sets
        """
        # FIXME
        # self.augmentation_flag = str
        # self.preprocess_flag(apply same preprocessing for every spectrum)
        self.db: list[Spectrum]


    def load_db(self, data_folder='/Users/yifeigu/Documents/Carney_Lab/NoodlePy/noodlepy/data/Raman_DB'):
        """
        Load the database of spectra from the txt file

        Args:
        data_folder (str): The folder where the spectra are stored

        Returns:
        list[Spectrum]: A list of Spectrum objects
        """

        # FIXME: This is a temporary function to create a list of spectrum objects on the fly from the csv file
        # Once we have a database, we will need to change this to a database query
        list_of_spectrum_objects = []
        laser_wavelength= 785
        list_of_file_names = sorted([f for f in os.listdir(data_folder) if f.endswith(('.txt'))])

        for filename in list_of_file_names:
            f_split = filename.split('_')
            patient_id = f_split[0]
            sample_type = f_split[1]

            with open(os.path.join(data_folder, filename)) as f:
                data = pd.read_csv(f, sep=",", header=None)

            repeated_wavelengths = data.iloc[:,0].value_counts()
            first_repeated_wavelength = repeated_wavelengths.idxmax()
            start_indexes = data[data.iloc[:,0] == first_repeated_wavelength].index.tolist()

            # split the repeated measurements into individual spectra
            for i in range(len(start_indexes)):
                spectrum_id = i + 1
                if i == len(start_indexes) - 1:
                    wavelength_nm = data.iloc[start_indexes[i]:, 0].values.round(3)
                    intensity = data.iloc[start_indexes[i]:, 1].values.round(3)
                else:
                    wavelength_nm = data.iloc[start_indexes[i]:start_indexes[i + 1], 0].values.round(3)
                    intensity = data.iloc[start_indexes[i]:start_indexes[i + 1], 1].values.round(3)
                list_of_spectrum_objects.append(Spectrum(patient_id, sample_type, spectrum_id, laser_wavelength, wavelength_nm, intensity))

        self.db = list_of_spectrum_objects
        return self.db

    def __len__(self):
        return len(self.db)
    
    def __getitem__(self, 
                    idx:int, 
                    augmentation_step_option_list: list[str])-> tuple[Spectrum, Spectrum]:
        """
        Return 2 augmented spectra from the chosen spectrum

        Args:
        idx (int): The index of the spectrum to augment
        augmentation_step_option_list (list[str]): The list of augmentation steps to choose and apply randomly

        returns:
        augmented_spectrum_list (list[Spectrum]): a tuple of 2 augmented Spectrum objects
        """
        
        chosen_spectrum:Spectrum = self.db[idx]
        # REQ: Only need 2 children of the chosen_spectrum
        augmented_spectrum_list = Spectrum.apply_augmentations(chosen_spectrum, augmentation_step_option_list, 2)

        chosen_spectrum.display()
        augmented_spectrum_list[0].display()
        augmented_spectrum_list[1].display()
        return augmented_spectrum_list[0],augmented_spectrum_list[1]

if __name__ == "__main__":
    dataset = SyntheticRamanFileDataset()
    dataset.load_db()
    augmented_spectrum1,augmented_spectrum2 = dataset.__getitem__(idx= 1, augmentation_step_option_list = ['baseline','shot_noise','dark_current_noise',
                                                                                                       'photo_response_non_uniformity','cosmic_ray'])

