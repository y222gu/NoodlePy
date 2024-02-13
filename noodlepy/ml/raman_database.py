# https://pytorch.org/get-started/locally/
import dis
from numpy import number
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

    def load_db(self, data_folder='/Users/yifeigu/Documents/Carney_Lab/NoodlePy/noodlepy/data/Raman_DB'):
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
    
    def __getitem__(self, idx):
        chosen_spectrum = self.db[idx]
        augmented_spectrum1, augmented_spectrum2 = create_augment.apply_augmentations(chosen_spectrum,number_of_augmentation=2)
        return augmented_spectrum1, augmented_spectrum2
    
if __name__ == "__main__":
    dataset = SyntheticRamanFileDataset()
    dataset.load_db()
    augmented_spectrum1, augmented_spectrum2 = dataset.__getitem__(1)
    augmented_spectrum1.display()
    augmented_spectrum2.display()