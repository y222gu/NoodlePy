# https://pytorch.org/get-started/locally/
from re import S, split
import scipy as sp
from torch.utils.data import Dataset
import pandas as pd
import torch
import re
import numpy as np
import noodlepy.utils.create_augment as create_augment
import os
import yaml
from noodlepy.utils.spectrum_class import Spectrum

class SyntheticRamanFileDataset(Dataset):
    def __init__(self):
        """
            Assumes that your spectra are saved as files
            Advantages:  You can distribute your training data to others
            Disadvantages:  You have to keep track of your training sets
        """
        pass

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

    def __getitem__(self, idx):
        chosen_spectrum = self.db[idx]

        # TODO: call noodlepy to create two random augmentation dictionaries on the fly
        with open('/Users/yifeigu/Documents/Carney_Lab/NoodlePy/noodlepy/config/config_test.yml', "rb") as yaml_file:
            config = yaml.safe_load(yaml_file)

        augmentation_pipeline = create_augment.create_augmentation_pipeline(config)
        augumentation_par_sets = create_augment.create_augumentation_par_sets(config,2)
        
        
        # TODO: random augmentation
        transform1, transform2 = create_augment.create_transform()

        # REQ: no more than two augmentations for selfsupervised learning
        augmented_spectrum1 = self.transform(original_spectrum)
        augmented_spectrum2 = self.transform(original_spectrum)

        return augmented_spectrum1, augmented_spectrum2
    



dataset = SyntheticRamanFileDataset(csv_file_path='/Users/yifeigu/Documents/Carney_Lab/NoodlePy/noodlepy/data/Raman_DB.csv')





















'''
class SyntheticRamanOnTheFlyDataset(Dataset):
    def __init__(self, number_of_spectra):
        """
            Creates the spectra on call
            Advantage:  No additional creation of files
            Disadvantage:   You need to make sure that your random numpy and torch seeds are 
                            properly intilizalized or you will get a different answer every time

                            see https://numpy.org/doc/stable/reference/random/generator.html
                            see https://pytorch.org/docs/stable/notes/randomness.html
                            see https://lightning.ai/docs/pytorch/stable/common/trainer.html#reproducibility
        """
        # TODO: Initialize random spectra parameters e.g. valid ranges for each metabolite self.dictionary = {'metabolite 1':[low,high]}        
        #                                            e.g. open a config file with ranges for each metabolite
        self.number_of_spectra = number_of_spectra

    def __len__(self):
        return self.number_of_spectra

    def __getitem__(self, idx):
        # TODO: Draw a random set of metabolite compositions out of the ranges
        # TODO: Turn them into a speectrum
        # TODO: Create two random augmentation dictionaries
        # TODO: apply the augmentation to the spectrum        
        return augmentation1, augmentation2
    
    '''