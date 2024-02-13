# define spectrum class
import pandas as pd
import numpy as np
from sympy import plot
import noodlepy.utils.create_augment as create_augment

class Spectrum:
    def __init__(self, 
                 patient_id:str ='', 
                 sample_type:str ='',
                 spectrum_id:str ='', 
                 laser_wavelength:float =785,
                 wavelength_nm:np.array =[],
                 intensity:np.array =[]):
        
        self.patient_id:str = patient_id
        self.sample_type:str = sample_type
        self.spectrum_id:str = spectrum_id
        self.laser_wavelength:float = laser_wavelength
        self.wavelength_nm:np.array = wavelength_nm
        self.wavenumber:np.array = create_augment.wavelength_to_wavenumber(self.wavelength_nm)
        self.intensity:np.array = intensity
        
    def __len__(self):
        return len(self.wavenumber)
    
    def augment(self, augment_type:str = 'none',augment_pars:dict = {}):
        
        # TODO: This needs to be fixed
        augmentation_par_dictionaries = create_augment.augmention_pars_generator(2)
        
        if augment_type == 'none':
            pass
        elif augment_type == 'random_noise':
            self.intensity = create_augment.random_noise(self,)
        elif augment_type == 'random_shift':
            self.intensity = create_augment.random_shift(self.intensity)
        elif augment_type == 'random_scale':
            self.intensity = create_augment.random_scale(self.intensity)
        elif augment_type == 'random_shift_scale':
            self.intensity = create_augment.random_shift_scale(self.intensity)
        else:
            raise ValueError('Invalid augment_type')
    
    def display(self):
        plot(self.wavenumber, self.intensity, title='Spectrum', xlabel='Wavenumber (cm^-1)', ylabel='Intensity')