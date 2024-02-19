# define spectrum class
import pandas as pd
import numpy as np
from sympy import plot
import noodlepy.utils.create_augment as create_augment
import matplotlib.pyplot as plt

class Spectrum:
    def __init__(self, 
                 patient_id:str ='',
                 sample_type:str ='',
                 spectrum_id:str ='', 
                 laser_wavelength_nm:float =785,
                 wavelength_nm:np.array =[],
                 intensity:np.array =[]):

        self.patient_id:str = patient_id
        self.sample_type:str = sample_type
        self.spectrum_id:str = spectrum_id
        self.laser_wavelength_nm:float = round(laser_wavelength_nm,3)
        self.wavelength_nm:np.array = wavelength_nm.round(3)
        self.raman_shift_cm:np.array = create_augment.wavelength_to_raman_shift(self.wavelength_nm, self.laser_wavelength_nm)
        self.intensity:np.array = intensity
        
    def __len__(self):
        return len(self.raman_shift_cm)
    
    #
    def display(self, path:str = './plot.png'):
        f, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.plot(self.raman_shift_cm, self.intensity)
        ax.set_xlabel('Wavenumber (cm^-1)')
        ax.set_ylabel('Intensity (a.u.)]')
        ax.set_title('Spectrum')
        f.savefig(path, bbox_inches='tight', dpi=300)
        plt.close()

        