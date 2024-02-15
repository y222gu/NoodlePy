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
                 laser_wavelength:float =785,
                 wavelength_nm:np.array =[],
                 intensity:np.array =[]):
        
        self.patient_id:str = patient_id
        self.sample_type:str = sample_type
        self.spectrum_id:str = spectrum_id
        self.laser_wavelength:float = laser_wavelength
        self.wavelength_nm:np.array = wavelength_nm
        self.wavenumber:np.array = create_augment.wavelength_to_wavenumber(self.wavelength_nm).round(3)
        self.intensity:np.array = intensity
        
    def __len__(self):
        return len(self.wavenumber)
    
    def display(self):
        plt.plot(self.wavenumber, self.intensity)
        plt.xlabel('Wavenumber (cm^-1)')
        plt.ylabel('Intensity (a.u.)]')
        plt.title('Spectrum')
        plt.show()

        