import numpy as np
import matplotlib.pyplot as plt
from typing import Any
from scipy import signal
from collections import defaultdict
import scipy.signal
import pybaselines
import os
import copy
import pandas as pd

class Spectrum:
    def __init__(self, 
                 wavelength_nm:np.array =[],
                 intensity:np.array =[],
                 laser_wavelength_nm:float = 785, #TODO: this should be loaded from the experimental metadata
                 metadata:dict = defaultdict(dict),
                 file_path:str = None):

        self.laser_wavelength_nm:float = round(laser_wavelength_nm,3)
        self.wavelength_nm:np.array = wavelength_nm.round(3)
        self.raman_shift_cm:np.array = Spectrum.wavelength_to_raman_shift(self.wavelength_nm, self.laser_wavelength_nm)
        self.intensity:np.array = intensity
        self.metadata = metadata
        self.file_path = file_path
        
    def __len__(self):
        return len(self.intensity)
    
    def crop_spectrum(self, start_raman_shift_cm: float,
                      end_raman_shift_cm: float
                      ):
        """
        Crop the spectrum to the specified raman shift (wavenumber cm^-1) range.

        Parameters:
        start_raman_shift_cm (float): The start of the raman shift (wavenumber cm^-1) range to crop to.
        end_raman_shift_cm (float): The end of the raman shift (wavenumber cm^-1) range to crop to.

        Returns:
        Spectrum: The cropped spectrum obejct.
        """
        spectrum_raman_shift_cm = self.raman_shift_cm

        if start_raman_shift_cm in spectrum_raman_shift_cm and end_raman_shift_cm in spectrum_raman_shift_cm:
            start_index = np.where(spectrum_raman_shift_cm == start_raman_shift_cm)[0][0]
            end_index = np.where(spectrum_raman_shift_cm == end_raman_shift_cm)[0][0]
            self.raman_shift_cm = self.raman_shift_cm[start_index:end_index]
            self.intensity = self.intensity[start_index:end_index]
        else:
            raise ValueError('Wavenumber range could not be aligned to the specified range')
        
        # print(f'Spectrum is cropped between Wavenumber {start_raman_shift_cm} to {end_raman_shift_cm} cm^-1')
        return self
    
    def normalize_spectrum(self,
        normalization_type: str = 'by_max'
        ):
        """
        Normalize the spectra.
        
        Parameters:
        normalization_type (str): The type of normalization to apply. Options are 'by_area' or 'by_max'.
        
        Returns:
        Spectrum: The normalized spectrum object.
        """

        # print(f'Normalizing spectra by: ', normalization_type, '...')
        if normalization_type == 'by_area':
            self.intensity /= np.trapz(self.intensity, self.raman_shift_cm)
        elif normalization_type == 'by_max':
            self.intensity = (self.intensity - np.min(self.intensity))/(np.max(self.intensity)-np.min(self.intensity))
        else:
            raise ValueError(f'Normalization method is not defined')
        return self

    def remove_cosmic_rays(self, 
                kernel_size: int = 2, 
                threshold: float = 3.5
                ):
        """
        Despike the spectrum using WhitakerHayes's modified z-scores filtering.

        Parameters:
        kernel_size (int): The size of the kernel to average to replace the spike. (The spike itself is not included in the average.)
        threshold (float): The modified z_score threshold to use to identify spikes.

        Returns:
        Spectrum: The despike spectrum object.

        References:
        Whitaker, D.A. and Hayes, K., 2018. A simple algorithm for despiking Raman spectra. Chemometrics and Intelligent Laboratory Systems, 179, pp.82-84.
        
        https://towardsdatascience.com/removing-spikes-from-raman-spectra-8a9fdda0ac22
        """
        # print('Despiking spectrum with kernal size:' , kernel_size, 'and threshold:', threshold, '...')

        def modified_z_score(delta_intensity: np.array):
            median_int = np.median(delta_intensity)
            mad_int = np.median([np.abs(delta_intensity - median_int)])
            modified_z_scores = 0.6745 * (delta_intensity - median_int) / mad_int
            return np.array(modified_z_scores)
        
        delta_intensity = np.diff(self.intensity)
        spikes = abs(modified_z_score(delta_intensity)) > threshold

        while any(spike for spike in spikes if spike):
            changes = False

            for i in range(len(spikes)):
                if spikes[i]:
                    neighbours = np.arange(max(0, i - kernel_size),
                                        min(len(self.intensity) - 1, i + 1 + kernel_size))
                    fixed_value = np.mean(self.intensity[neighbours[spikes[neighbours] == 0]])

                    if np.isnan(fixed_value):
                        continue

                    self.intensity[i] = fixed_value
                    spikes[i] = 0
                    changes = True

            if not changes:
                break

    def airPLS(self, lam = 1E3, diff_order=1, max_iter=15, tol=1e-3, weights=None):
        '''
        Baseline removal algorithm.
        Uses an exponential weighting of the negative residuals to attempt to provide a better fit to baseline than the asls method.
        Then remove the baseline.

        Documentation:
        https://pybaselines.readthedocs.io/en/latest/algorithms/whittaker.html#airpls-adaptive-iteratively-reweighted-penalized-least-squares
        '''
        # print(f'Removing baseline using airPLS method with parameters: lam: {lam}, diff_order:{diff_order}, max_iter:{max_iter}, tol:{tol}, weights:{weights}...')
        baseline_fitter = pybaselines.Baseline(x_data=self.raman_shift_cm)
        baseline, _ = baseline_fitter.airpls(self.intensity, lam, diff_order, max_iter, tol, weights) 
        self.intensity = self.intensity - baseline

    
    def savgol_filter(self, window_length=9, polyorder=2):
        # print('Smoothing spectrum using Savitzky-Golay filter with window length:', window_length, 'and polynomial order:', polyorder, '...')
        self.intensity = scipy.signal.savgol_filter(self.intensity, window_length, polyorder)
        

    def add_noise(self, **noise_pars: Any):
        """
        Add noise to the spectrum.

        Parameters:
        **noise_pars: The parameters of the noise to add (arbitrary number of parameters).
        
        Returns:
        Spectrum: The spectrum object with added noise."""
        rng = np.random.default_rng()
        noise_type = noise_pars["noise_type"]

        if noise_type == "poisson":
            self.intensity += 0.001 * rng.poisson(noise_pars["lam"], len(self.intensity))
        elif noise_type == "gaussian":
            self.intensity += rng.normal(noise_pars["mean"], noise_pars["std"], len(self.intensity))
        elif noise_type == "uniform":
            self.intensity += rng.uniform(noise_pars["low"], noise_pars["high"], len(self.intensity))
        elif noise_type == "exponential":
            self.intensity += rng.exponential(noise_pars["mean"], len(self.intensity))
        elif noise_type == "lognormal":
            self.intensity += rng.lognormal(noise_pars["mean"], noise_pars["sigma"], len(self.intensity))
        else:
            raise ValueError("noise_type must be one among 'poisson', 'gaussian', 'uniform', 'expoenetial', and 'lognormal'")
        
        # print(f'Noise type: {noise_type} is added to the spectrum with parameters: {noise_pars}...')
        return self

    def add_cosmic_rays(self, **cosmic_ray_pars: float):
        """
        Add cosmic rays to the spectrum.

        Parameters:
        **cosmic_ray_pars: The parameters of the cosmic rays to add.

        Returns:
        Spectrum: The spectrum object with added cosmic rays.
        """
        spike_locations = np.random.randint(0, len(self.raman_shift_cm), cosmic_ray_pars["spike_number"])

        for spike_location in spike_locations:
            temp_cosmic_ray_intensity = abs(self.intensity[spike_location])* cosmic_ray_pars["spike_amplitude"]

            if temp_cosmic_ray_intensity > max(self.intensity)* 1.2:
                cosmic_ray_intensity = max(self.intensity)* 1.2
            else:
                cosmic_ray_intensity = temp_cosmic_ray_intensity
            self.intensity[spike_location] = cosmic_ray_intensity

        # print(f'{cosmic_ray_pars["spike_number"]} number of cosmic rays are added to the spectrum...')
        return self

    def add_baseline(
        self, 
        **baseline_pars: Any
        ):
        """
        Add baseline to the spectrum.

        Parameters:
        **baseline_pars: The parameters of the baseline to add

        Returns:
        Spectrum: The spectrum object with added baseline.
        """

        baseline_type = baseline_pars["baseline_type"]

        if baseline_type == "poly":
            poly_orders = baseline_pars["poly_orders"]
            poly_coefficients = baseline_pars["poly_coefficients"]

            baseline_intensity = 0
            for order in range(poly_orders + 1):
                baseline_intensity += poly_coefficients[order] * (self.raman_shift_cm ** order)

        elif baseline_type == "sine":  # Sine wave baseline
            sine_amplitude = baseline_pars["sine_amplitude"]
            sine_frequency = baseline_pars["sine_frequency"]
            sine_phase = baseline_pars["sine_phase"]
            
            baseline_intensity = sine_amplitude * np.sin(sine_frequency * self.raman_shift_cm + sine_phase)
        else:
            raise ValueError("Baseline type is not defined")

        self.intensity = self.intensity + baseline_intensity * baseline_pars["baseline_amplifying_factor"]

        # self.intensity = self.intensity + baseline_pars["baseline_offset"]
        
        return self

    def horizontal_shift(self, wavenumber_shift: float):
        """
        Shift the spectrum horizontally.

        Parameters:
        wavenumber_shift (float): The wavenumber to shift.

        Returns:
        Spectrum: The spectrum object horizontally shifted.
        """
        self.raman_shift_cm = self.raman_shift_cm + wavenumber_shift
        return self

    def amplify_spectrum(self, amplifying_pars: dict):
        """
        Amplify the spectrum.

        Parameters:
        spectrum_amplifying_factor (float): The multiplier for the spectrum intensity.

        Returns:
        Spectrum: The amplified spectrum object.
        """
        self.intensity = self.intensity * amplifying_pars["spectrum_amplifying_factor"]
        return self

    def convolve_with_gaussian(self, gaussian_std: float):
        """
        Convolute a Gaussian kernel to the spectrum.

        Parameters:
        gaussian_std (float): The standard deviation of the Gaussian kernel.

        Returns:
        Spectrum: The spectrum object with the Gaussian kernel convoluted.
        """
        kernel = signal.windows.gaussian(len(self.raman_shift_cm), gaussian_std)
        self.intensity = signal.convolve(kernel, self.intensity, mode="same") * sum(kernel)
        return self

    def interpolate_spectrum(self, target_raman_shift: np.array):
        """
        Interpolate the spectrum to the target raman shift range.

        Parameters:
        target_raman_shift (np.array): The target raman shift range to interpolate to.

        Returns:
        Spectrum: The interpolated spectrum object.
        """
        interpolated_intensity = np.interp(target_raman_shift, self.raman_shift_cm, self.intensity)
        self.raman_shift_cm= target_raman_shift
        self.intensity = interpolated_intensity
        return self

    def display(self, filename:str = 'Spectrum'):
        """
        Display the spectrum and save the plot to default path (current working directory).

        Parameters:
        path (str): The path to save the plot.
        """
        # path = './' + filename + '.png'
        current_directory = os.getcwd()
        folder = os.path.join(current_directory, "output_plots")

        if not os.path.exists(folder):
            os.makedirs(folder)
        
        # increment the filename if it already exists
        new_filename = filename

        i = 0
        while os.path.exists(os.path.join(folder, new_filename + '.png')):
            i += 1
            new_filename = filename + f'_{i}'

        path = os.path.join(folder, new_filename + '.png')

        f, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.plot(self.raman_shift_cm, self.intensity, color='w')
        ax.set_xlabel('Raman Shift (cm^-1)', fontsize=18, color='w')
        ax.set_ylabel('Intensity (a.u.)]', fontsize=18, color='w')
        # ax.set_title(new_filename, fontsize=18, color='w')
        ax.tick_params(axis='x', colors='w')
        ax.tick_params(axis='y', colors='w')
        ax.spines['bottom'].set_color('w')
        ax.spines['top'].set_color('w')
        ax.spines['right'].set_color('w')
        ax.spines['left'].set_color('w')
        ax.yaxis.label.set_color('w')
        ax.xaxis.label.set_color('w')
        f.savefig(path, bbox_inches='tight', dpi=300, transparent=True)
        plt.close()


    @staticmethod
    def wavelength_to_wavenumber(wavelength_nm:np.array) -> np.array:
        """
        Convert wavelength to wavenumber.

        Parameters:
        wavelength_nm (np.array): The wavelength in nm.

        Returns:
        np.array: The wavenumber in cm^-1.
        """
        wavenumber_cm = np.power(wavelength_nm * (1e-7), -1)
        wavenumber_cm = wavenumber_cm.round(3)
        return wavenumber_cm

    def wavelength_to_raman_shift(wavelength_nm:np.array,laser_wavelength_nm:np.array) -> np.array:
        """
        Convert spectrum from wavelength to Raman shift.

        Parameters:
        wavelength_nm (np.array): The wavelength of a spectrum in nm.
        laser_wavelength_nm (np.array): The laser wavelength in nm.

        Returns:
        np.array: The Raman shift of the spectrum in cm^-1.
        """
        # first convent from nm to cm
        wavenumber_cm = Spectrum.wavelength_to_wavenumber(wavelength_nm)
        laser_wavenumber_cm = Spectrum.wavelength_to_wavenumber(laser_wavelength_nm)
        raman_shift_cm = laser_wavenumber_cm - wavenumber_cm
        raman_shift_cm = raman_shift_cm.round(3)
        return raman_shift_cm
    
    def count_number_of_cosmic_rays(self, threshold: float = 3.5):
        """
        Count the number of cosmic rays in the spectrum.

        Parameters:
        threshold (float): The threshold to identify cosmic rays.

        Returns:
        int: The number of cosmic rays in the spectrum.
        """

        def modified_z_score(delta_intensity: np.array):
            median_int = np.median(delta_intensity)
            mad_int = np.median([np.abs(delta_intensity - median_int)])
            modified_z_scores = 0.6745 * (delta_intensity - median_int) / mad_int
            return np.array(modified_z_scores)

        delta_intensity = np.diff(self.intensity)
        spikes = abs(modified_z_score(delta_intensity)) > threshold
        return sum(spikes)


    @classmethod
    def load_from_file(cls, file_path:str):
        # load Spectrum objects from a file
        with open(file_path) as f:
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
                
            
                metadata = {}
                metadata['spectrum_id'] = spectrum_id
                spectrum = Spectrum(wavelength_nm=wavelength_nm,
                                    intensity=intensity,
                                    metadata=metadata,)
                spectrum_objects.append(spectrum)
        return spectrum_objects
    
    def update_metadata(cls, metadata:dict):
        cls.metadata.update(metadata)
        return cls

