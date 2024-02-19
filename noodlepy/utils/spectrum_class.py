# define spectrum class
import numpy as np
import matplotlib.pyplot as plt
from typing import Any
from scipy import signal
import copy
import yaml
from collections import defaultdict

class Spectrum:
    def __init__(self, 
                 patient_id:str ='',
                 sample_type:str ='',
                 spectrum_id:str ='', 
                 laser_wavelength_nm:float = [],
                 wavelength_nm:np.array =[],
                 intensity:np.array =[]):

        self.patient_id:str = patient_id
        self.sample_type:str = sample_type
        self.spectrum_id:str = spectrum_id
        self.laser_wavelength_nm:float = round(laser_wavelength_nm,3)
        self.wavelength_nm:np.array = wavelength_nm.round(3)
        self.raman_shift_cm:np.array = Spectrum.wavelength_to_raman_shift(self.wavelength_nm, self.laser_wavelength_nm)
        self.intensity:np.array = intensity
        
    def __len__(self):
        return len(self.raman_shift_cm)
    
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
        
        print(f'Spectrum is aligned and cropped between Wavenumber {start_raman_shift_cm} cm^-1 to {end_raman_shift_cm} cm^-1')
        return self
    
    def normalize_spectra(self,
        normalization_type: str
        ):
        """
        Normalize the spectra.
        
        Parameters:
        normalization_type (str): The type of normalization to apply. Options are 'by_area' or 'by_max'.
        
        Returns:
        Spectrum: The normalized spectrum object.
        """

        print(f'Normalizing spectra by: ', normalization_type, '...')
        if normalization_type == 'by_area':
            self.intensity /= np.trapz(self.intensity, self.raman_shift_cm)
        elif normalization_type == 'by_max':
            self.intensity /= np.max(self.intensity)
        else:
            raise ValueError(f'Normalization method is not defined')
        return self

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
        return self

    def add_cosmic_rays(self, **cosmic_ray_pars: Any):
        """
        Add cosmic rays to the spectrum.

        Parameters:
        **cosmic_ray_pars: The parameters of the cosmic rays to add.

        Returns:
        Spectrum: The spectrum object with added cosmic rays.
        """
        spike_locations = np.random.randint(0, len(self.raman_shift_cm), cosmic_ray_pars["spike_number"])

        for spike_location in spike_locations:
            self.intensity[spike_location] = abs(self.intensity[spike_location])* cosmic_ray_pars["spike_amplitude"]
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

    def amplify_spectrum(self, spectrum_amplifying_factor: float):
        """
        Amplify the spectrum.

        Parameters:
        spectrum_amplifying_factor (float): The multiplier for the spectrum intensity.

        Returns:
        Spectrum: The amplified spectrum object.
        """
        self.intensity = self.intensity * spectrum_amplifying_factor
        return self

    def convolute_gaussian_to_spectrum(self, gaussian_std: float):
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

    def display(self, path:str = './plot.png'):
        """
        Display the spectrum and save the plot to default path (current working directory).

        Parameters:
        path (str): The path to save the plot.
        """
        f, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.plot(self.raman_shift_cm, self.intensity)
        ax.set_xlabel('Wavenumber (cm^-1)')
        ax.set_ylabel('Intensity (a.u.)]')
        ax.set_title('Spectrum')
        f.savefig(path, bbox_inches='tight', dpi=300)
        plt.close()

    def apply_augmentations(self, augmentation_step_option_list: Any, number_dictionary: int):
        """
        Apply a set of augmentations to a spectrum.
        
        Parameters:
        spectrum (Spectrum): The spectrum to be augmented.
        number_of_augmentation (int): The number of augmentations to be applied.
            
        Returns:
        Spectrum: List of augmented spectrum objects.
        """
        augmented_spectrum_list = []
        for i, i_augmentation_dictionary in Spectrum.augmentation_par_dictionary_generator(augmentation_step_option_list, number_dictionary).items():
            augmented_spectrum = copy.deepcopy(self)

            if 'normalization' in i_augmentation_dictionary.keys():
                augmented_spectrum = Spectrum.normalize_spectra(augmented_spectrum, i_augmentation_dictionary["normalization"])

            if 'spectrum_amplifying_factor' in i_augmentation_dictionary.keys():
                augmented_spectrum = Spectrum.amplify_spectrum(augmented_spectrum, i_augmentation_dictionary["amplification"])

            if 'horizontal_shift' in i_augmentation_dictionary.keys():
                augmented_spectrum = Spectrum.horizontal_shift(augmented_spectrum, i_augmentation_dictionary["horizontal_shift"])

            if 'baseline' in i_augmentation_dictionary.keys():
                augmented_spectrum = Spectrum.add_baseline(augmented_spectrum, **i_augmentation_dictionary["baseline"])

            if 'convoluting_gaussian' in i_augmentation_dictionary.keys():
                augmented_spectrum = Spectrum.convolute_gaussian_to_spectrum(augmented_spectrum, i_augmentation_dictionary["convoluting_gaussian"])

            if 'shot_noise' in i_augmentation_dictionary.keys():
                augmented_spectrum = Spectrum.add_noise(augmented_spectrum, **i_augmentation_dictionary['shot_noise'])

            if 'dark_current_noise' in i_augmentation_dictionary.keys():
                augmented_spectrum = Spectrum.add_noise(augmented_spectrum, **i_augmentation_dictionary['dark_current_noise'])

            if 'photo_response_non_uniformity' in i_augmentation_dictionary.keys():
                augmented_spectrum = Spectrum.add_noise(augmented_spectrum, **i_augmentation_dictionary['photo_response_non_uniformity'])

            if 'FPN_noise' in i_augmentation_dictionary.keys():
                augmented_spectrum = Spectrum.add_noise(augmented_spectrum, **i_augmentation_dictionary['FPN_noise'])

            if 'cosmic_ray' in i_augmentation_dictionary.keys():
                augmented_spectrum = Spectrum.add_cosmic_rays(augmented_spectrum, **i_augmentation_dictionary['cosmic_ray'])

            augmented_spectrum_list.append(augmented_spectrum)

        return augmented_spectrum_list

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

    def random_augmentation_steps_generator(augmentation_step_option_list: list) -> list:
        """
        Choose random number of random step from augmentation_step_option_list

        Returns:
        list: The list of random augmentation steps.
        """
        number_of_augmentation_steps = np.random.randint(1,len(augmentation_step_option_list))
        random_augmentation_steps = np.random.choice((augmentation_step_option_list),number_of_augmentation_steps,replace=False)
        return random_augmentation_steps


    def augmentation_par_dictionary_generator(augmentation_step_option_list: list,
        number_dictionary: int
    ) -> dict:
        """
        Generate dictionaries with random parameters for random augmentation step chosen from augmentation_step_option_list.

        Parameters:
        augmentation_step_option_list (list): The list of augmentation steps to choose from.
        number_dictionary (int): The number of dictionaries to create.

        Returns:
        dict: Dictionary of augmentation parameters dictionaries.
        """
        # TODO: This should not be hard coded for future use 
        # Load the configuration file
        config_path = '/Users/yifeigu/Documents/Carney_Lab/NoodlePy/noodlepy/config/config.yml'
        with open(config_path, "rb") as yaml_file:
            config = yaml.safe_load(yaml_file)

        augmentation_par_dictionaries = defaultdict(dict)
        for i_dictionary in range(number_dictionary):
            
            # pick a random augmentation steps
            augmentation_step_option_list = Spectrum.random_augmentation_steps_generator(augmentation_step_option_list)

            # Create random parameters for each augment
            if "normalization" in augmentation_step_option_list:
                augmentation_par_dictionaries[i_dictionary]["normalization"]= {
                    "normalization_type": np.random.choice(
                    config["normalization"]["normalization_type_options"]
                    )
                }

            if "amplification" in augmentation_step_option_list:
                augmentation_par_dictionaries[i_dictionary]["amplification"]= {
                    "spectrum_amplifying_factor": np.random.uniform(
                    config["amplification"]["spectrum_amplifying_factor"]["low"],
                    config["amplification"]["spectrum_amplifying_factor"]["high"]
                    )
                }

            if "horizontal_shift" in augmentation_step_option_list:
                augmentation_par_dictionaries[i_dictionary]["horizontal_shift"]= np.random.uniform(
                    config["horizontal_shift"]["low"],
                    config["horizontal_shift"]["high"]
                )

            if "convoluting_gaussian" in augmentation_step_option_list:
                augmentation_par_dictionaries[i_dictionary]["convoluting_gaussian"]= {
                    "gaussian_std":np.random.uniform(
                    config["convoluting_gaussian"]["gaussian_std"]["low"],
                    config["convoluting_gaussian"]["gaussian_std"]["high"]
                    )
                }

            if "shot_noise" in augmentation_step_option_list:
                augmentation_par_dictionaries[i_dictionary]["shot_noise"]= {
                    "noise_type": "poisson",
                    "lam": np.random.uniform(
                        config["shot_noise"]["lam"]["low"],
                        config["shot_noise"]["lam"]["high"]
                    )
                }

            if "dark_current_noise" in augmentation_step_option_list:
                augmentation_par_dictionaries[i_dictionary]["dark_current_noise"]= {
                    "noise_type": "poisson",
                    "lam": np.random.uniform(
                        config["dark_current_noise"]["lam"]["low"],
                        config["dark_current_noise"]["lam"]["high"]
                    )
                }

            if "photo_response_non_uniformity" in augmentation_step_option_list:
                augmentation_par_dictionaries[i_dictionary]["photo_response_non_uniformity"]= {
                    "noise_type": "gaussian",
                    "mean": np.random.uniform(
                        config["photo_response_non_uniformity"]["mean"]["low"],
                        config["photo_response_non_uniformity"]["mean"]["high"]
                    ),
                    "std": np.random.uniform(
                        config["photo_response_non_uniformity"]["std"]["low"],
                        config["photo_response_non_uniformity"]["std"]["high"]
                    )
                }
                
                # TODO: This should be measured or generated once and used for all spectra
            if "FPN_noise" in augmentation_step_option_list:
                augmentation_par_dictionaries[i_dictionary]["FPN_noise"]= {
                    "noise_type": "lognormal",
                    "mean": np.random.uniform(
                        config["FPN_noise"]["mean"]["low"],
                        config["FPN_noise"]["mean"]["high"]
                    ),
                    "sigma": np.random.uniform(
                        config["FPN_noise"]["sigma"]["low"],
                        config["FPN_noise"]["sigma"]["high"]
                    )
                }

            if "cosmic_ray" in augmentation_step_option_list:
                augmentation_par_dictionaries[i_dictionary]["cosmic_ray"] = {  
                    "spike_number": np.random.randint(
                        config["cosmic_ray"]["spike_number"]["low"],
                        config["cosmic_ray"]["spike_number"]["high"]
                    ),
                    "spike_amplitude": np.random.uniform(
                        config["cosmic_ray"]["spike_amplitude"]["low"],
                        config["cosmic_ray"]["spike_amplitude"]["high"]
                    )
                }

            if "baseline" in augmentation_step_option_list:
                baseline_type = np.random.choice(
                    config["baseline"]["baseline_type_options"]
                )

                baseline_amplifying_factor = np.random.uniform(
                config["baseline"]["baseline_amplification_multiplier"]["low"],
                config["baseline"]["baseline_amplification_multiplier"]["high"]
                )

                poly_orders = np.random.randint(
                            config["baseline"]["poly_orders"]["low"],
                            config["baseline"]["poly_orders"]["high"]
                            )
                
                # REQ: poly_orders+1 because the zero order (constant)
                poly_coefficients = np.random.uniform(
                            config["baseline"]["poly_coefficients"]["low"],
                            config["baseline"]["poly_coefficients"]["high"],
                            poly_orders+1
                            )

                augmentation_par_dictionaries[i_dictionary]["baseline"] = {
                "baseline_type": baseline_type,
                "baseline_amplifying_factor": baseline_amplifying_factor,
                "poly_orders": poly_orders,
                "poly_coefficients": poly_coefficients
                }
        return augmentation_par_dictionaries


    @classmethod
    def from_file(cls, file_path:str):
        # placeholder for reading spectrum from file
        pass