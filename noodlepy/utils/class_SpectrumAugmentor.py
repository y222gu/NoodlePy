import copy
from noodlepy.utils.class_Spectrum import Spectrum
import os
import yaml
import numpy as np
from collections import defaultdict
   
class SpectrumAugmentor:
    def __init__(self, 
                ramdom_augmentations = False,
                augmentation_step_list = None,
                config_path: str = None):
        
        if config_path is None:
            current_directory = os.getcwd()
            config_path = os.path.join(current_directory, "noodlepy", "config", "config.yml")

        with open(config_path, "rb") as yaml_file:
            config = yaml.safe_load(yaml_file)
        self.config = config['augmentation']

        if augmentation_step_list is None:
            augmentation_step_list = self.config['default_augmentation_steps']
        self.augmentation_step_list = augmentation_step_list

        self.ramdom_augmentations = ramdom_augmentations

    def augment(self, spectrum: Spectrum, number_of_augmentation) -> list[Spectrum]:
        """
        Apply a set of augmentations to a spectrum.
        
        Parameters:
        spectrum (Spectrum): The spectrum to be augmented.
        number_of_augmentation (int): The number of augmentations to be applied.
            
        Returns:
        Spectrum: The augmented spectra.
        """

        augmented_spectrum_list = []
        for i in range(number_of_augmentation):

            if self.ramdom_augmentations:
                augmentation_steps_to_apply = SpectrumAugmentor.random_augmentation_steps_generator(self.augmentation_step_list)
            else:
                augmentation_steps_to_apply = self.augmentation_step_list
                
            i_augmentation_par_dictionary = SpectrumAugmentor.augmentation_par_dictionary_generator(self, augmentation_steps_to_apply=augmentation_steps_to_apply)
            
            augmented_spectrum = copy.deepcopy(spectrum)

            if 'normalization' in i_augmentation_par_dictionary.keys():
                augmented_spectrum.normalize_spectrum(i_augmentation_par_dictionary["normalization"])

            if 'spectrum_amplifying_factor' in i_augmentation_par_dictionary.keys():
                augmented_spectrum.amplify_spectrum(i_augmentation_par_dictionary["amplification"])

            if 'horizontal_shift' in i_augmentation_par_dictionary.keys():
                augmented_spectrum.horizontal_shift(i_augmentation_par_dictionary["horizontal_shift"])

            if 'baseline' in i_augmentation_par_dictionary.keys():
                augmented_spectrum.add_baseline(**i_augmentation_par_dictionary["baseline"])

            if 'convoluting_gaussian' in i_augmentation_par_dictionary.keys():
                augmented_spectrum.convolve_with_gaussian(i_augmentation_par_dictionary["convoluting_gaussian"])

            if 'shot_noise' in i_augmentation_par_dictionary.keys():
                augmented_spectrum.add_noise(**i_augmentation_par_dictionary['shot_noise'])

            if 'dark_current_noise' in i_augmentation_par_dictionary.keys():
                augmented_spectrum.add_noise(**i_augmentation_par_dictionary['dark_current_noise'])

            if 'photo_response_non_uniformity' in i_augmentation_par_dictionary.keys():
                augmented_spectrum.add_noise(**i_augmentation_par_dictionary['photo_response_non_uniformity'])

            if 'FPN_noise' in i_augmentation_par_dictionary.keys():
                augmented_spectrum.add_noise(**i_augmentation_par_dictionary['FPN_noise'])

            if 'cosmic_ray' in i_augmentation_par_dictionary.keys():
                augmented_spectrum.add_cosmic_rays(**i_augmentation_par_dictionary['cosmic_ray'])

            augmented_spectrum_list.append(augmented_spectrum)
        
            # plot the augmented spectrum
            # augmented_spectrum.display('augmented spectrum' + str(i+1))

        return augmented_spectrum_list
    
    def random_augmentation_steps_generator(augmentation_step_list) -> list:
        """
        Choose random number of random step from augmentation_step_option_list

        Returns:
        list: The list of random augmentation steps.
        """
        number_of_random_steps = np.random.randint(1,len(augmentation_step_list))
        random_augmentation_steps = np.random.choice((augmentation_step_list),number_of_random_steps,replace=False)
        return random_augmentation_steps


    def augmentation_par_dictionary_generator(self, augmentation_steps_to_apply) -> dict:
        """
        Generate dictionaries with random parameters for random augmentation step chosen from augmentation_step_option_list.

        Parameters:
        augmentation_step_option_list (list): The list of augmentation steps to choose from.
        number_dictionary (int): The number of dictionaries to create.

        Returns:
        dict: Dictionary of augmentation parameters dictionaries.
        """
        augmentation_par_dictionary = defaultdict(dict)
            
        # Create random parameters for each augment
        if "normalization" in augmentation_steps_to_apply:
            augmentation_par_dictionary["normalization"]= {
                "normalization_type": np.random.choice(
                self.config["normalization"]["normalization_type_options"]
                )
            }

        if "amplification" in augmentation_steps_to_apply:
            augmentation_par_dictionary["amplification"]= {
                "spectrum_amplifying_factor": np.random.uniform(
                self.config["amplification"]["spectrum_amplifying_factor"]["low"],
                self.config["amplification"]["spectrum_amplifying_factor"]["high"]
                )
            }

        if "horizontal_shift" in augmentation_steps_to_apply:
            augmentation_par_dictionary["horizontal_shift"]= np.random.uniform(
                self.config["horizontal_shift"]["low"],
                self.config["horizontal_shift"]["high"]
            )

        if "convoluting_gaussian" in augmentation_steps_to_apply:
            augmentation_par_dictionary["convoluting_gaussian"]= {
                "gaussian_std":np.random.uniform(
                self.config["convoluting_gaussian"]["gaussian_std"]["low"],
                self.config["convoluting_gaussian"]["gaussian_std"]["high"]
                )
            }

        if "shot_noise" in augmentation_steps_to_apply:
            augmentation_par_dictionary["shot_noise"]= {
                "noise_type": "poisson",
                "lam": np.random.uniform(
                    self.config["shot_noise"]["lam"]["low"],
                    self.config["shot_noise"]["lam"]["high"]
                )
            }

        if "dark_current_noise" in augmentation_steps_to_apply:
            augmentation_par_dictionary["dark_current_noise"]= {
                "noise_type": "poisson",
                "lam": np.random.uniform(
                    self.config["dark_current_noise"]["lam"]["low"],
                    self.config["dark_current_noise"]["lam"]["high"]
                )
            }

        if "photo_response_non_uniformity" in augmentation_steps_to_apply:
            augmentation_par_dictionary["photo_response_non_uniformity"]= {
                "noise_type": "gaussian",
                "mean": np.random.uniform(
                    self.config["photo_response_non_uniformity"]["mean"]["low"],
                    self.config["photo_response_non_uniformity"]["mean"]["high"]
                ),
                "std": np.random.uniform(
                    self.config["photo_response_non_uniformity"]["std"]["low"],
                    self.config["photo_response_non_uniformity"]["std"]["high"]
                )
            }
            
            # TODO: This should be measured or generated once and used for all spectra
        if "FPN_noise" in augmentation_steps_to_apply:
            augmentation_par_dictionary["FPN_noise"]= {
                "noise_type": "lognormal",
                "mean": np.random.uniform(
                    self.config["FPN_noise"]["mean"]["low"],
                    self.config["FPN_noise"]["mean"]["high"]
                ),
                "sigma": np.random.uniform(
                    self.config["FPN_noise"]["sigma"]["low"],
                    self.config["FPN_noise"]["sigma"]["high"]
                )
            }

        if "cosmic_ray" in augmentation_steps_to_apply:
            augmentation_par_dictionary["cosmic_ray"] = {  
                "spike_number": np.random.randint(
                    self.config["cosmic_ray"]["spike_number"]["low"],
                    self.config["cosmic_ray"]["spike_number"]["high"]
                ),
                "spike_amplitude": np.random.uniform(
                    self.config["cosmic_ray"]["spike_amplitude"]["low"],
                    self.config["cosmic_ray"]["spike_amplitude"]["high"]
                )
            }

        if "baseline" in augmentation_steps_to_apply:
            baseline_type = np.random.choice(
                self.config["baseline"]["baseline_type_options"]
            )

            baseline_amplifying_factor = np.random.uniform(
            self.config["baseline"]["baseline_amplifying_factor"]["low"],
            self.config["baseline"]["baseline_amplifying_factor"]["high"]
            )

            poly_orders = np.random.randint(
                        self.config["baseline"]["poly_orders"]["low"],
                        self.config["baseline"]["poly_orders"]["high"]
                        )
            
            # REQ: poly_orders+1 because the zero order (constant)
            poly_coefficients = np.random.uniform(
                        self.config["baseline"]["poly_coefficients"]["low"],
                        self.config["baseline"]["poly_coefficients"]["high"],
                        poly_orders+1
                        )

            augmentation_par_dictionary["baseline"] = {
            "baseline_type": baseline_type,
            "baseline_amplifying_factor": baseline_amplifying_factor,
            "poly_orders": poly_orders,
            "poly_coefficients": poly_coefficients
            }

        return augmentation_par_dictionary