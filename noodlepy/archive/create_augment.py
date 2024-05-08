# https://w3.cs.jmu.edu/spragunr/CS240_F12/style_guide.shtml

# Loading the required packages:
import numpy as np
from sklearn import base
import yaml
from noodlepy.utils.class_Spectrum import Spectrum
from typing import Any
import copy
from collections import defaultdict
import os

def pre_process(spectrum: Spectrum) -> Spectrum:

    # Load the configuration file

    #spectrum.display('chosen spectrum')

    #pre_processed_spectrum = copy.deepcopy(spectrum)
    #pre_processed_spectrum.crop_spectrum(config['cropping']['start'], config['cropping']['end']) # Range temporary chosen by Victor
    #pre_processed_spectrum.airPLS(lam= 1E3, diff_order=1, max_iter=15, tol=1e-3, weights=None) #baseline correction
    #pre_processed_spectrum.despike(kernel_size= 2, threshold= 3.5) # cosmic ray removal
    #pre_processed_spectrum.normalize_spectrum(normalization_type= 'by_max') # normalization
    #pre_processed_spectrum.savgol_filter(window_length=9, polyorder=2) # smoothing

    pre_processed_spectrum = copy.deepcopy(spectrum)
    pre_processed_spectrum.crop_spectrum(624.573, 1784.104) # Range temporary chosen by Victor
    #pre_processed_spectrum.display('cropped_spectrum')
    pre_processed_spectrum.airPLS(lam= 1E3, diff_order=1, max_iter=15, tol=1e-3, weights=None) #baseline correction
    pre_processed_spectrum.despike(kernel_size= 2, threshold= 3.5) # cosmic ray removal
    pre_processed_spectrum.normalize_spectrum(normalization_type= 'by_max') # normalization
    pre_processed_spectrum.savgol_filter(window_length=9, polyorder=2) # smoothing

    # pre_processed_spectrum.display('prepocessed spectrum')
    return pre_processed_spectrum


def random_augmentation_steps_generator(augmentation_step_option_list: list) -> list:
        """
        Choose random number of random step from augmentation_step_option_list

        Returns:
        list: The list of random augmentation steps.
        """
        number_of_augmentation_steps = np.random.randint(1,len(augmentation_step_option_list))
        random_augmentation_steps = np.random.choice((augmentation_step_option_list),number_of_augmentation_steps,replace=False)
        return random_augmentation_steps


def augmentation_par_dictionary_generator(augmentation_step_option_list: list[str],
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
    current_directory = os.getcwd()
    config_path = os.path.join(current_directory, "noodlepy", "config", "config.yml")
    
    with open(config_path, "rb") as yaml_file:
        config = yaml.safe_load(yaml_file)

    augmentation_par_dictionaries = defaultdict(dict)
    for i_dictionary in range(number_dictionary):
        
        # pick a random augmentation steps
        augmentation_step_list = random_augmentation_steps_generator(augmentation_step_option_list)
        # augmentation_step_list = augmentation_step_option_list # use all the augmentation steps

        # Create random parameters for each augment
        if "normalization" in augmentation_step_list:
            augmentation_par_dictionaries[i_dictionary]["normalization"]= {
                "normalization_type": np.random.choice(
                config["normalization"]["normalization_type_options"]
                )
            }

        if "amplification" in augmentation_step_list:
            augmentation_par_dictionaries[i_dictionary]["amplification"]= {
                "spectrum_amplifying_factor": np.random.uniform(
                config["amplification"]["spectrum_amplifying_factor"]["low"],
                config["amplification"]["spectrum_amplifying_factor"]["high"]
                )
            }

        if "horizontal_shift" in augmentation_step_list:
            augmentation_par_dictionaries[i_dictionary]["horizontal_shift"]= np.random.uniform(
                config["horizontal_shift"]["low"],
                config["horizontal_shift"]["high"]
            )

        if "convoluting_gaussian" in augmentation_step_list:
            augmentation_par_dictionaries[i_dictionary]["convoluting_gaussian"]= {
                "gaussian_std":np.random.uniform(
                config["convoluting_gaussian"]["gaussian_std"]["low"],
                config["convoluting_gaussian"]["gaussian_std"]["high"]
                )
            }

        if "shot_noise" in augmentation_step_list:
            augmentation_par_dictionaries[i_dictionary]["shot_noise"]= {
                "noise_type": "poisson",
                "lam": np.random.uniform(
                    config["shot_noise"]["lam"]["low"],
                    config["shot_noise"]["lam"]["high"]
                )
            }

        if "dark_current_noise" in augmentation_step_list:
            augmentation_par_dictionaries[i_dictionary]["dark_current_noise"]= {
                "noise_type": "poisson",
                "lam": np.random.uniform(
                    config["dark_current_noise"]["lam"]["low"],
                    config["dark_current_noise"]["lam"]["high"]
                )
            }

        if "photo_response_non_uniformity" in augmentation_step_list:
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
        if "FPN_noise" in augmentation_step_list:
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

        if "cosmic_ray" in augmentation_step_list:
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

        if "baseline" in augmentation_step_list:
            baseline_type = np.random.choice(
                config["baseline"]["baseline_type_options"]
            )

            baseline_amplifying_factor = np.random.uniform(
            config["baseline"]["baseline_amplifying_factor"]["low"],
            config["baseline"]["baseline_amplifying_factor"]["high"]
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


def apply_augmentations(spectrum: Spectrum, augmentation_step_list, number_of_augmentation) -> list[Spectrum]:
    """
    Apply a set of augmentations to a spectrum.
    
    Parameters:
    spectrum (Spectrum): The spectrum to be augmented.
    number_of_augmentation (int): The number of augmentations to be applied.
        
    Returns:
    Spectrum: The augmented spectra.
    """
    augmented_spectrum_list = []
    for i, i_augmentation_dictionary in augmentation_par_dictionary_generator(augmentation_step_list, number_of_augmentation).items():
        augmented_spectrum = copy.deepcopy(spectrum)

        if 'normalization' in i_augmentation_dictionary.keys():
            augmented_spectrum.normalize_spectrum(i_augmentation_dictionary["normalization"])

        if 'spectrum_amplifying_factor' in i_augmentation_dictionary.keys():
            augmented_spectrum.amplify_spectrum(i_augmentation_dictionary["amplification"])

        if 'horizontal_shift' in i_augmentation_dictionary.keys():
            augmented_spectrum.horizontal_shift(i_augmentation_dictionary["horizontal_shift"])

        if 'baseline' in i_augmentation_dictionary.keys():
            augmented_spectrum.add_baseline(**i_augmentation_dictionary["baseline"])

        if 'convoluting_gaussian' in i_augmentation_dictionary.keys():
            augmented_spectrum.convolve_with_gaussian(i_augmentation_dictionary["convoluting_gaussian"])

        if 'shot_noise' in i_augmentation_dictionary.keys():
            augmented_spectrum.add_noise(**i_augmentation_dictionary['shot_noise'])

        if 'dark_current_noise' in i_augmentation_dictionary.keys():
            augmented_spectrum.add_noise(**i_augmentation_dictionary['dark_current_noise'])

        if 'photo_response_non_uniformity' in i_augmentation_dictionary.keys():
            augmented_spectrum.add_noise(**i_augmentation_dictionary['photo_response_non_uniformity'])

        if 'FPN_noise' in i_augmentation_dictionary.keys():
            augmented_spectrum.add_noise(**i_augmentation_dictionary['FPN_noise'])

        if 'cosmic_ray' in i_augmentation_dictionary.keys():
            augmented_spectrum.add_cosmic_rays(**i_augmentation_dictionary['cosmic_ray'])

        augmented_spectrum_list.append(augmented_spectrum)
    
        # plot the augmented spectrum
        # augmented_spectrum.display('augmented spectrum' + str(i+1))


    return augmented_spectrum_list