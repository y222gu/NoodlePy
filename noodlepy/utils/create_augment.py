# https://w3.cs.jmu.edu/spragunr/CS240_F12/style_guide.shtml

# Loading the required packages:
import re
from cvxopt import uniform
import numpy as np
import ramanspy
from scipy import signal
import yaml
from noodlepy.utils.spectrum_class import Spectrum
from typing import Union
import copy
from collections import defaultdict

def crop_spectra(
    spectrum: Spectrum,
    start_wavenumber: int,
    end_wavenumber: int
) -> Spectrum:
    
    print(f'Start to align all spectra to the specified range...')
    print(f'Checking if the wavenumber are matching')

    wavenumber = spectrum.wavenumber_cm

    if start_wavenumber in wavenumber and end_wavenumber in wavenumber:
        start_index = np.where(wavenumber == start_wavenumber)[0][0]
        end_index = np.where(wavenumber == end_wavenumber)[0][0]
        spectrum.wavenumber_cm = spectrum.wavenumber_cm[start_index:end_index]
        spectrum.intensity = spectrum.intensity[start_index:end_index]
    else:
        raise ValueError('Wavenumber range could not be aligned to the specified range')
    
    print(f'All spectra are aligned and cropped between Wavenumber {start_wavenumber} to {end_wavenumber}')
    return spectrum

def normalize_spectra(
    spectrum: Spectrum,
    normalization_type: str,
) -> Union[Spectrum, list]:
    
    print(f'Normalizing spectra by: ', normalization_type, '...')
    
    if normalization_type == 'by_area':
        spectrum.intensity /= np.trapz(spectrum.intensity, spectrum.wavelength_nm)
    elif normalization_type == 'by_max':
        spectrum.intensity /= np.max(spectrum.intensity)
    else:
        raise ValueError(f'Normalization method is not defined')

    return spectrum

def apply_quantumn_efficiency(spectrum: Spectrum, quantumn_efficiency: float) -> Spectrum:
    
    spectrum.intensity = spectrum.intensity * quantumn_efficiency
    return spectrum

def mix_spectra(
    spectrum_list: list[Spectrum],
    ratios: np.array,
) -> Spectrum:
    
    print(f'Start to mix spectra...')
    mixture_spectrum = Spectrum()

    for i, spectrum in enumerate(spectrum_list):
        if i == 0:
            mixture_spectrum.intensity = (ratios[i] * spectrum.intensity)
            mixture_spectrum.wavenumber_cm = spectrum.wavenumber_cm
        else:
            if np.array_equal(spectrum.wavenumber_cm, mixture_spectrum.wavenumber_cm):
                mixture_spectrum.intensity += (ratios[i] * spectrum.intensity)
            else:
                raise ValueError(f'Raman shift range of "{i}" spectrum in the list does not match with the other spectra')
    
    print(f'Mixing is done...')
    return mixture_spectrum

def add_noise(spectrum: Spectrum, noise_pars: dict) -> Spectrum:
    
    rng = np.random.default_rng()
    noise_type = noise_pars["noise_type"]

    if noise_type == "poisson":
        spectrum.intensity += 0.001 * rng.poisson(noise_pars["lam"], len(spectrum.intensity))
    elif noise_type == "gaussian":
        spectrum.intensity += rng.normal(noise_pars["mean"], noise_pars["std"], len(spectrum.intensity))
    elif noise_type == "uniform":
        spectrum.intensity += rng.uniform(noise_pars["low"], noise_pars["high"], len(spectrum.intensity))
    elif noise_type == "exponential":
        spectrum.intensity += rng.exponential(noise_pars["mean"], len(spectrum.intensity))
    elif noise_type == "lognormal":
        spectrum.intensity += rng.lognormal(noise_pars["mean"], noise_pars["sigma"], len(spectrum.intensity))
    else:
        raise ValueError("noise_type must be one among 'poisson', 'gaussian', 'uniform', 'expoenetial', and 'lognormal'")

    return spectrum

def add_cosmic_rays(spectrum: Spectrum, cosmic_ray_pars: dict) -> Spectrum:
    
    number_spikes = cosmic_ray_pars["spike_num"]
    spike_amplitude = cosmic_ray_pars["spike_amplitude"]
    spike_locations = np.random.randint(0, len(spectrum.wavenumber_cm), number_spikes)

    for spike_location in spike_locations:
        spectrum.intensity[spike_location] = abs(spectrum.intensity[spike_location])* spike_amplitude
    return spectrum

def add_baseline(
    spectrum: Spectrum,
    baseline_pars: dict
) -> Spectrum:
    
    baseline_type = baseline_pars["baseline_type"]
    baseline_amplifying_factor = baseline_pars["baseline_amplifying_factor"]
    if baseline_type == "poly":
        poly_orders = baseline_pars["poly_orders"]
        poly_coefficients = baseline_pars["poly_coefficients"]

        baseline_intensity = 0
        for order in range(poly_orders + 1):
            baseline_intensity += poly_coefficients[order] * (spectrum.wavenumber_cm ** order)

    elif baseline_type == "sine":  # Sine wave baseline
        sine_amplitude = baseline_pars["sine_amplitude"]
        sine_frequency = baseline_pars["sine_frequency"]
        sine_phase = baseline_pars["sine_phase"]
        
        baseline_intensity = sine_amplitude * np.sin(sine_frequency * spectrum.wavenumber_cm + sine_phase)
    else:
        raise ValueError("Baseline type is not defined")

    spectrum.intensity = spectrum.intensity + baseline_intensity * baseline_amplifying_factor
    return spectrum

def shift_spectrum(spectrum: Spectrum, wavenumber_shift: float) -> Spectrum:
    spectrum.wavenumber_cm = spectrum.wavenumber_cm + wavenumber_shift
    return spectrum

def amplify_spectrum(spectrum: Spectrum, spectrum_amplifying_factor: float) -> Spectrum:
    spectrum.intensity = spectrum.intensity * spectrum_amplifying_factor
    return spectrum

def convolute_gaussian_to_spectrum(spectrum: Spectrum, gaussian_std: float) -> Spectrum:
    kernel = signal.windows.gaussian(len(spectrum.wavenumber_cm), gaussian_std)
    spectrum.intensity = signal.convolve(kernel, spectrum.intensity, mode="same") * sum(kernel)
    return spectrum

def interpolate_spectrum(spectrum: Spectrum, target_raman_shift: np.array) -> Spectrum:
    interpolated_intensity = np.interp(target_raman_shift, spectrum.wavenumber_cm, spectrum.intensity)
    spectrum.wavenumber_cm= target_raman_shift
    spectrum.intensity = interpolated_intensity
    return spectrum

def pre_process(spectrum: Spectrum, config: dict) -> Spectrum:
    pipe = ramanspy.preprocessing.protocols.Pipeline(
        [
            ramanspy.preprocessing.despike.WhitakerHayes(),
            ramanspy.preprocessing.denoise.SavGol(window_length=12, polyorder=3),
            ramanspy.preprocessing.baseline.ASPLS(),
            ramanspy.preprocessing.normalise.MinMax(pixelwise=True),
        ]
    )
    preprocessed_spectrum = pipe.apply(
        ramanspy.Spectrum(spectrum.intensity, spectrum.wavenumber_cm)
    )
    return preprocessed_spectrum

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

def wavelength_to_raman_shift(wavelength_nm:np.array,laser_wavelength_nm) -> np.array:
    """
    Convert spectrum from wavelength to Raman shift.
    """
    # first convent from nm to cm
    wavenumber_cm = wavelength_to_wavenumber(wavelength_nm)
    laser_wavenumber_cm = wavelength_to_wavenumber(laser_wavelength_nm)
    raman_shift_cm = laser_wavenumber_cm - wavenumber_cm
    raman_shift_cm = raman_shift_cm.round(3)
    return raman_shift_cm

def random_augmentation_steps_generator(augmentation_step_list: list) -> list:
    """
    Create a set of augmentation steps.

    Returns:
    dict: The set of augmentation steps.
    """
    number_of_augmentation_steps = np.random.randint(0,len(augmentation_step_list))
    random_augmentation_steps = np.random.choice((augmentation_step_list),number_of_augmentation_steps,replace=False)
    return random_augmentation_steps


def augmentation_pars_generator(augmentation_step_list: list,
    number_dictionaries: int
) -> dict:
    """
    Create a set of augmentation parameters.

    Parameters:
    config (dict): The configuration file.
    n_sets (int): The number of sets to be created.

    Returns:
    dict: The set of augmentation parameters.
    """
    # Load the configuration file
    config_path = '/Users/yifeigu/Documents/Carney_Lab/NoodlePy/noodlepy/config/config.yml'
    with open(config_path, "rb") as yaml_file:
        config = yaml.safe_load(yaml_file)

    augmentation_par_dictionaries = defaultdict(dict)
    for i_dictionary in range(number_dictionaries):
        
        # pick a random augmentation steps
        augmentation_steps = random_augmentation_steps_generator(augmentation_step_list)

        # Create random parameters for each augment
        if "normalization" in augmentation_steps:
            augmentation_par_dictionaries[i_dictionary]["normalization"]= np.random.choice(
                config["normalization"]["normalization_type_options"]
                )

        if "amplification" in augmentation_steps:
            augmentation_par_dictionaries[i_dictionary]["amplification"]= {
                "multipliers": np.random.uniform(
                config["amplification"]["multipliers"]["low"],
                config["amplification"]["multipliers"]["high"]
                )
            }

        if "horizontal_shift" in augmentation_steps:
            augmentation_par_dictionaries[i_dictionary]["horizontal_shift"]= np.random.uniform(
                config["horizontal_shift"]["low"],
                config["horizontal_shift"]["high"]
            )

        if "convoluting_gaussian" in augmentation_steps:
            augmentation_par_dictionaries[i_dictionary]["convoluting_gaussian"]= {
                "gaussian_std":np.random.uniform(
                config["convoluting_gaussian"]["gaussian_std"]["low"],
                config["convoluting_gaussian"]["gaussian_std"]["high"]
                )
            }

        if "shot_noise" in augmentation_steps:
            augmentation_par_dictionaries[i_dictionary]["shot_noise"]= {
                "noise_type": "poisson",
                "lam": np.random.uniform(
                    config["shot_noise"]["lam"]["low"],
                    config["shot_noise"]["lam"]["high"]
                )
            }

        if "dark_current_noise" in augmentation_steps:
            augmentation_par_dictionaries[i_dictionary]["dark_current_noise"]= {
                "noise_type": "poisson",
                "lam": np.random.uniform(
                    config["dark_current_noise"]["lam"]["low"],
                    config["dark_current_noise"]["lam"]["high"]
                )
            }

        if "photo_response_non_uniformity" in augmentation_steps:
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
        if "FPN_noise" in augmentation_steps:
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

        if "cosmic_ray" in augmentation_steps:
            augmentation_par_dictionaries[i_dictionary]["cosmic_ray"] = {  
                "spike_num": np.random.randint(
                    config["cosmic_ray"]["spike_num"]["low"],
                    config["cosmic_ray"]["spike_num"]["high"]
                ),
                "spike_amplitude": np.random.uniform(
                    config["cosmic_ray"]["spike_amplitude"]["low"],
                    config["cosmic_ray"]["spike_amplitude"]["high"]
                )
            },

        if "baseline" in augmentation_steps:
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
            "baseline_type": "poly",
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
    for i, i_augmentation_dictionary in augmentation_pars_generator(augmentation_step_list, number_of_augmentation).items():
        augmented_spectrum = copy.deepcopy(spectrum)

        if 'normalization' in i_augmentation_dictionary.keys():
            augmented_spectrum = normalize_spectra(augmented_spectrum, i_augmentation_dictionary["normalization"])

        if 'spectrum_amplifying_factor' in i_augmentation_dictionary.keys():
            augmented_spectrum = amplify_spectrum(augmented_spectrum, i_augmentation_dictionary["amplification"])

        if 'horizontal_shift' in i_augmentation_dictionary.keys():
            augmented_spectrum = shift_spectrum(augmented_spectrum, i_augmentation_dictionary["horizontal_shift"])

        if 'baseline' in i_augmentation_dictionary.keys():
            augmented_spectrum = add_baseline(augmented_spectrum, i_augmentation_dictionary["baseline"])

        if 'convoluting_gaussian' in i_augmentation_dictionary.keys():
            augmented_spectrum = convolute_gaussian_to_spectrum(augmented_spectrum, i_augmentation_dictionary["convoluting_gaussian"])

        if 'shot_noise' in i_augmentation_dictionary.keys():
            augmented_spectrum = add_noise(augmented_spectrum, i_augmentation_dictionary['shot_noise'])

        if 'dark_current_noise' in i_augmentation_dictionary.keys():
            augmented_spectrum = add_noise(augmented_spectrum, i_augmentation_dictionary['dark_current_noise'])

        if 'photo_response_non_uniformity' in i_augmentation_dictionary.keys():
            augmented_spectrum = add_noise(augmented_spectrum, i_augmentation_dictionary['photo_response_non_uniformity'])

        if 'FPN_noise' in i_augmentation_dictionary.keys():
            augmented_spectrum = add_noise(augmented_spectrum, i_augmentation_dictionary['FPN_noise'])

        if 'cosmic_ray' in i_augmentation_dictionary.keys():
            augmented_spectrum = add_cosmic_rays(augmented_spectrum, i_augmentation_dictionary['cosmic_ray'])

        augmented_spectrum_list.append(augmented_spectrum)

    return augmented_spectrum_list