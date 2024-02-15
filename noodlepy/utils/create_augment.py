# https://w3.cs.jmu.edu/spragunr/CS240_F12/style_guide.shtml

# Loading the required packages:
from cvxopt import uniform
import numpy as np
import ramanspy
from scipy import signal
import yaml
from noodlepy.utils.spectrum_class import Spectrum
from typing import Union
import copy

def crop_spectra(
    spectrum: Spectrum,
    start_wavenumber: int,
    end_wavenumber: int
) -> Spectrum:
    
    print(f'Start to align all spectra to the specified range...')
    print(f'Checking if the wavenumber are matching')

    wavenumber = spectrum.wavenumber

    if start_wavenumber in wavenumber and end_wavenumber in wavenumber:
        start_index = np.where(wavenumber == start_wavenumber)[0][0]
        end_index = np.where(wavenumber == end_wavenumber)[0][0]
        spectrum.wavenumber = spectrum.wavenumber[start_index:end_index]
        spectrum.intensity = spectrum.intensity[start_index:end_index]
    else:
        raise ValueError('Wavenumber range could not be aligned to the specified range')
    
    print(f'All spectra are aligned and cropped between Wavenumber {start_wavenumber} to {end_wavenumber}')
    return spectrum

def normalize_spectra(
    spectrum: Spectrum,
    normalization_option: str,
) -> Union[Spectrum, list]:
    
    print(f'Normalizing spectra by: ', normalization_option, '...')
    
    if normalization_option == 'area':
        spectrum.intensity /= np.trapz(spectrum.intensity, spectrum.wavelength_nm)
    elif normalization_option == 'max':
        spectrum.intensity /= np.max(spectrum.intensity)
    else:
        raise ValueError(f'Mormalization method is not defined')

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
            mixture_spectrum.wavenumber = spectrum.wavenumber
        else:
            if np.array_equal(spectrum.wavenumber, mixture_spectrum.wavenumber):
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
    spike_locations = np.random.randint(0, len(spectrum.wavenumber), number_spikes)

    for spike_location in spike_locations:
        spectrum.intensity[spike_location] = abs(spectrum.intensity[spike_location])* spike_amplitude
    return spectrum

def create_baseline(
    spectrum: Spectrum,
    baseline_pars: dict
) -> Spectrum:
    
    baseline_type = baseline_pars["baseline_type"]

    if baseline_type == "poly":
        poly_orders = baseline_pars["poly_orders"]
        poly_coefficients = baseline_pars["poly_coefficients"]
        poly_displacement = baseline_pars["poly_displacement"]

        baseline_intensity = 0
        for i in range(poly_orders + 1):
            baseline_intensity += poly_coefficients[i] * (spectrum.wavenumber - poly_displacement[i]) ** i

    elif baseline_type == "sine":  # Sine wave baseline
        sine_amplitude = baseline_pars["sine_amplitude"]
        sine_frequency = baseline_pars["sine_frequency"]
        sine_phase = baseline_pars["sine_phase"]
        baseline_intensity = sine_amplitude * np.sin(sine_frequency * spectrum.wavenumber + sine_phase)

    else:
        baseline_intensity = np.zeros_like(spectrum.wavenumber)  # No baseline

    spectrum.intensity = spectrum.intensity + baseline_intensity
        
    return spectrum

def shift_spectrum(spectrum: Spectrum, wavenumber_shift: float) -> Spectrum:

    spectrum.wavenumber = spectrum.wavenumber + wavenumber_shift
    return spectrum

def amplify(spectrum: Spectrum, spectrum_amplifying_factor: float) -> Spectrum:
 
    spectrum.intensity = spectrum.intensity * spectrum_amplifying_factor
    return spectrum

def convolute_kernel(spectrum: Spectrum, kernel_std: float) -> Spectrum:

    kernel = signal.windows.gaussian(len(spectrum.wavenumber), kernel_std)
    spectrum.intensity = signal.convolve(kernel, spectrum.intensity, mode="same") * sum(kernel)
    return spectrum

def interpolate_spectrum(spectrum: Spectrum, target_raman_shift: np.array) -> Spectrum:

    interpolated_intensity = np.interp(target_raman_shift, spectrum.wavenumber, spectrum.intensity)
    spectrum.wavenumber= target_raman_shift
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
        ramanspy.Spectrum(spectrum.intensity, spectrum.wavenumber)
    )
    return preprocessed_spectrum

def wavelength_to_wavenumber(wl:np.array) -> np.array:
    """
    Convert wavelength to wavenumber.
    Parameters:
    wl (np.array): The wavelength in nm.
    Returns:
    np.array: The wavenumber in cm^-1.
    """
    # first convent from nm to cm
    wlCM = wl * (1e-7)
    # then invert to cm^-1
    wn = np.power(wlCM, -1)
    return wn

def augmentation_pars_generator(
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
    config_path = '/Users/yifeigu/Documents/Carney_Lab/NoodlePy/noodlepy/config/config.yml'
    # Load the configuration file
    with open(config_path, "rb") as yaml_file:
        config = yaml.safe_load(yaml_file)

    # Create a set of augmentation parameters
    augmentation_par_dictionaries = {}
    for i_dictionary in range(number_dictionaries):
        # Create a set of augmentation parameters
        augmentation_par_dictionaries[i_dictionary] = {
            "spectrum_amplifying_factor": np.random.uniform(
                config["spectrum_amplifying_factor"]["low"],
                config["spectrum_amplifying_factor"]["high"],
            ),
            "instrument_shift": np.random.uniform(
                config["instrument_shift"]["low"], config["instrument_shift"]["high"]
            ),
            "aberration_kernel_std": np.random.uniform(
                config["aberration_kernel_std"]["low"],
                config["aberration_kernel_std"]["high"],
            ),
            "photon_shot_noise_pars": {
                "noise_type": "poisson",
                "lam": np.random.uniform(
                    config["photon_shot_noise_pars"]["lam"]["low"],
                    config["photon_shot_noise_pars"]["lam"]["high"],
                ),
            },
            "dark_current_shot_noise_pars": {
                "noise_type": "poisson",
                "lam": np.random.uniform(
                    config["dark_current_shot_noise_pars"]["lam"]["low"],
                    config["dark_current_shot_noise_pars"]["lam"]["high"],
                ),
            },
            "photo_response_non_uniformity_pars": {
                "noise_type": "gaussian",
                "mean": np.random.uniform(
                    config["photo_response_non_uniformity_pars"]["mean"]["low"],
                    config["photo_response_non_uniformity_pars"]["mean"]["high"],
                ),
                "std": np.random.uniform(
                    config["photo_response_non_uniformity_pars"]["std"]["low"],
                    config["photo_response_non_uniformity_pars"]["std"]["high"],
                ),
            },
            "dark_signal_FPN_noise_pars": {
                "noise_type": "lognormal",
                "mean": np.random.uniform(
                    config["dark_signal_FPN_noise_pars"]["mean"]["low"],
                    config["dark_signal_FPN_noise_pars"]["mean"]["high"],
                ),
                "sigma": np.random.uniform(
                    config["dark_signal_FPN_noise_pars"]["sigma"]["low"],
                    config["dark_signal_FPN_noise_pars"]["sigma"]["high"],
                ),
            },
            "cosmic_ray_pars": {
                "spike_num": np.random.randint(
                    config["cosmic_ray_pars"]["spike_num"]["low"],
                    config["cosmic_ray_pars"]["spike_num"]["high"],
                ),
                "spike_amplitude": np.random.uniform(
                    config["cosmic_ray_pars"]["spike_amplitude"]["low"],
                    config["cosmic_ray_pars"]["spike_amplitude"]["high"],
                ),
            },
            "aberration_kernel_std": np.random.uniform(
                config["aberration_kernel_std"]["low"], config["aberration_kernel_std"]["high"]
            ),

            # TODO: ADD PARS FOR BASELINE
        }
    return augmentation_par_dictionaries

def apply_augmentations(spectrum: Spectrum, number_of_augmentation) -> list[Spectrum]:
    """
    Apply a set of augmentations to a spectrum.
    
    Parameters:
    spectrum (Spectrum): The spectrum to be augmented.
    number_of_augmentation (int): The number of augmentations to be applied.
        
    Returns:
    Spectrum: The augmented spectra.
    """
    augmented_spectrum_list = []
    for i, i_augmentation_dictionary in augmentation_pars_generator(number_of_augmentation).items():
        augmented_spectrum = copy.deepcopy(spectrum)
        augmented_spectrum = amplify(augmented_spectrum, i_augmentation_dictionary["spectrum_amplifying_factor"])
        augmented_spectrum = shift_spectrum(augmented_spectrum, i_augmentation_dictionary["instrument_shift"])
        augmented_spectrum = convolute_kernel(augmented_spectrum, i_augmentation_dictionary["aberration_kernel_std"])
        augmented_spectrum = add_noise(augmented_spectrum, i_augmentation_dictionary["photon_shot_noise_pars"])
        augmented_spectrum = add_noise(augmented_spectrum, i_augmentation_dictionary["dark_current_shot_noise_pars"])
        augmented_spectrum = add_noise(augmented_spectrum, i_augmentation_dictionary["photo_response_non_uniformity_pars"])
        augmented_spectrum = add_noise(augmented_spectrum, i_augmentation_dictionary["dark_signal_FPN_noise_pars"])
        augmented_spectrum = add_cosmic_rays(augmented_spectrum, i_augmentation_dictionary["cosmic_ray_pars"])
        augmented_spectrum_list.append(augmented_spectrum)
    return augmented_spectrum_list