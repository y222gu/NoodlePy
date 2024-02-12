# https://w3.cs.jmu.edu/spragunr/CS240_F12/style_guide.shtml

# Loading the required packages:
import numpy as np
import ramanspy
from scipy import signal
import yaml
from noodlepy.utils.spectrum_class import Spectrum

def crop_spectra(
    spectrum_list: Spectrum,
    start_wavenumber: int,
    end_wavenumber: int
) -> Spectrum:

    print(f'Start to align all spectra to the specified range...')
    print(f'Checking if the raman shift range of molecules are matching')

    for spectrum in spectrum_list:
        wavenumber = spectrum.wavenumber

        # test if the start_wavenumber exist in the raman shift

        if start_wavenumber in wavenumber and end_wavenumber in wavenumber:
            # Find the indices corresponding to the start and end wavelengths
            start_index = np.where(wavenumber==start_wavenumber)[0][0]
            end_index = np.where(wavenumber==end_wavenumber)[0][0]

            # Crop the spectrum based on the specified wavelength range
            spectrum.wavenumber = spectrum.wavenumber[start_index:end_index]
            spectrum.intensity = spectrum.intensity[start_index:end_index]

        else:
            raise ValueError('Wavenumber range could not be aligned to the specified range')
    print(f'All spectra are aligned and cropped between Wavenumber {start_wavenumber} to {end_wavenumber}')

    return spectrum_list

def normalize_spectra(
    spectrum_list: Spectrum,
    normalization_option: str,
) -> Spectrum:

    print(f'Normalizing spectra by: ', normalization_option, '...')

    for spectrum in spectrum_list:

        # normalize by the area under the curve:
        if normalization_option == 'area':
            spectrum.intensity /= np.trapz(spectrum.intensity, spectrum.wavelength_nm)
        # normalize by the maximum intensity:
        elif normalization_option == 'max':
            spectrum.intensity /= np.max(spectrum.intensity)
        else:
            raise ValueError(f'Mormalization method is not defined')

    return spectrum_list

def apply_quantumn_efficiency(spectrum_list: Spectrum, quantumn_efficiency: float) -> dict:

    for spectrum in spectrum_list:
        # intensity
        spectrum.intensity = spectrum.intensity* quantumn_efficiency

    return spectrum_list
    

def mix_spectra(
    spectrum_list: Spectrum,
    metabolite_ratios: np.array,
    mixture_name: str,
) -> Spectrum:

    print(f'Start to mix spectra...')
    mixture_spectrum = Spectrum()

    for i, spectrum in enumerate(spectrum_list):
        if i == 0:
            mixture_spectrum.intensity = (metabolite_ratios[i] * spectrum.intensity)
            mixture_spectrum.wavenumber = spectrum.wavenumber

        else:
            # double check if the wavenumber_range is matching
            if np.array_equal(spectrum.wavenumber, mixture_spectrum.wavenumber):
                mixture_spectrum.intensity += (metabolite_ratios[i] * spectrum.intensity)
            
            else:
                raise ValueError(f'Raman shift range of "{i}" spectrum in the list does not match with the other spectra')
            
    print(f'Mixing is done...')        

    return mixture_spectrum


def add_noise(spectrum_list: Spectrum, noise_pars: dict) -> dict:

    rng = np.random.default_rng() # this is using the default PCG64 generator

    for spectrum in spectrum_list:

        noise_type = noise_pars["noise_type"]

        if noise_type == "poisson":
            spectrum.intensity += 0.001*rng.poisson(noise_pars["lam"], len(spectrum.intensity))

        elif noise_type == "gaussian":
            spectrum.intensity += rng.normal(noise_pars["mean"], noise_pars["std"], len(spectrum.intensity))

        elif noise_type == "uniform":
            spectrum.intensity += rng.uniform(noise_pars["low"],noise_pars["high"],len(spectrum.intensity))

        elif noise_type == "exponential":
            spectrum.intensity += rng.exponential(noise_pars["mean"], len(spectrum.intensity))

        elif noise_type == "lognormal":
            spectrum.intensity += rng.lognormal(noise_pars["mean"],noise_pars["sigma"],len(spectrum.intensity))
        else:
            raise ValueError("noise_type must be one among 'poisson', 'gaussian', 'uniform', 'expoenetial', and 'lognormal'")

    return spectrum_list


def add_cosmic_rays(spectrum_list: Spectrum, cosmic_ray_pars: dict) -> dict:

    for spectrum in spectrum_list:
        number_spikes = cosmic_ray_pars["spike_num"]
        spike_amplitude = cosmic_ray_pars["spike_amplitude"]
        spikes = np.random.randint(0, len(spectrum.wavenumber), number_spikes)

        for spike in spikes:
            spectrum.intensity[spike] = spectrum.intensity[spike] + spike_amplitude * np.random.random()
    return spectrum_list


def create_baseline(
    spectrum_list: Spectrum, baseline_pars: dict
) -> Spectrum:
    for spectrum in spectrum_list:

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
        
    return spectrum_list

def shift_spectrum(spectrum_list: Spectrum, wavenumber_shift: float) -> Spectrum:
    for spectrum in spectrum_list:
        spectrum.wavenumber = spectrum.wavenumber + wavenumber_shift

    return spectrum_list


def amplify(spectrum_list: Spectrum, spectrum_amplifying_factor: float) -> Spectrum:

    for spectrum in spectrum_list:

        spectrum.intensity = spectrum.intensity * spectrum_amplifying_factor
    
    return spectrum_list


def convolute_kernel(
    spectrum_list: Spectrum, kernel_std: float
) -> Spectrum:
    for spectrum in spectrum_list:
        # Create the kernel
        kernel = signal.windows.gaussian(len(spectrum.wavenumber), kernel_std)
        # Calculate the convolution
        spectrum.intensity = signal.convolve(kernel, spectrum.intensity, mode="same") * sum(kernel)
    return spectrum_list

# interpolate all spectra in the file to the common wavelength range (scipy.interpolate.interp1d(method='bilinear'))
def interpolate_spectrum(spectrum_list:Spectrum , target_raman_shift: np.array) -> Spectrum:

    for spectrum in spectrum_list:
        interpolated_intensity = np.interp(target_raman_shift, spectrum.wavenumber, spectrum.intensity)
        spectrum.wavenumber= target_raman_shift
        spectrum.intensity = interpolated_intensity

    return spectrum_list


def pre_process(spectrum: Spectrum, config: dict) -> np.array:
    # Define the pipeline for preprocessing
    pipe = ramanspy.preprocessing.protocols.Pipeline(
        [
            ramanspy.preprocessing.despike.WhitakerHayes(),
            ramanspy.preprocessing.denoise.SavGol(window_length=12, polyorder=3),
            ramanspy.preprocessing.baseline.ASPLS(),
            ramanspy.preprocessing.normalise.MinMax(pixelwise=True),
        ]
    )
    # Preprocess the spectra with the assembled pipeline
    preprocessed_spectrum = pipe.apply(
        ramanspy.Spectrum(spectrum.intensity, spectrum.wavenumber)
    )
    return preprocessed_spectrum

def wavelength_to_wavenumber(wl:np.array)->np.array:
    """
    Convert wavelength to wavenumber
    
    Parameters:
    wl (np.array): The array of wavelengths.
    
    Returns:
    np.array: The array of wavenumbers.
    """
    # takes a 1-D array of wavelengths(nm) and returns a 1-D array of
    # wavenumber (cm^-1) to perform element-wise multiplication rather than matrix multiplication,
    # use the .* operator. (e.g. .^3 or ./10 will put each element to the third power or divide by 10, respectively). 
    # Copied from Noodle Matlab code

    # first convent from nm to cm
    wlCM = wl*(1e-7)
    # then invert to cm^-1
    wn = wlCM^(-1)

    return wn

def augmention_pars_generator(
    n_dictionaries: int
) -> dict:
    """
    Create a set of augmentation parameters.

    Parameters:
    config (dict): The configuration file.
    n_sets (int): The number of sets to be created.

    Returns:
    dict: The set of augmentation parameters.
    """
    config_path = '/Users/yifeigu/Documents/Carney_Lab/NoodlePy/noodlepy/config/config_test.yml'
    # Load the configuration file
    with open(config_path, "rb") as yaml_file:
        config = yaml.safe_load(yaml_file)

    # Create a set of augmentation parameters
    augmentation_par_dictionaries = {}
    for i_dictionary in range(n_dictionaries):
        # Create a set of augmentation parameters
        augmentation_par_dictionaries[i_dictionary] = {
            "spectrum_amplifying_factor": np.random.uniform(
                config["spectrum_amplifying_factor"]["low"],
                config["spectrum_amplifying_factor"]["high"],
            ),
            "instrument_shift": np.random.uniform(
                config["instrument_shift"]["low"], config["instrument_shift"]["high"]
            ),
            "abbrration_kernel_std": np.random.uniform(
                config["abbrration_kernel_std"]["low"],
                config["abbrration_kernel_std"]["high"],
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
                "spike_num": np.random.uniform(
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

def apply_augmentations(spectrum: Spectrum, augmentation_par_dictionaries: dict) -> Spectrum:
    """
    Apply augmentations to the spectrum.

    Parameters:
    spectrum (Spectrum): The spectrum.
    augment_pars (dict): The augmentation parameters.

    Returns:
    Spectrum: The augmented spectrum.
    """
    # augmented_spectrum_list = []

    for i_augmentation_dictionary in augmentation_par_dictionaries:
        # Amplify the spectrum
        augmented_spectrum = amplify(spectrum, i_augmentation_dictionary["spectrum_amplifying_factor"])
        # Shift the spectrum
        augmented_spectrum = shift_spectrum(augmented_spectrum, i_augmentation_dictionary["instrument_shift"])
        # Apply the convolution kernel
        augmented_spectrum = convolute_kernel(augmented_spectrum, i_augmentation_dictionary["abbrration_kernel_std"])
        # Add photon shot noise
        augmented_spectrum = add_noise(augmented_spectrum, i_augmentation_dictionary["photon_shot_noise_pars"])
        # Add dark current shot noise
        augmented_spectrum = add_noise(augmented_spectrum, i_augmentation_dictionary["dark_current_shot_noise_pars"])
        # Add photo response non-uniformity noise
        augmented_spectrum = add_noise(augmented_spectrum, i_augmentation_dictionary["photo_response_non_uniformity_pars"])
        # Add dark signal FPN noise
        augmented_spectrum = add_noise(augmented_spectrum, i_augmentation_dictionary["dark_signal_FPN_noise_pars"])
        # Add cosmic rays
        augmented_spectrum = add_cosmic_rays(augmented_spectrum, i_augmentation_dictionary["cosmic_ray_pars"])
        # Apply the convolution kernel
        augmented_spectrum = convolute_kernel(augmented_spectrum, i_augmentation_dictionary["aberration_kernel_std"])

        #augmented_spectrum_list.append(augmented_spectrum)

        yield augmented_spectrum