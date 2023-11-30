# Loading the required packages:
import numpy as np
import ramanspy
import matplotlib.pyplot as plt
from scipy import stats
from scipy import signal
import yaml
import pandas as pd
import pickle
import seaborn as sns

def create_spectrum(
    metabolite_spectrum_dict: list,
    concentrations: np.array,
    config: dict,
    gaussian_pars: dict,
) -> tuple:
    """
    Create a pristine spectrum base on the leaterature data.

    Parameters:
    metabolite_spectrum_dict (dict): List of component spectra.
    concentrations: (np.array): Concentrations of the components.
    config (dict): Experiment conditions.
    gaussian_pars (dict): The parameters for the gaussian distribution.

    Returns:
    tuple[np.array, np.array]: The generated spectrum and corresponding wavenumbers.
    """
    raman_shift_pars = config["raman_shift_pars"]
    raman_shift_range = np.linspace(*raman_shift_pars)
    spectrum = np.zeros(len(raman_shift_range))

    for i, component_i in enumerate(metabolite_spectrum_dict):
        peak_locations = component_i["peak_locations"]
        peak_shapes = [gaussian_pars[peak] for peak in component_i["peak_shapes"]]
        peak_intensities = [
            gaussian_pars[peak] for peak in component_i["peak_intensities"]
        ]

        for peak_i in range(len(peak_locations)):
            spectrum += (
                concentrations[i]
                * stats.norm.pdf(
                    raman_shift_range, peak_locations[peak_i], peak_shapes[peak_i]
                )
                * peak_intensities[peak_i]
            )
    spectrum /= np.max(spectrum)
    spectrum *= config[
        "spectrum_amplifying_factor"
    ]  # Normalize the mixture spectrum
    return spectrum, raman_shift_range

def load_metabolites(config: dict) -> tuple[list, np.array]:
    """
    Load metabolomics names and concentrations from an Excel file.

    Parameters:
    file_path (str): The path to the Excel file.
    sample_type (str): The type of sample.

    Returns:
    tuple[list, np.array]: The list of metabolomics and the corresponding concentrations.
    """
    # Load the XLSX file into a DataFrame
    sample_type = config["sample_type"]
    file_path = config["metabolomics_file_path"]

    # Load the XLSX file into a DataFrame
    df = pd.read_excel(file_path)
    # Extract the column as a NumPy array
    if sample_type == "Saliva":
        metabolite_name_list = df["label"]
        concentration = df["Saliva"].to_numpy()
    elif sample_type == "Plasma":
        metabolite_name_list = df["label"]
        concentration = df["Plasma"].to_numpy()
    elif sample_type == "Serum":
        metabolite_name_list = df["label"]
        concentration = df["Serum"].to_numpy()
    else:
        raise ValueError("sample_type must be either Saliva, Plasma or Serum")

    metabolite_name_list = metabolite_name_list[~np.isnan(concentration)]
    concentration = concentration[~np.isnan(concentration)]
    metabolite_name_list = [x for _, x in sorted(zip(concentration, metabolite_name_list), reverse=True)]
    concentration = sorted(concentration, reverse=True)

    return metabolite_name_list, concentration

def load_spectra(metabolite_name_list: list, config: dict) -> dict:
    """
    Load the spectra database from a pickle file.

    Parameters:
    file_name (str): The path to the pickle file.

    Returns:
    dict: The spectra database.
    """
    print(f'Start to load spectra from the pickle file...')

    # Load the Pickle file
    file_name = config["DFT_file_path"]

    # Load data from the pickle file
    with open(file_name, 'rb') as file:
        database = pickle.load(file)

    # take an array out of the list
    metabolite_name_list = np.array(metabolite_name_list)
 
   # Extract values for the specified molecules
    spectrum_dict = {}
    for metabolite_name in metabolite_name_list:
        metabolite_data = database.get(metabolite_name, None)
        if metabolite_data:
            spectrum = extract_spectrum(metabolite_data, config
    )
            spectrum_dict[metabolite_name] = spectrum
        else:
            print(f'Molecule "{metabolite_name}" not found in the pickle file. Skipping...')

    print(f'Loding spectra from the pickle file is done...')

    return spectrum_dict


    # Define a function to extract values
def extract_spectrum(metabolite_data, config):
    # Define the intensity generated at specific laser wavelength to be used
    intensity_to_use = "intensity_" + str(config["laser_wavelength"])
    raman_shift = np.array(metabolite_data['freq'])
    intensity = metabolite_data[intensity_to_use]
    spectrum = {
        'raman_shift': raman_shift,
        'intensity': intensity
    }
    return spectrum

def crop_spectrum(spectrum: dict, start_wavenumber: float, end_wavenumber: float) -> dict:
    """
    Crop a spectrum based on a specified wavelength range.

    Args:
        raman_shift_range (np.array): Array of wavenumber values starting with.
        spectrum (np.array): Array of corresponding spectrum values.
        start_wavenumber (np.array): The starting wavenumber for cropping.
        end_wavenumber (np.array): The ending wavenumber for cropping.

    Returns:
        cropped_raman_shift_range (np.array): Cropped wavenumber values.
        cropped_spectrum (np.array): Cropped spectrum values.
    """
    # Find the indices corresponding to the start and end wavelengths
    start_index = np.where(spectrum["raman_shift"]==start_wavenumber)[0][0]
    end_index = np.where(spectrum["raman_shift"]==end_wavenumber)[0][0]

    # Crop the spectrum based on the specified wavelength range
    cropped_raman_shift = spectrum["raman_shift"][start_index:end_index]
    cropped_intensity = spectrum["intensity"][start_index:end_index]
    cropped_spectrum = {
        "raman_shift":cropped_raman_shift, 
        "intensity":cropped_intensity
    }
    return cropped_spectrum

def align_spectra(
    spectrum_dict: dict,
    config: dict,
) -> dict:
    """
    Crop a spectrum based on a specified wavelength range.

    Args:
        raman_shift_range (np.array): Array of wavenumber values starting with.
        spectrum (np.array): Array of corresponding spectrum values.
        start_wavenumber (np.array): The starting wavenumber for cropping.
        end_wavenumber (np.array): The ending wavenumber for cropping.

    Returns:
        cropped_raman_shift_range (np.array): Cropped wavenumber values.
        cropped_spectrum (np.array): Cropped spectrum values.
    """
    print(f'Start to align all spectra to the specified range...')
    print(f'Checking if the raman shift range of molecules are matching')

    raman_shift_range_pars = config["raman_shift_range_pars"]
    start_wavenumber = raman_shift_range_pars[0]
    end_wavenumber = raman_shift_range_pars[1]

    aligned_spectrum_dict={}

    for name in spectrum_dict:
        raman_shift = spectrum_dict[name]["raman_shift"]

        # test if the start_wavenumber exist in the raman shift

        if start_wavenumber in raman_shift and end_wavenumber in raman_shift:
            aligned_spectrum_dict[name] = crop_spectrum(spectrum_dict[name], start_wavenumber, end_wavenumber)

        else:
            print(f'Molecule {name} Raman shift range could not be aligned to the specified range. Removing from the spectrum dict...')
    
    print(f'All spectra are aligned and cropped between Wavenumber {start_wavenumber} to {end_wavenumber}')

    return aligned_spectrum_dict

def normalize_spectrum(
    spectrum_dict: dict,
    normalization_option: str,
) -> dict:
    """
    Normalize a spectrum.

    Args:
        spectrum (np.array): Array of corresponding spectrum values.
        normalization_option (str): The normalization method.

    Returns:
        normalized_spectrum (np.array): Normalized spectrum values.
    """
    print(f'Normalizing spectra by: ', normalization_option, '...')

    for name in spectrum_dict:
       
        # intensity
        intensity = spectrum_dict[name]["intensity"]

        # normalize by the area under the curve:
        if normalization_option == 'area':
            intensity /= np.trapz(intensity, spectrum_dict[name]["raman_shift"])
        # normalize by the maximum intensity:
        elif normalization_option == 'max':
            intensity /= np.max(intensity)
        else:
            raise ValueError(f'Mormalization method is not defined')
        
        spectrum_dict[name]["intensity"] = intensity

    return spectrum_dict


def mix_spectra(
    spectrum_dict: dict,
    concentrations: np.array,
    config: dict,
) -> dict:
    """
    Create a mixture spectrum from individual spectra.

    Parameters:
    component_spectrum_list (list of dict): List of component spectra.
    concentrations: (np.array): Concentrations of the components.
    config (dict): Experiment conditions.

    Returns:
    tuple[np.array, np.array]: The generated spectrum and corresponding wavenumbers.
    """
    print(f'Start to mix spectra...')
    mixture_spectrum_dict = {}

    for i, name in enumerate(spectrum_dict):
        if i == 0:
            raman_shift = spectrum_dict[name]["raman_shift"]

            intensity = spectrum_dict[name]["intensity"]
            mixture_intensity = (concentrations[i] * intensity)

            print(f'Molecule "{name}" is added to the mixture...')

        else:
            # double check if the raman_shift_range is matching
            if np.array_equal(spectrum_dict[name]["raman_shift"], raman_shift):
                raman_shift = spectrum_dict[name]["raman_shift"]

                intensity = spectrum_dict[name]["intensity"]
                mixture_intensity += (concentrations[i] * intensity)

                print(f'Molecule "{name}" is added to the mixture...')
            
            else:
                raise ValueError(f'Raman shift range of "{name}" does not match with added metabolites')
            
        mixture_spectrum_dict[name] = {"raman_shift": raman_shift, "intensity": mixture_intensity}

    print(f'Mixing is done...')
   

    return mixture_spectrum_dict


def add_noise(spectrum: dict, noise_pars: dict) -> dict:
    """
    Add shot noise to a spectrum.

    Parameters:
    spectrum (np.array): The input spectrum.
    shot_noise_factor (float): The factor determining the amount of shot noise.

    Returns:
    np.array: The spectrum with added shot noise.
    """
    # TODO: Check if shot noise should be a strictly positive distribution.  Maybe Poisson
    # TODO: Make different sources of noise explicit
    # intensity
    intensity = spectrum["intensity"]
    noise_type = noise_pars["noise_type"]

    if noise_type == "poisson":
        intensity += np.random.poisson(noise_pars["lam"], len(intensity))

    elif noise_type == "gaussian":
        intensity += np.random.normal(noise_pars["mean"], noise_pars["gaussian"]["std"], len(intensity))

    elif noise_type == "uniform":
        intensity += np.random.uniform(noise_pars["low"],noise_pars["uniform"]["high"],len(intensity))

    elif noise_type == "exponential":
        intensity += np.random.exponential(noise_pars["mean"], len(intensity))

    elif noise_type == "lognormal":
        intensity += np.random.lognormal(noise_pars["mean"],noise_pars["lognormal"]["sigma"],len(intensity))
    else:
        raise ValueError("noise_type must be one among 'poisson', 'gaussian', 'uniform', 'expoenetial', and 'lognormal'")
    
    spectrum["intensity"] = intensity

    return spectrum


def add_cosmic_rays(spectrum: dict, cosmic_ray_pars: dict) -> dict:
    """
    Add cosmic rays to a spectrum.

    Parameters:
    spectrum (np.array): The input spectrum.
    cosmic_ray_pars (dict): Parameters for cosmic rays.

    Returns:
    np.array: The spectrum with added cosmic rays.
    """
    # intensity
    intensity = spectrum["intensity"]
    number_spikes = cosmic_ray_pars["spike_num"]
    spike_amplitude = cosmic_ray_pars["spike_amplitude"]
    spikes = np.random.randint(0, len(spectrum), number_spikes)

    for spike in spikes:
        intensity[spike] = intensity[spike] + spike_amplitude * np.random.random()
    spectrum["intensity"] = intensity
    return spectrum


def create_baseline(
    spectrum: dict, config: dict
) -> dict:
    """
    Add a baseline to a spectrum.

    Parameters:
    spectrum_range (np.array): The array of wavenumbers.
    spectrum (np.array): The input spectrum.
    config (dict): Experiment conditions.

    Returns:
    np.array: The spectrum with the added baseline.
    """
    # Raman shift
    raman_shift = spectrum["raman_shift"]

    baseline_type = config["baseline_type"]
    baseline_pars = config[baseline_type]

    if baseline_type == "poly":
        poly_orders = baseline_pars["poly_orders"]
        poly_coefficients = baseline_pars["poly_coefficients"]
        poly_displacement = baseline_pars["poly_displacement"]

        baseline_intensity = 0
        for i in range(poly_orders + 1):
            baseline_intensity += poly_coefficients[i] * (raman_shift - poly_displacement[i]) ** i

    elif baseline_type == "sine":  # Sine wave baseline
        sine_amplitude = baseline_pars["sine_amplitude"]
        sine_frequency = baseline_pars["sine_frequency"]
        sine_phase = baseline_pars["sine_phase"]
        baseline_intensity = sine_amplitude * np.sin(sine_frequency * raman_shift + sine_phase)

    else:
        baseline_intensity = np.zeros_like(raman_shift)  # No baseline

    # Add the baseline to the spectrum
    # intensity = intensity + baseline * config["baseline_amplifying_factor"]
    # spectrum["intensity"] = intensity
    baseline = {"raman_shift": raman_shift ,"intensity": baseline_intensity}
    return baseline


def shift_spectrum(spectrum: dict, shift: float) -> dict:
    """
    Shift the spectrum's wavenumbers.

    Parameters:
    spectrum_range (np.array): The array of wavenumbers.
    shift (float): The amount to shift the wavenumbers.

    Returns:
    np.array: The shifted wavenumbers.
    """
    spectrum["raman_shift"] = spectrum["raman_shift"] + shift
    return spectrum

# define a function to amplify signal with nan as the default input for the baseline


def amplify_signal(spectrum: dict, spectrum_amplifying_factor: float, baseline: dict, baseline_amplifying_factor: float) -> dict:
    
    """
    Amplify a spectrum.

    Parameters:
    spectrum (np.array): The input spectrum.
    amplifying_factor (float): The factor to amplify the spectrum.

    Returns:
    np.array: The amplified spectrum.
    """
    print(f'Amplifying factor applied to the spectrum signal but not the noise yet...')

    spectrum["intensity"] = spectrum["intensity"] * spectrum_amplifying_factor + baseline["intensity"]* baseline_amplifying_factor
    return spectrum


def convolute_kernel(
    spectrum: dict, kernel_std: float
) -> dict:
    """
    Convolute the spectrum with a kernel.

    Parameters:
    spectrum_range (np.array): The array of wavenumbers.
    spectrum (np.array): The input spectrum.
    kernel_std (float): The standard deviation of the kernel.

    Returns:
    np.array: The convoluted spectrum.
    """
    # raman shift
    raman_shift = spectrum["raman_shift"]
    # intensity
    intensity = spectrum["intensity"]

    # Create the kernel
    kernel = signal.windows.gaussian(len(raman_shift), kernel_std)
    # Calculate the convolution
    intensity = signal.convolve(kernel, intensity, mode="same") * sum(kernel)
    # update the raman_shift in the spectrum
    spectrum["intensity"] = intensity

    return spectrum

def add_transforms(
    spectrum_dict: dict,
    config: dict
) -> dict:
    """
    Transform the spectrum by adding shot noise, cosmic rays, and baseline.

    Parameters:
    config (dict): Experiment conditions.
    raman_shift_range (np.array): Array of wavenumbers for the spectrum range.
    spectrum (np.array): Array of intensities for the spectrum.
    number_spikes (int): The number of random cosmic ray spikes to add.

    Returns:
    np.array: The transformed spectrum.
    """
    transformed_spectrum_dict = {}
    for spectrum_name in spectrum_dict:
        
        # get the spectrum
        spectrum = spectrum_dict[spectrum_name]

        # smear a guassian curve on the spectrum to simulate the laser instability
        kernel_std = config["abbreviation_pars"]["kernel_std"]
        spectrum = convolute_kernel(spectrum, kernel_std)

        # adding a baseline with different options:
        baseline = create_baseline(spectrum, config)
        spectrum = amplify_signal(spectrum, config["spectrum_amplifying_factor"],baseline, config["baseline_amplifying_factor"])

        # add shot noise
        shot_noise_pars = config["shot_noise_pars"]
        spectrum = add_noise(spectrum, shot_noise_pars)

        # add dark current noise
        # add camera readout noise

        # add spikes of cosmic rays:
        cosmic_ray_pars = config["cosmic_ray_pars"]
        spectrum = add_cosmic_rays(spectrum, cosmic_ray_pars)

        # adding shifting from the instrument
        instrument_shift = config["instrumment_shift"]
        spectrum = shift_spectrum(spectrum, instrument_shift)

        # write the spectrum into a dictionary
        transformed_spectrum_dict[spectrum_name] = spectrum

    return transformed_spectrum_dict

def pre_process(spectrum: dict, config: dict) -> np.array:
    """
    Preprocess a spectrum.

    Parameters:
    spectrum (np.array): The input spectrum.
    config (dict): Experiment conditions.

    Returns:
    np.array: The preprocessed spectrum.
    """
    # raman shift
    raman_shift = spectrum["raman_shift"]
    # intensity
    intensity = spectrum["intensity"]

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
        ramanspy.Spectrum(intensity, raman_shift)
    )
    return preprocessed_spectrum

def plot_spectrum(
    spectrum: dict, filename: str
):
    """
    Plot and save a spectrum to a file.

    Parameters:
    spectrum (np.array): The spectrum to be plotted.
    spectrum_range (np.array): The corresponding wavenumbers.
    color (str): The color for the plot.
    filename (str): The filename for the saved plot.
    """

    """
    Creates a 2x2 grid of subplots with different types of plots.
    """
    raman_shift = spectrum["raman_shift"]
    intensity = spectrum["intensity"]

    # line plot
    sns.lineplot(x=raman_shift, y=intensity)

    # show subplot number on top of each subplot
    # ax.text(0, 1.15, "C", fontsize=16, transform=ax["C"].transAxes)

    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()

def wavelengthToWavenumber(wl:np.array)->np.array:
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


def main():
    # Initiate constant variables
    with open("config_test.yml", "rb") as yaml_file:
        config = yaml.safe_load(yaml_file)

    # load metabolomics names and concentrations
    metabolite_name_list, concentrations = load_metabolites(config)

    # load metabolomic spectra from DFT database
    metabolite_spectrum_dict = load_spectra(metabolite_name_list, config)
    
    # crop all expectra to the same range
    metabolite_spectrum_dict = align_spectra(metabolite_spectrum_dict, config)

    # normalize all spectra individually
    metabolite_spectrum_dict = normalize_spectrum(metabolite_spectrum_dict, config["normalization_option_for_individual_spectrum"])

    # Create the spectrum of a mixture
    mixture_spectrum = mix_spectra(
        metabolite_spectrum_dict,
        concentrations,
        config,
    )

    # Normalize the mixture spectrum
    mixture_spectrum_dict = normalize_spectrum(mixture_spectrum, config["normalization_option_for_mixture_spectrum"])

    # Transform the spectrum
    transformed_spectrum_dict = add_transforms(
        mixture_spectrum_dict, config
    )

    with plt.style.context('seaborn-v0_8-colorblind'):
        plot_spectrum(transformed_spectrum_dict, "transformed_spectrum.png")

    # Preprocess the spectra with the assembled pipeline
    preprocessed_spectrum = pre_process(transformed_spectrum_dict, config)


if __name__ == "__main__":
    main()


