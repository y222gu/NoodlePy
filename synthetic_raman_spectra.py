# Loading the required packages:
import numpy as np
import ramanspy
import matplotlib.pyplot as plt
from scipy import stats
from scipy import signal
import yaml
import pandas as pd


def create_mixture_spectrum( 
    component_spectrum_list: list,
    concentrations: np.array,
    exp_conditions: dict,
    gaussian_pars: dict
) -> tuple:
    """
    Create a mixture spectrum from individual spectra.

    Parameters:
    component_spectrum_list (list of dict): List of component spectra.
    concentrations: (np.array): Concentrations of the components.
    exp_conditions (dict): Experiment conditions.
    gaussian_pars (dict): Parameters for Gaussian peaks.

    Returns:
    tuple[np.array, np.array]: The generated spectrum and corresponding wavenumbers.
    """
    spectrum_range_pars = exp_conditions["spectrum_range_pars"]
    spectrum_range = np.linspace(*spectrum_range_pars)
    spectrum = np.zeros(len(spectrum_range))

    for i, component_i in enumerate(component_spectrum_list):
        peak_locations = component_i["peak_locations"]
        peak_shapes = [gaussian_pars[peak] for peak in component_i["peak_shapes"]]
        peak_intensities = [gaussian_pars[peak] for peak in component_i["peak_intensities"]]

        for peak_i in range(len(peak_locations)):
            spectrum += (
                concentrations[i]
                * stats.norm.pdf(
                    spectrum_range, peak_locations[peak_i], peak_shapes[peak_i]
                )
                * peak_intensities[peak_i]
            )
    spectrum /= np.max(spectrum)
    spectrum *= exp_conditions["spectrum_amplifying_factor"]# Normalize the mixture spectrum
    return spectrum, spectrum_range

def add_shot_noise(spectrum: np.array, shot_noise_factor: float)-> np.array:
    """
    Add shot noise to a spectrum.

    Parameters:
    spectrum (np.array): The input spectrum.
    shot_noise_factor (float): The factor determining the amount of shot noise.

    Returns:
    np.array: The spectrum with added shot noise.
    """
    spectrum = spectrum + np.random.normal(0, shot_noise_factor, len(spectrum))
    return spectrum

def add_cosmic_rays(spectrum: np.array, cosmic_ray_pars: dict)-> np.array:
    """
    Add cosmic rays to a spectrum.

    Parameters:
    spectrum (np.array): The input spectrum.
    cosmic_ray_pars (dict): Parameters for cosmic rays.

    Returns:
    np.array: The spectrum with added cosmic rays.
    """
    number_spikes = cosmic_ray_pars["spike_num"]
    spike_amplitude = cosmic_ray_pars["spike_amplitude"]
    spikes = np.random.randint(0, len(spectrum), number_spikes)
    for spike in spikes:
        spectrum[spike] = spectrum[spike] + spike_amplitude * np.random.random()
    return spectrum


def add_baseline(spectrum_range: np.array, spectrum: np.array, exp_conditions: dict)-> np.array:
    """
    Add a baseline to a spectrum.

    Parameters:
    spectrum_range (np.array): The array of wavenumbers.
    spectrum (np.array): The input spectrum.
    exp_conditions (dict): Experiment conditions.

    Returns:
    np.array: The spectrum with the added baseline.
    """
    baseline_type = exp_conditions["baseline_type"]
    baseline_pars = exp_conditions[baseline_type]
    if baseline_type == "poly":
        poly_orders = baseline_pars["poly_orders"]
        poly_pars = baseline_pars["poly_pars"]
        poly_shift = baseline_pars["poly_shift"]
        baseline = 0
        for i in range(poly_orders + 1):
            baseline += poly_pars[i] * (spectrum_range - poly_shift[i]) ** i
    elif baseline_type == "sine":# Sine wave baseline
        sine_amplitude = baseline_pars["sine_amplitude"]
        sine_frequency = baseline_pars["sine_frequency"]
        sine_phase = baseline_pars["sine_phase"]
        baseline = sine_amplitude * np.sin(sine_frequency * spectrum_range
                                       + sine_phase) 
    else:
        baseline = np.zeros_like(spectrum_range)  # No baseline

    spectrum = spectrum + baseline * exp_conditions["baseline_amplifying_factor"]
    return spectrum 

def shift_spectrum(spectrum_range: np.array, shift: float)-> np.array:
    """
    Shift the spectrum's wavenumbers.

    Parameters:
    spectrum_range (np.array): The array of wavenumbers.
    shift (float): The amount to shift the wavenumbers.

    Returns:
    np.array: The shifted wavenumbers.
    """
    spectrum_range = spectrum_range + shift
    return spectrum_range

def amplify_spectrum(spectrum: np.array, amplifying_factor: float)-> np.array:
    """
    Amplify a spectrum.

    Parameters:
    spectrum (np.array): The input spectrum.
    amplifying_factor (float): The factor to amplify the spectrum.

    Returns:
    np.array: The amplified spectrum.
    """
    spectrum = spectrum * amplifying_factor
    return spectrum

def convolute_kernel(spectrum_range: np.array, spectrum: np.array, kernel_std: float)-> np.array:
    """
    Convolute the spectrum with a kernel.

    Parameters:
    spectrum_range (np.array): The array of wavenumbers.
    spectrum (np.array): The input spectrum.
    kernel_std (float): The standard deviation of the kernel.

    Returns:
    np.array: The convoluted spectrum.
    """
    kernel = signal.windows.gaussian(len(spectrum_range), kernel_std)
    # Calculate the convolution
    spectrum = signal.convolve(kernel, spectrum, mode='same') * sum(kernel)
    return spectrum


def crop_spectrum(spectrum_range:np.array, spectrum: np.array, start_wavenumber: np.array, end_wavenumber: np.array)-> tuple[np.array, np.array]:
    """
    Crop a spectrum based on a specified wavelength range.

    Args:
        spectrum_range (np.array): Array of wavenumber values starting with.
        spectrum (np.array): Array of corresponding spectrum values.
        start_wavenumber (np.array): The starting wavenumber for cropping.
        end_wavenumber (np.array): The ending wavenumber for cropping.

    Returns:
        cropped_spectrum_range (np.array): Cropped wavenumber values.
        cropped_spectrum (np.array): Cropped spectrum values.
    """
    # Find the indices corresponding to the start and end wavelengths
    start_index = np.searchsorted(spectrum_range, start_wavenumber)
    end_index = np.searchsorted(spectrum_range, end_wavenumber, side='right')

    # Crop the spectrum based on the specified wavelength range
    cropped_spectrum_range = spectrum_range[start_index:end_index]
    cropped_spectrum = spectrum[start_index:end_index]

    return cropped_spectrum_range, cropped_spectrum


def add_transforms(exp_conditions: dict,
                   spectrum_range: np.array, 
                   spectrum: np.array, 
) -> np.array:
    """
    Transform the spectrum by adding shot noise, cosmic rays, and baseline.

    Parameters:
    exp_conditions (dict): Experiment conditions.
    spectrum_range (np.array): Array of wavenumbers for the spectrum range.
    spectrum (np.array): Array of intensities for the spectrum.
    number_spikes (int): The number of random cosmic ray spikes to add.

    Returns:
    np.array: The transformed spectrum.
    """

    colors = ["#FF5733", "#33FF57", "#3366FF", "#FFFF33"]

    # smear a guassian curve on the spectrum to simulate the laser instability
    kernel_std = exp_conditions["abbreviation_pars"]["kernel_std"]
    spectrum = convolute_kernel(spectrum_range, spectrum, kernel_std)
    plot_spectrum(spectrum, spectrum_range, colors[1], "plot_after_kernel.pdf")

    # adding a baseline with different options: 
    spectrum = add_baseline(spectrum_range, spectrum, exp_conditions)
    plot_spectrum(spectrum, spectrum_range, colors[1], "plot_after_baseline.pdf")

    # add shot noise
    shot_noise_factor = exp_conditions["shot_noise_factor"]
    spectrum = add_shot_noise(spectrum, shot_noise_factor)
    plot_spectrum(spectrum, spectrum_range, colors[1], "plot_after_noise.pdf")
    
    # add spikes of cosmic rays:
    cosmic_ray_pars = exp_conditions["cosmic_ray_pars"]
    spectrum = add_cosmic_rays(spectrum, cosmic_ray_pars)
    plot_spectrum(spectrum, spectrum_range, colors[1], "plot_after_ray.pdf")
    
    # adding shifting from the instrument
    instrument_shift = exp_conditions["instrumment_shift"]
    spectrum_range = shift_spectrum(spectrum_range, instrument_shift)
    plot_spectrum(spectrum, spectrum_range, colors[1], "plot_after_shift.pdf")

    return spectrum_range, spectrum


def load_metabolomics(file_path: str, sample_type: str)-> tuple[list, np.array]:
    """
    Load metabolomics data from an Excel file.

    Parameters:
    file_path (str): The path to the Excel file.
    sample_type (str): The type of sample.

    Returns:
    tuple[list, np.array]: The list of components and the corresponding concentrations.   
    """
    # Load the XLSX file into a DataFrame
    df = pd.read_excel(file_path)
    # Extract the column as a NumPy array
    if sample_type == 'Saliva':
        components = df['label']
        concentration = df['Saliva'].to_numpy()
    elif sample_type == 'Plasma':
        components = df['label']
        concentration = df['Plasma'].to_numpy()
    elif sample_type == 'Serum':
        components = df['label']
        concentration = df['Serum'].to_numpy()
    else:
        raise ValueError('sample_type must be either Saliva, Plasma or Serum')
    
    components = components[~np.isnan(concentration)]
    concentration = concentration[~np.isnan(concentration)]
    components = [x for _, x in sorted(zip(concentration, components), reverse=True)]
    concentration = sorted(concentration, reverse=True)   

    return components, concentration


# Assemble the pipeline for preprocessing using RamanSpy
pipe = ramanspy.preprocessing.protocols.Pipeline(
    [
        ramanspy.preprocessing.despike.WhitakerHayes(),
        ramanspy.preprocessing.denoise.SavGol(window_length=12, polyorder=3),
        ramanspy.preprocessing.baseline.ASPLS(),
        ramanspy.preprocessing.normalise.MinMax(pixelwise=True),
    ]
)

def plot_spectrum(spectrum: np.array, spectrum_range: np.array, color: str, filename: str):
    """
    Plot and save a spectrum to a file.

    Parameters:
    spectrum (np.array): The spectrum to be plotted.
    spectrum_range (np.array): The corresponding wavenumbers.
    color (str): The color for the plot.
    filename (str): The filename for the saved plot.
    """
    ramanspy.plot.spectra(
        ramanspy.Spectrum(spectrum, spectrum_range),
        color=color,
        plot_type="stacked",
    )
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()


def main():

    # Initiate constant variables
    with open("config.yml", "rb") as yaml_file:
        config = yaml.safe_load(yaml_file)
    components, concentrations = load_metabolomics("metabolomics.xlsx", "Saliva")
    
    colors = config["plot_design"]["colors"]# load color scheme for ploting
    component_spectrum_list = [config[component] for component in components]# load components spectra
    exp_conditions = config["exp_conditions"]# load spectrum range to be created in wavenumbers
    gaussian_pars = config["gaussian_pars"]# load the pars for gaussian amplitude and sigma
    
    # Create the spectrum of a mixture
    mixture_spectrum, spectrum_range = create_mixture_spectrum(
        component_spectrum_list,
        concentrations,
        exp_conditions,
        gaussian_pars,
    )
    plot_spectrum(mixture_spectrum, spectrum_range, colors[1], "plot_after_mixing_molecules.pdf")


    # Transform the spectrum
    transformed_spectrum_range, transformed_spectrum = add_transforms(exp_conditions, spectrum_range, mixture_spectrum)
    plot_spectrum(transformed_spectrum, transformed_spectrum_range, colors[1], "plot_after_adding_transforms.pdf")

    # Preprocess the spectra with the assembled pipeline
    preprocessed_spectrum = pipe.apply(ramanspy.Spectrum(transformed_spectrum, spectrum_range))
    ramanspy.plot.spectra(preprocessed_spectrum, color=colors[3], plot_type="stacked")
    plt.savefig("plot_after_preprocessing.pdf", bbox_inches="tight", dpi=300)
    plt.close()

if __name__ == "__main__":
    main()


