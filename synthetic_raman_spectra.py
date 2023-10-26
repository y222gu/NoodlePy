# Loading the required packages:
import numpy as np
import ramanspy
import matplotlib.pyplot as plt
from scipy import stats
import yaml

def create_mixture_spectrum(
    sample_pars: dict,
    component_spectrum_list: list,
    exp_conditions: dict,
    gaussian_pars: dict
) -> tuple:
    """
    Create a mixture spectrum from individual spectra.

    Parameters:
    exp_conditions (dict): Experiment conditions.
    sample_pars (dict): Sample parameters.
    component_spectrum_list (list of dict): List of component spectra.
    gaussian_pars (dict): Parameters for Gaussian peaks.

    Returns:
    Tuple[np.array, np.array]: The generated spectrum and corresponding wavenumbers.
    """
    mixing_conc = sample_pars["mixing_conc"]
    spectrum_range_pars = exp_conditions["spectrum_range_pars"]
    spectrum_range = np.linspace(*spectrum_range_pars)
    spectrum = np.zeros(len(spectrum_range))

    for i, component_i in enumerate(component_spectrum_list):
        peak_locations = component_i["peak_locations"]
        peak_shapes = [gaussian_pars[peak] for peak in component_i["peak_shapes"]]
        peak_intensities = [gaussian_pars[peak] for peak in component_i["peak_intensities"]]

        for peak_i in range(len(peak_locations)):
            spectrum += (
                mixing_conc[i]
                * stats.norm.pdf(
                    spectrum_range, peak_locations[peak_i], peak_shapes[peak_i]
                )
                * peak_intensities[peak_i]
            )
    spectrum /= np.max(spectrum)  # Normalize the mixture spectrum
    return spectrum, spectrum_range



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

    # add shot noise
    shot_noise_factor = exp_conditions["shot_noise_factor"]
    spectrum = spectrum + np.random.normal(0, shot_noise_factor, len(spectrum))

    # add spikes of cosmic rays:
    cosmic_ray = exp_conditions["cosmic_ray"]
    number_spikes = cosmic_ray["spike_num"]
    spike_amplitude = cosmic_ray["spike_amplitude"]
    spikes = np.random.randint(0, len(spectrum_range), number_spikes)
    for spike in spikes:
        spectrum[spike] = spectrum[spike] + spike_amplitude * np.random.random()

    # adding a baseline with different options: 
    baseline_type = exp_conditions["baseline_type"]
    if baseline_type == "poly":
        baseline_pars = exp_conditions["poly"]
        poly_orders = baseline_pars["poly_orders"]
        poly_pars = baseline_pars["poly_pars"]
        poly_shift = baseline_pars["poly_shift"]
        baseline = 0
        for i in range(poly_orders + 1):
            baseline += poly_pars[i] * (spectrum_range - poly_shift[i]) ** i
    elif baseline_type == "sine":# Sine wave baseline
        baseline_pars = exp_conditions["sine"]
        sine_amplitude = baseline_pars["sine_amplitude"]
        sine_frequency = baseline_pars["sine_frequency"]
        sine_phase = baseline_pars["sine_phase"]
        baseline = sine_amplitude * np.sin(sine_frequency * spectrum_range
                                       + sine_phase) 
    else:
        baseline = np.zeros_like(spectrum_range)  # No baseline


    # shift the whole spectrum
    # constant amplification factor
    # convolution kernel
    # cropping spectrum, interpo, multiple
    # generate random experiment conditions
    # add all above in the 

    spectrum = spectrum + baseline
    return spectrum


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

    colors = config["plot_design"]["colors"]# load color scheme for ploting
    sample_pars = config["sample_pars"]# load concentration of the sample
    component_spectrum_list = [config[component] for component in sample_pars["components"]]# load components spectra
    exp_conditions = config["exp_conditions"]# load spectrum range to be created in wavenumbers
    gaussian_pars = config["gaussian_pars"]# load the pars for gaussian amplitude and sigma
    
    # Create the spectrum of a mixture
    mixture_spectrum, spectrum_range = create_mixture_spectrum(
        sample_pars,
        component_spectrum_list,
        exp_conditions,
        gaussian_pars,
    )
    plot_spectrum(mixture_spectrum, spectrum_range, colors[1], "plot_after_mixing_molecules.pdf")


    # Transform the spectrum
    transformed_spectrum = add_transforms(exp_conditions, spectrum_range, mixture_spectrum)
    plot_spectrum(transformed_spectrum, spectrum_range, colors[1], "plot_after_adding_transforms.pdf")

    # Preprocess the spectra with the assembled pipeline
    preprocessed_spectrum = pipe.apply(ramanspy.Spectrum(transformed_spectrum, spectrum_range))
    ramanspy.plot.spectra(preprocessed_spectrum, color=colors[3], plot_type="stacked")
    plt.savefig("plot_after_preprocessing.pdf", bbox_inches="tight", dpi=300)
    plt.close()

if __name__ == "__main__":
    main()


