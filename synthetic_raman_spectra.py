# Loading the required packages:
import numpy as np
import ramanspy
import matplotlib.pyplot as plt
from scipy import stats
import tomllib

def create_mixture_spectrum(
    spectrum_range: np.array,
    peak_location_list: list,
    peak_shape_list: list,
    peak_intensity_list: list,
    mixing_conc: np.array,
) -> np.array:
    """
    This function creates a mixture spectrum from multiple individual spectra.

    Parameters:
    spectrum_range_list: np.array
        Array of wavenumber for the spectrum range of the pristine spectrum.
    peak_location_list: list of np.array
        List of peak locations for each individual spectrum.
    peak_shape_list: list of np.array
        List of peak shapes for each individual spectrum.
    peak_intensity_list: list of np.array
        List of peak intensities for each individual spectrum.
    mixing_conc: np.array
        Array of concentrations of each individual spectrum in the mixture spectrum.

    Returns:
    spectrum: np.array
        The spectrum of the mixture.
    """

    num_spectra = len(mixing_conc)
    num_points = len(spectrum_range)
    spectrum = np.zeros(num_points)

    for i in range(num_spectra):
        peak_location = peak_location_list[i]
        peak_shape = peak_shape_list[i]
        peak_intensity = peak_intensity_list[i]

        for peak_i in np.arange(len(peak_location)):
            spectrum += (
                mixing_conc[i]
                * stats.norm.pdf(
                    spectrum_range, peak_location[peak_i], peak_shape[peak_i]
                )
                * peak_intensity[peak_i]
            )

    # Normalize the mixture spectrum
    spectrum = spectrum / np.max(spectrum)

    return spectrum


def add_transforms(
    spectrum_range: np.array, spectrum: np.array, number_spikes: int = None
) -> np.array:
    """
    This function transforms the spectrum by adding shot noise, cosmic rays, polynomial fluorescence background.

    Parameters:
    spectrum_range: np.array
        Array of wavenumbers for the spectrum range of the spectrum.
    spectrum: np.array
        Array of intensities for the spectrum.
    number_spikes: int
        The number of random cosmic ray spikes to add.

    Returns:
    spectrum: np.array
        The transformed spectrum.
    """

    # add shot noise
    spectrum = spectrum + np.random.normal(0, 0.06, len(spectrum))

    # add spikes of cosmic rays:
    spikes = np.random.randint(0, len(spectrum_range), number_spikes)
    for spike in spikes:
        spectrum[spike] = spectrum[spike] + 2 * np.random.random()

    # adding a polynomial baseline as fluorescence background:
    poly = (
        0.03 * np.ones(len(spectrum_range))
        + 0.00003 * spectrum_range
        + 0.000003 * (spectrum_range - 680) ** 2
    )
    spectrum = spectrum + poly

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

'''
def prep_spectrum():
    # Initiate constant variables
    with open("config.toml", "rb") as toml_file:
        config = tomllib.load(toml_file)

    # signal_intensity_level
    gaussian_amplitude_pars = config["signal_intensity_pars"]
    w = gaussian_amplitude_pars["w"]
    mw = gaussian_amplitude_pars["mw"]
    m = gaussian_amplitude_pars["m"]
    ms = gaussian_amplitude_pars["ms"]
    s = gaussian_amplitude_pars["s"]
    vs = gaussian_amplitude_pars["vs"]
    # sigma
    gaussian_sigma_pars = config["gaussian_sigma_pars"]
    n = gaussian_sigma_pars["norm"]
    sh = gaussian_sigma_pars["sh"]
    br = gaussian_sigma_pars["br"]

    # Create a mapping from strings to numbers using dictionary comprehension
    string_to_number = {string: num for num, string in enumerate(set(str_array))}

    # Convert the array of strings to an array of numbers using list comprehension
    num_array = [string_to_number[string] for string in str_array]
    return
'''

def main():

    # Initiate constant variables
    with open("config.toml", "rb") as toml_file:
        config = tomllib.load(toml_file)

    # signal_intensity_level
    gaussian_amplitude_pars = config["signal_intensity_pars"]
    w = gaussian_amplitude_pars["w"]
    mw = gaussian_amplitude_pars["mw"]
    m = gaussian_amplitude_pars["m"]
    ms = gaussian_amplitude_pars["ms"]
    s = gaussian_amplitude_pars["s"]
    vs = gaussian_amplitude_pars["vs"]
    # sigma
    gaussian_sigma_pars = config["gaussian_sigma_pars"]
    n = gaussian_sigma_pars["n"]
    sh = gaussian_sigma_pars["sh"]
    br = gaussian_sigma_pars["br"]
    # color scheme for ploting
    colors = config["plot_design"]["colors"]
    # spectrum range to be created in wavenumbers
    spectrum_range_pars = config["experiment_conditions"]["spectrum_range_pars"]
    spectrum_range = np.linspace(spectrum_range_pars[0], spectrum_range_pars[1], spectrum_range_pars[2])
    # concentration of the components of the sample
    mixing_conc = config["sample_info"]["mixing_conc"]

    # Load Stored data
    # valine
    peak_location_valine = config["valine"]["peak_locations"]
    peak_intensity_valine = config["valine"]["peak_intensities"]
    peak_shape_valine = config["valine"]["peak_shapes"]
    # histidine
    peak_location_histidine = config["histidine"]["peak_locations"]
    peak_intensity_histidine = config["histidine"]["peak_intensities"]
    peak_shape_histidine = config["histidine"]["peak_shapes"]
    # tryptophan
    peak_location_tryptophan = config["tryptophan"]["peak_locations"]
    peak_intensity_tryptophan = config["tryptophan"]["peak_intensities"]
    peak_shape_tryptophan = config["tryptophan"]["peak_shapes"]


    # Make all components into a list
    peak_location_list = [
        peak_location_valine,
        peak_location_histidine,
        peak_location_tryptophan,
    ]
    peak_intensity_list = [
        peak_intensity_valine,
        peak_intensity_histidine,
        peak_intensity_tryptophan,
    ]
    peak_shape_list = [peak_shape_valine, peak_shape_histidine, peak_shape_tryptophan]



    # Create the spectrum of a mixture
    mixture_spectrum = create_mixture_spectrum(
        spectrum_range,
        peak_location_list,
        peak_shape_list,
        peak_intensity_list,
        mixing_conc,
    )
    ramanspy.plot.spectra(
        ramanspy.Spectrum(mixture_spectrum, spectrum_range),
        color=colors[1],
        plot_type="stacked",
    )
    plt.savefig("plot_after_mixing_molecules.pdf", bbox_inches="tight", dpi=300)
    plt.close()

    # Transform the spectrum
    transformed_spectrum = add_transforms(spectrum_range, mixture_spectrum, 3)

    # Wrap spectrum into a ramanspy container
    wrapped_spectrum = ramanspy.Spectrum(transformed_spectrum, spectrum_range)
    ramanspy.plot.spectra(wrapped_spectrum, color=colors[1], plot_type="stacked")
    plt.savefig("plot_after_adding_transforms.pdf", bbox_inches="tight", dpi=300)
    plt.close()

    # Preprocess the spectra with the assembled pipeline
    preprocessed_spectrum = pipe.apply(wrapped_spectrum)
    ramanspy.plot.spectra(preprocessed_spectrum, color=colors[3], plot_type="stacked")
    plt.savefig("plot_after_preprocessing.pdf", bbox_inches="tight", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()


