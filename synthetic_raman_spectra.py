# Loading the required packages:
import numpy as np
import ramanspy
import matplotlib.pyplot as plt
from scipy import stats
import tomllib

def create_mixture_spectrum(
    sample_pars: dict,
    component_spectrum_list: dict,
    exp_conditions: dict,
    gaussian_pars: dict,
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


def main():

    # Initiate constant variables
    with open("config.toml", "rb") as toml_file:
        config = tomllib.load(toml_file)

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


