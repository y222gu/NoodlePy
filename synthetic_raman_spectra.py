# TODO:One function for generating spectra with different concentrations out of a list of molecular spectra
# TODO:One function augment the spectra
# TODO:add a proper main when run the script
# TODO:CHANGE VARIBLES TO GLOBAL VARIBALES TO ALL CAPITALS
# TODO: add a convolution to simulate the laser


# Loading the required packages:
import numpy as np
import ramanspy
import matplotlib.pyplot as plt
from scipy import stats

def create_mixture_spectrum(spectrum_range: np.array, 
                            peak_location_list: list, 
                            peak_shape_list: list, 
                            peak_intensity_list: list, mixing_conc:np.array) -> np.array:
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
    num_points = len(spectrum_range[0])
    spectrum = np.zeros(num_points)

    for i in range(num_spectra):
        peak_location = peak_location_list[i]
        peak_shape = peak_shape_list[i]
        peak_intensity = peak_intensity_list[i]

        for peak_i in np.arange(len(peak_location)):
            spectrum += mixing_conc[i]*stats.norm.pdf(spectrum_range, peak_location[peak_i], peak_shape[peak_i]) * peak_intensity[peak_i]

    # Normalize the mixture spectrum
    # spectrum = spectrum / np.max(spectrum)

    return spectrum


def add_transforms(spectrum_range: np.array,
              spectrum: np.array,
              number_spikes: int) -> np.array:

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
    spectrum = spectrum + np.random.normal(0,0.4,len(spectrum))

    # add spikes of cosmic rays: 
    spikes = np.random.randint(0, len(spectrum_range),number_spikes)
    for spike in spikes:
        spectrum[spike] = spectrum[spike] + 10*np.random.random()

    # adding a polynomial baseline as fluorescence background:
    poly = 0.05 * np.ones(len(spectrum_range)) + 0.00005 * spectrum_range + 0.000005 * (spectrum_range - 680)**2 
    spectrum = spectrum + poly

    return spectrum

# Assemble the pipeline for preprocessing using RamanSpy
pipe = ramanspy.preprocessing.protocols.Pipeline([
    ramanspy.preprocessing.despike.WhitakerHayes(),
    ramanspy.preprocessing.denoise.SavGol(window_length=12, polyorder=3),
    ramanspy.preprocessing.baseline.ASPLS(),
    ramanspy.preprocessing.normalise.MinMax(pixelwise=True),
])

def main(spectrum_range, mixing_conc):
    # Initiate constant variables
    # color scheme for ploting
    colors = plt.cm.get_cmap()(np.linspace(0, 1, 4))

    # signal_intensity_level
    w=0.05
    mw=0.1
    m=0.2
    ms=0.25
    s=0.5
    vs=1

    # sigma
    sh=10
    br=15

    # Stored database
    # valine
    peak_location_valine = np.array([374, 396, 429, 472, 497, 542, 664, 715, 753, 776, 824, 849, 891, 902, 923, 948, 964, 1029, 1035, 1066, 1106, 1125, 1146, 1179, 1191, 1272, 1321, 1330, 1343, 1351, 1398,1427, 1452, 1508, 1567,1587, 1619, 1633, 1660])
    peak_intensity_valine = np.array([mw, mw, mw, w, w, vs, mw, m, m, s, m, s, w, m, mw, s, m, mw, mw, mw, w, m, mw, mw, m, m, m, m, s, s, m, m, s, mw, w, w, mw, mw, w])
    peak_shape_valine = 5*np.ones(len(peak_location_valine))
    peak_shape_valine[12]=sh
    peak_shape_valine[14]=sh
    peak_shape_valine[26]=sh
    peak_shape_valine[28]=sh
    # histidine
    peak_location_histidine=np.array([404, 422, 540, 623,656, 680, 731, 784, 804, 824,852, 918, 929, 963, 976, 1061, 1087, 1111, 1140, 1174,1224, 1250, 1271, 1317, 1336,1347, 1407, 1430, 1476, 1498,1538, 1571, 1608, 1639])
    peak_intensity_histidine=np.array([m, mw, mw, mw, m, w, mw, mw, m, mw, m, m, mw, m, m, m, s, m, mw, m, m, m, s, vs, m, m, m, m, mw, m, w, m, w, w])
    peak_shape_histidine=5*np.ones(len(peak_location_histidine))
    peak_shape_histidine[1]=sh
    peak_shape_histidine[12]=sh
    # tryptophan
    peak_location_tryptophan = np.array([393, 425, 456, 498, 509,534, 548, 574, 596, 626, 683, 706,741, 755, 766, 778, 802, 840, 848, 865,874, 988, 1009, 1046,1076, 1103, 1118, 1160, 1207, 1231, 1253, 1278, 1309, 1314, 1328, 1338, 1358, 1423, 1450,1457, 1486, 1556, 1576, 1616])
    peak_intensity_tryptophan = np.array([w, w, w, m, m, m, w, m, m, m, w, m, m, vs, m, m,w, m, m, m,s, w, vs, w,w, w, m, w, w, m, w, w, w, m,m, s, s, s, m,m, m, s, m, m])
    peak_shape_tryptophan=5*np.ones(len(peak_location_tryptophan))

    # Make all components into a list
    peak_location_list = [peak_location_valine, peak_location_histidine, peak_location_tryptophan]
    peak_intensity_list = [peak_intensity_valine, peak_intensity_histidine, peak_intensity_tryptophan]
    peak_shape_list = [peak_shape_valine, peak_shape_histidine, peak_shape_tryptophan]

    # Create the spectrum of a mixture
    mixture_spectrum = create_mixture_spectrum(spectrum_range, peak_location_list, peak_shape_list, peak_intensity_list, mixing_conc)
    ramanspy.plot.spectra(ramanspy.Spectrum(mixture_spectrum, spectrum_range), color=colors[1], plot_type='stacked')
    plt.savefig('plot_after_mixing_molecules.pdf', bbox_inches='tight', dpi=300)
    plt.close()

    # Transform the spectrum
    transformed_spectrum = add_transforms(spectrum_range, mixture_spectrum, 3)

    # Wrap spectrum into a ramanspy container
    wrapped_spectrum = ramanspy.Spectrum(transformed_spectrum, spectrum_range)
    ramanspy.plot.spectra(wrapped_spectrum, color=colors[1], plot_type='stacked')
    plt.savefig('plot_after_adding_transforms.pdf', bbox_inches='tight', dpi=300)
    plt.close()

    # Preprocess the spectra with the assembled pipeline
    preprocessed_spectrum = pipe.apply(wrapped_spectrum)
    ramanspy.plot.spectra(preprocessed_spectrum, color=colors[3], plot_type='stacked')
    plt.savefig('plot_after_preprocessing.pdf', bbox_inches='tight', dpi=300)
    plt.close()


# Manual input
SPECTRUM_RANGE=  np.linspace(350, 1800, 1450) # spectrum range to be created in wavenumbers
MIXING_CONC = [5, 3, 2] # concentration of valine : histidine : tryptophan

if __name__ == "__main__":
    main(spectrum_range=SPECTRUM_RANGE, mixing_conc=MIXING_CONC)