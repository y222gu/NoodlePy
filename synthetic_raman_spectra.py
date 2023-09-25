# One function for generating spectra with different concentrations out of a list of molecular spectra
# One function augment the spectra
# add a proper main when run the script





# Loading the required packages:
import numpy as np
import ramanspy
import matplotlib.pyplot as plt
from scipy import stats

colors = plt.cm.get_cmap()(np.linspace(0, 1, 4))
# Defined a Gaussian function

def create_spectrum(spectrum_range:np.array, 
                    peak_location:np.array, 
                    peak_shape:np.array, 
                    peak_intensity:np.array)->np.array:
    """
    add two sentences to describe the fuction

    parameters:
    spectrum_range:np.array
        describe what this parameter is
    peak_location:np.array
        describe what this parameter is
    peak_shape:np.array
        describe what this parameter is
    peak_intensity:np.array
        describe what this parameter is

    returns:
    spectrum:np.array
         describe what this parameter is

    """
    spectrum =np.zeros(len(spectrum_range))

    for peak_i in np.arange(len(peak_location)):
        #spectrum += Gauss(spectrum_range,peak_location[peak_i],peak_shape[peak_i],peak_intensity[peak_i])
        spectrum += stats.norm.pdf(spectrum_range,peak_location[peak_i],peak_shape[peak_i]) * peak_intensity[peak_i]
    spectrum = spectrum/np.max(spectrum)

    return spectrum

# Create fingerprint sprectra for significant components in plasma samples

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

# valine
spectrum_range =  np.linspace(350, 1800, 1450)

peak_location_valine = np.array([374, 396, 429, 472, 497, 542, 664, 715, 753, 776, 824, 849, 891, 902, 923, 948, 964, 1029, 1035, 1066, 1106, 1125, 1146, 1179, 1191, 1272, 1321, 1330, 1343, 1351, 1398,1427, 1452, 1508, 1567,1587, 1619, 1633, 1660])
peak_intensity_valine = np.array([mw, mw, mw, w, w, vs, mw, m, m, s, m, s, w, m, mw, s, m, mw, mw, mw, w, m, mw, mw, m, m, m, m, s, s, m, m, s,mw, w, w, mw, mw, w])
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

# TODO:CHANGE VARIBLES TO GLOBAL VARIBALES TO ALL CAPITALS
SPECTRUM_VALINE = create_spectrum(spectrum_range, peak_location_valine, peak_shape_valine, peak_intensity_valine)
spectrum_histidine= create_spectrum(spectrum_range, peak_location_histidine, peak_shape_histidine, peak_intensity_histidine)
spectrum_tryptophan= create_spectrum(spectrum_range, peak_location_tryptophan, peak_shape_tryptophan, peak_intensity_tryptophan)

SPECTRA = [SPECTRUM_VALINE, spectrum_histidine, spectrum_tryptophan]

ramanspy_spectra = [ramanspy.Spectrum(SPECTRUM_VALINE, spectrum_range), ramanspy.Spectrum(spectrum_histidine, spectrum_range),ramanspy.Spectrum(spectrum_tryptophan,spectrum_range)]

plot_before_mixture = ramanspy.plot.spectra(ramanspy_spectra, color=colors[1], plot_type='stacked')
plt.savefig('plot_before_mixing_molecules.pdf', bbox_inches='tight', dpi=300)
plt.close()


# mix molecules at random concentrations
conc_valine = 5
conc_histidine = 3
conc_tryptophan = 2

mixture_spectrum = conc_valine * SPECTRUM_VALINE + conc_histidine * spectrum_histidine + conc_tryptophan * spectrum_tryptophan

#mixture_spectra = mixture_spectrum
ramanspy_spectra = ramanspy.Spectrum(mixture_spectrum,spectrum_range)
plot_before_mixture = ramanspy.plot.spectra(ramanspy_spectra, color=colors[1], plot_type='stacked')
plt.savefig('plot_after_mixing_molecules.pdf', bbox_inches='tight', dpi=300)
plt.close()




# adding noise
#for mixture_spectrum in mixture_spectra:
mixture_spectrum = mixture_spectrum + np.random.normal(0,0.4,len(mixture_spectrum))

#spectrum_histidine_noisy = spectrum_histidine + np.random.normal(0,0.04,len(spectrum_histidine))

# Spikes: 
number_spikes = 2

#for mixture_spectrum in mixture_spectra:
spikes = np.random.randint(0, len(spectrum_range),number_spikes)
for spike in spikes:
    mixture_spectrum[spike] = mixture_spectrum[spike] + 10*np.random.random()


# Baseline as a polynomial background:
#for mixture_spectrum in mixture_spectra:
poly = 0.05 * np.ones(len(spectrum_range)) + 0.00005 * spectrum_range + 0.000005 * (spectrum_range - 680)**2 
mixture_spectrum = mixture_spectrum + poly
    #spectrum_histidine_noisy = spectrum_histidine_noisy + poly

ramanspy_mixture_spectra = ramanspy.Spectrum(mixture_spectrum, spectrum_range)

# set up a preprocess pipeline
pipe = ramanspy.preprocessing.protocols.Pipeline([
    ramanspy.preprocessing.despike.WhitakerHayes(),
    ramanspy.preprocessing.denoise.SavGol(window_length=12, polyorder=3),
    ramanspy.preprocessing.baseline.ASPLS(),
    ramanspy.preprocessing.normalise.MinMax(pixelwise=True),
])

# preprocess the spectra
preprocessed_mixture_spectra = pipe.apply(ramanspy_mixture_spectra)

# plot the results
plot_before_preprocessing = ramanspy.plot.spectra(mixture_spectrum, color=colors[1], plot_type='stacked')
plt.savefig('plot_after_adding_noise.pdf', bbox_inches='tight', dpi=300)
plt.close()

plot_after_preprocessing = ramanspy.plot.spectra(preprocessed_mixture_spectra, color=colors[3], plot_type='stacked')
plt.savefig('plot_after_preprocessing.pdf', bbox_inches='tight', dpi=300)
plt.close()


'''
nfindr = ramanspy.analysis.unmix.NFINDR(n_endmembers=3, abundance_method='fcls')
abundance_maps, endmembers = nfindr.apply(preprocessed_mixture_spectra)

ramanspy.plot.spectra(endmembers, preprocessed_mixture_spectra.spectral_axis, plot_type="single stacked", label=[f"Endmember {i + 1}" for i in range(len(endmembers))])
plt.savefig('plot_after_unmixing.pdf', bbox_inches='tight', dpi=300)
plt.close()
'''

if __name__ == "__main__":