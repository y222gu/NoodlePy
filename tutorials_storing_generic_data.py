import ramanspy

import ramanspy
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import random

# an evenly spaced Raman wavenumber axis between 100 and 3000 cm^-1, consisting of 1500 elements.
spectral_axis = np.linspace(100, 3600, 1500)

# randomly generating intensity data array of shape (20, 1500)
spectral_data = np.random.rand(20, 1500)

# wrapping the data into a SpectralContainer instance
raman_object = ramanspy.SpectralContainer(spectral_data, spectral_axis)

spectral_data = np.random.rand(1500)
raman_spectrum = ramanspy.SpectralContainer(spectral_data, spectral_axis)

spectral_data = np.random.rand(20, 20, 1500)
raman_image = ramanspy.SpectralContainer(spectral_data, spectral_axis)

spectral_data = np.random.rand(20, 20, 20, 1500)
raman_volume = ramanspy.SpectralContainer(spectral_data, spectral_axis)

spectral_data = np.random.rand(20, 20, 20, 20, 1500)
raman_hypervolume = ramanspy.SpectralContainer(spectral_data, spectral_axis)

raman_spectra = [ramanspy.Spectrum(np.random.rand(1500), spectral_axis) for _ in range(5)]
raman_spectra_list = ramanspy.SpectralContainer.from_stack(raman_spectra)

print(raman_spectra_list.shape)