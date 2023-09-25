import pandas as pd
import ramanspy as rp
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import random


file_1 ='275_plasma_dry_60x_1mm_65mW_cw-853_grating3_quartz_combined_cleaned_avg.txt'
file_2 = '277_plasma_dry_60x_1mm_65mW_cw-853_grating3_quartz_combined_cleaned_avg.txt'

raw_data_1 = pd.read_csv(file_1)
raw_data_2 = pd.read_csv(file_2)

print(raw_data.shape)
spectral_axis = raw_data.iloc[:,0].values
spectral_data_1 = raw_data_1.iloc[:,1].values
spectral_data_2 = raw_data_2.iloc[:,1].values


print(spectral_data)
print(type(spectral_data))

# Open the file for reading

raman_object = rp.SpectralContainer(spectral_data, spectral_axis)

plt.plot(spectral_axis,spectral_data)
#rp.plot.spectra(raman_object)

raman_spectra = [rp.Spectrum(np.random.rand(1500), spectral_axis) for _ in range(5)]
raman_spectra_list = rp.SpectralContainer.from_stack(raman_spectra)

raman_spectra_list.shape

raman_object_ = rp.SpectralContainer()
