import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
from noodlepy.utils.spectrum import Spectrum
from noodlepy.utils.spectrumpreprocessor import SpectrumPreprocessor
# Path to the folder containing images
data_folder = os.path.join(os.getcwd(),'noodlepy', 'data', 'bec_hnc', 'all')

# list of files in the folder
files = sorted([f for f in os.listdir(data_folder) if f.endswith(('.txt'))])

# open each text file and find the repeated measurements
data = []
preprocessor = SpectrumPreprocessor(cropping=True,
                                        baseline_correction=True,
                                        remove_cosmic_rays= True,
                                        normalization=True,
                                        smoothing=True)

for file in files:
    with open(os.path.join(data_folder, file)) as f:
        data = pd.read_csv(f, sep=",", header=None)

        repeated_wavelengths = data.iloc[:,0].value_counts()
        first_repeated_wavelength = repeated_wavelengths.idxmax()
        start_indexes = data[data.iloc[:,0] == first_repeated_wavelength].index.tolist()

        spectra = []
        # split the repeated measurements into individual spectra
        for i in range(len(start_indexes)):
            spectrum_id = i + 1
            if i == len(start_indexes) - 1:
                wavelength_nm = data.iloc[start_indexes[i]:, 0].values.round(3)
                intensity = data.iloc[start_indexes[i]:, 1].values.round(3)
            else:
                wavelength_nm = data.iloc[start_indexes[i]:start_indexes[i + 1], 0].values.round(3)
                intensity = data.iloc[start_indexes[i]:start_indexes[i + 1], 1].values.round(3)
            spectra.append((spectrum_id, wavelength_nm, intensity))


        # take the median of the repeated measurements
        median_spectrum = []
        for i in range(len(wavelength_nm)):
            intensity_at_wavelength = [spectrum[2][i] for spectrum in spectra]
            median_intensity = sum(intensity_at_wavelength) / len(intensity_at_wavelength)
            median_spectrum.append((wavelength_nm[i], median_intensity))

        intensity = [row[1] for row in median_spectrum]
        
        # load them as a Spectrum object
        spectrum_object = Spectrum(wavelength_nm=wavelength_nm, intensity=intensity)
        spectrum_object = preprocessor.preprocess(spectrum_object)
        # spectrum_object.display()
        preprocessed_wavelength_nm = spectrum_object.wavelength_nm
        preprocessed_intensity = spectrum_object.intensity


        preprocessed_median_spectrum = list(zip(preprocessed_wavelength_nm, preprocessed_intensity))

        new_file_name = "_".join([file.split('_')[0], file.split('_')[-2], file.split('_')[-1]])

        # save the median spectrum to a new file
        with open(os.path.join(data_folder, new_file_name), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['wavelength_nm', 'intensity'])
            for row in preprocessed_median_spectrum:
                writer.writerow(row)