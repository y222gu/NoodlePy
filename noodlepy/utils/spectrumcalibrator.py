import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os
import copy

class SpectrumCalibrator:
    def __init__(self, measured_neon_file=None, measured_white_lamp=None, standard_neon_file=None, calibrate_x = True, calibrate_y = True):
        self.calibrate_x = calibrate_x
        self.calibrate_y = calibrate_y
        self.measured_neon_file = measured_neon_file
        self.measured_white_lamp = measured_white_lamp
        if standard_neon_file is None:
            current_directory = os.getcwd()
            self.standard_neon_file = os.path.join(current_directory, "noodlepy", "config", "neon_NIST_peak_locations.csv")
        else:
            self.standard_neon_file = standard_neon_file
        self.ccd_efficiency = SpectrumCalibrator._get_ccd_efficiency(self)
    
    def _get_ccd_efficiency(self):
        lamp = SpectrumCalibrator.load_mean_of_repeated_measurements(self.measured_white_lamp)
        avg_lamp = np.mean(lamp["intensity"])
        efficiency = lamp["intensity"]/avg_lamp
        efficiency = efficiency.values

        # plt.plot(lamp["wavelength"], efficiency)
        # plt.xlabel("Wavelength (nm)")
        # plt.ylabel("Efficiency")
        # plt.title("Efficiency of CCD sensor at different wavelengths")
        # plt.text(800, 1.14, "Efficiency = spectrum of white lamp / mean(spectrum of white lamp)", color='red')
        # plt.show()

        # plt.plot(lamp["wavelength"], lamp["intensity"])
        # plt.xlabel("Wavelength (nm)")
        # plt.ylabel("Intensity")
        # plt.title("Spectrum of White Lamp")
        # plt.show()
        return efficiency
      
    def load_mean_of_repeated_measurements(file):
        with open(file) as f:
            data = pd.read_csv(f, sep=",", header=None)
            data.columns = ["wavelength", "intensity"]
        mean_data = data.groupby("wavelength").mean().reset_index()
        return mean_data
    
    # split the repeated measurements into individual spectra and return a list of lists
    def split_repeated_measurements(file):
        with open(file) as f:
            data = pd.read_csv(f, sep=",", header=None)
            data.columns = ["wavelength", "intensity"]

        repeated_wavelengths = data["wavelength"].value_counts()
        first_repeated_wavelength = repeated_wavelengths.idxmax()
        start_indexes = data[data["wavelength"] == first_repeated_wavelength].index.tolist()

        spectra = []
        # split the repeated measurements into individual spectra
        for i in range(len(start_indexes)):
            if i == len(start_indexes) - 1:
                wavelength_nm = data.iloc[start_indexes[i]:, 0].values.round(3)
                intensity = data.iloc[start_indexes[i]:, 1].values.round(3)
            else:
                wavelength_nm = data.iloc[start_indexes[i]:start_indexes[i + 1], 0].values.round(3)
                intensity = data.iloc[start_indexes[i]:start_indexes[i + 1], 1].values.round(3)
            spectrum = {"wavelength": wavelength_nm, "intensity": intensity}
            spectra.append(spectrum)
        return spectra
    
    def find_neon_peaks(measured_neon_file):
        # split the repeated measurements into individual spectra
        mean_measured_neon_spectrum = SpectrumCalibrator.load_mean_of_repeated_measurements(measured_neon_file)
        peak_list = pd.DataFrame(columns=["wavelength", "intensity"])
        # Crop the data
        mean_measured_neon_spectrum_cropped = mean_measured_neon_spectrum[250:700]
        all_peaks, properties = find_peaks(mean_measured_neon_spectrum_cropped["intensity"], height=1)

        for i in range(8):
            peak_index = properties["peak_heights"].argmax()
            peak_wavelength = mean_measured_neon_spectrum_cropped["wavelength"][all_peaks[peak_index]+250]
            peak_intensity = properties["peak_heights"][peak_index]

            new_row = pd.DataFrame({"wavelength": [peak_wavelength], "intensity": [peak_intensity]})
            peak_list = pd.concat([peak_list,new_row], ignore_index=True)

            properties["peak_heights"][peak_index] = 0

        peak_list = peak_list.sort_values(by="wavelength")

        # plt.scatter(peak_list["wavelength"], peak_list["intensity"], c='r')
        # plt.plot(mean_measured_neon_spectrum_cropped["wavelength"], mean_measured_neon_spectrum_cropped["intensity"])
        # plt.xlabel("Wavelength (nm)")
        # plt.ylabel("Intensity")
        # plt.title("Detect peaks in the Measured Neon Spectrum")
        # plt.show()
        return peak_list['wavelength'].values

    def calibrate_x_axis(self, spectrum:dict):
        with open(self.standard_neon_file) as f:
            standard_neon_peaks = pd.read_csv(f)

        standard_neon_peaks = standard_neon_peaks["wavelength"].values
        measured_neon_peaks = SpectrumCalibrator.find_neon_peaks(self.measured_neon_file)

        def linear(x, m, b):
            return m*x + b
        
        popt, _ = curve_fit(linear, measured_neon_peaks, standard_neon_peaks)
        m, b = popt
        linear_fit = linear(measured_neon_peaks, m, b)
        calibrated_wavelength = linear(spectrum["wavelength"], m, b)
        calibrated_wavelength = np.round(calibrated_wavelength, 6)

        # plt.figure()
        # plt.scatter(measured_neon_peaks, standard_neon_peaks, c = 'r')
        # plt.plot(measured_neon_peaks, linear_fit, 'b')
        # plt.text(835, 865, f"y = {round(m,3)}x + {round(b,3)}", color='red')
        # plt.xlabel("Measured Wavelength (nm)")
        # plt.ylabel("Standard Wavelength (nm)")
        # plt.title("Calibration of Measured Wavelengths to Standard Wavelengths")
        # plt.show()

        # plt.figure()
        # plt.plot(spectrum["wavelength"], spectrum["intensity"])
        # plt.plot(calibrated_wavelength, spectrum["intensity"])
        # plt.xlabel("Wavelength (nm)")
        # plt.ylabel("Intensity")
        # plt.title("Spectrum calibriated along x-axis using Neon lamp")
        # plt.legend(["Raw", "Calibrated"])
        # plt.show()
        return calibrated_wavelength
    
    def calibrate(self, file):
        spectra = SpectrumCalibrator.split_repeated_measurements(file)
        calibrated_spectra = copy.deepcopy(spectra)

        for i, calibrated_spectrum in enumerate(calibrated_spectra):
        # calibrate x axis data
            if self.calibrate_x:
                calibrated_spectrum["wavelength"]= SpectrumCalibrator.calibrate_x_axis(self, calibrated_spectrum)

        # calibrate y axis data
            if self.calibrate_y:
                intensity_calibrated = calibrated_spectrum["intensity"]/self.ccd_efficiency
                calibrated_spectrum["intensity"] = np.round(intensity_calibrated,0)

        # # get the path of the output plots folder
        # current_directory = os.getcwd()
        # output_folder = os.path.join(current_directory, "output_plots")

        # # # plot the raw data
        # for spectrum, calibrated_spectrum in zip(spectra, calibrated_spectra):
        #     plt.figure()
        #     plt.plot(spectrum["wavelength"], spectrum["intensity"])
        #     plt.plot(calibrated_spectrum["wavelength"], calibrated_spectrum["intensity"])
        #     plt.legend(["Raw", "Calibrated"])
        #     plt.xlabel("Wavelength (nm)")
        #     plt.ylabel("Intensity")
        #     plt.title(f"Calibrated Spectrum {i}")
        #     plt.savefig(os.path.join(output_folder, f"calibrated_spectrum_{i}.png"))
        return spectra
    
    @staticmethod
    def save_calibrated_spectra(spectra, save_file_to_path):
        with open(save_file_to_path, 'w') as f:
            for spectrum in spectra:
                # write the columns of the spectrum, and separate the columns with commas
                for i in range(len(spectrum["wavelength"])): 
                    f.write(f"{spectrum['wavelength'][i]},{spectrum['intensity'][i]}\n")

if __name__ == "__main__":

    folder = "/Users/yifeigu/Documents/Carney_Lab/NoodlePy/noodlepy/data/202404_OvCa-project"

    calibrated_folder = folder + "_calibrated"
    if not os.path.exists(calibrated_folder):
        os.mkdir(calibrated_folder)

    for root_layer_1, dirs_layer_1, files_layer_ in os.walk(folder):

        for dir_layer_1 in dirs_layer_1:

            calibrated_folder_layer_1 = os.path.join(calibrated_folder, dir_layer_1)
            if not os.path.exists(calibrated_folder_layer_1):
                os.mkdir(calibrated_folder_layer_1)

            # fine the neon and lamp files
            for root_layer_2, dirs_layer_2, files_layer_2 in os.walk(os.path.join(root_layer_1, dir_layer_1)):
                search_term_for_neon = "neon"
                neon_file = [file for file in files_layer_2 if search_term_for_neon in file]
                neon_file_path = os.path.join(root_layer_2, neon_file[0])
                search_term_for_lamp = "spectralcal"
                lamp_file = [file for file in files_layer_2 if search_term_for_lamp in file]
                lamp_file_path = os.path.join(root_layer_2, lamp_file[0])
                calibrator = SpectrumCalibrator(neon_file_path, lamp_file_path, calibrate_x = True, calibrate_y = False)

                # list all the spectra and calibrate them
                for dir_layer_2 in dirs_layer_2:

                    calibrated_folder_layer_2 = os.path.join(calibrated_folder_layer_1, dir_layer_2)
                    if not os.path.exists(calibrated_folder_layer_2):
                        os.mkdir(calibrated_folder_layer_2)

                    for root_layer_3, dirs_layer_3, files_layer_3 in os.walk(os.path.join(root_layer_2, dir_layer_2)):
                        for dir_layer_3 in dirs_layer_3:

                            calibrated_folder_layer_3 = os.path.join(calibrated_folder_layer_2, dir_layer_3)
                            if not os.path.exists(calibrated_folder_layer_3):
                                os.mkdir(calibrated_folder_layer_3)

                            for root_layer_4, dirs_layer_4, files_layer_4 in os.walk(os.path.join(root_layer_3, dir_layer_3)):
                                for file_layer_4 in files_layer_4:
                                    
                                    if file_layer_4.endswith(".txt"):
                                        file_path = os.path.join(root_layer_4, file_layer_4)
                                        spectra = calibrator.calibrate(file_path)
                                        save_file_to_path = os.path.join(calibrated_folder_layer_3, f"calibrated_{file_layer_4}")
                                        calibrator.save_calibrated_spectra(spectra, save_file_to_path)
                                break
                        break
                break
        break



