import os
import pandas as pd
import json
import shutil

def clean_files(data_folder = None,
                 annotation_for_low_quality_spectra = None, output_folder = None):
        """
        Load the database of spectra from the txt file 

        Args:
        data_folder (str): The folder where the spectra are stored

        Returns:
        list[Spectrum]: A list of Spectrum objects
        """
        print("Loading the Raman dataset")

        list_of_file_names = []
        for root, dirs, files in os.walk(data_folder):
            for file in files:
                if file.endswith('.txt'):
                    list_of_file_names.append(os.path.join(root, file))
        list_of_file_names = sorted(list_of_file_names)

        # Load json file in format of patient_id: [spectra_id]
        with open(annotation_for_low_quality_spectra) as f:
            low_quality_spectra = json.load(f)

            combined_low_quality_spectra = {}
            for low_quality_spectrum in low_quality_spectra:
                key = (low_quality_spectrum['patient_id'], low_quality_spectrum['date'], low_quality_spectrum['position'])
                if key not in combined_low_quality_spectra:
                    combined_low_quality_spectra[key] = []
                combined_low_quality_spectra[key].append(low_quality_spectrum['spectrum_id'])
            
            low_quality_spectra = combined_low_quality_spectra

            for key, low_quality_spectra in low_quality_spectra.items():
                patient_id, date, position = key
                spectra_to_remove = low_quality_spectra

                for file_name in list_of_file_names:
                    file_name_base = os.path.basename(file_name)
                    file_date, file_patient_id, file_position = extract_patient_labels(file_name_base)
                    if file_patient_id == int(patient_id) and file_date == date and file_position == position:
                        cleaned_spectra = remove_spectrum_id(file_name, spectra_to_remove)
                        
                        save_file_to_path = os.path.join(data_folder, file_name)
                        save_to_file(cleaned_spectra, save_file_to_path)
                        break

def extract_patient_labels(spectrum_file_name):
        f_split = spectrum_file_name.split('_')
        date = f_split[0]
        patient_id = int(f_split[1])
        position = f_split[-1].split('.')[0]
        return date, patient_id, position
    
def remove_spectrum_id(file_path, 
                               spectra_id_to_remove:list):
        with open(file_path) as f:
            data = pd.read_csv(f, sep=",", header=None)

            repeated_wavelengths = data.iloc[:,0].value_counts()
            first_repeated_wavelength = repeated_wavelengths.idxmax()
            start_indexes = data[data.iloc[:,0] == first_repeated_wavelength].index.tolist()

            spectrum_list = []
            # split the repeated measurements into individual spectra
            for i in range(len(start_indexes)):
                spectrum_id = i + 1
                # check if the spectrum_id is in the list of spectra to remove
                if spectrum_id in spectra_id_to_remove:
                    continue
                else:
                    if i == len(start_indexes) - 1:
                        wavelength_nm = data.iloc[start_indexes[i]:, 0].values.round(3)
                        intensity = data.iloc[start_indexes[i]:, 1].values.round(3)
                    else:
                        wavelength_nm = data.iloc[start_indexes[i]:start_indexes[i + 1], 0].values.round(3)
                        intensity = data.iloc[start_indexes[i]:start_indexes[i + 1], 1].values.round(3)
                    
                    # append wavelength and intensity to the spectrum_list and separated by a comma
                    spectrum_list.append([wavelength_nm, intensity])
            print(f'{spectra_id_to_remove} removed from {os.path.basename(file_path)}')
            print("Number of spectra after removing low quality spectra: ", len(spectrum_list))
            return spectrum_list

def save_to_file(cleaned_spectra, save_file_to_path:str):
    with open(save_file_to_path, 'w') as f:
        for i in range(len(cleaned_spectra)):
            for j in range(len(cleaned_spectra[i][0])):
                f.write(str(cleaned_spectra[i][0][j]) + "," + str(cleaned_spectra[i][1][j]) + "\n")
            if i != len(cleaned_spectra) - 1:
                f.write("\n")

if __name__ == "__main__":
    data_folder = r"C:\Users\Yifei\Documents\NoodlePy\noodlepy\data\hnc_raw_data_copy"
    output_folder = r"C:\Users\Yifei\Documents\NoodlePy\noodlepy\data\hnc_raw_data_cleaned"
    annotation_for_low_quality_spectra = r"C:\Users\Yifei\Documents\NoodlePy\quality_control\weird_spectra.json"
    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder)
    clean_files(data_folder, annotation_for_low_quality_spectra, output_folder)
