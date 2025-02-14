import os
import pandas as pd
import json
import shutil

def clean_files(data_folder = None,
                 annotation_for_low_quality_spectra = None):
        """
        Load the database of spectra from the txt file 

        Args:
        data_folder (str): The folder where the spectra are stored

        Returns:
        list[Spectrum]: A list of Spectrum objects
        """
        print("Loading the Raman dataset")

        list_of_file_names = sorted([f for f in os.listdir(data_folder) if f.endswith(('.txt'))])

        # Load json file in format of patient_id: [spectra_id]
        with open(annotation_for_low_quality_spectra) as f:
            low_quality_spectra = json.load(f)

        for filename in list_of_file_names:
            spectra_id_to_remove = []
            cleaned_spectra = []
            patient_id, sample_type = extract_patient_labels(filename)
            data_folder_updated = os.path.join(data_folder, sample_type)
            if not os.path.exists(data_folder_updated):
                os.makedirs(data_folder_updated)
            save_file_to_path = os.path.join(data_folder_updated, filename)

            for entry in low_quality_spectra:
                if entry[0] == patient_id and entry[1] == sample_type:
                    spectra_id_to_remove = entry[2]
                    print(spectra_id_to_remove)
                    cleaned_spectra = remove_spectrum_id(data_folder, filename, spectra_id_to_remove)
                    break
            
            if len(cleaned_spectra) > 0:
                save_to_file(cleaned_spectra, save_file_to_path)
            
            if len(cleaned_spectra) == 0 and spectra_id_to_remove != []:
                print(f"File {filename} is empty after removing low quality spectra. File will not be saved.")

            if spectra_id_to_remove == []:
                original_file_path = os.path.join(data_folder, filename)
                shutil.copy(original_file_path, save_file_to_path)

def extract_patient_labels(spectrum_file_name):
        f_split = spectrum_file_name.split('_')
        patient_id = int(f_split[0])
        sample_type = f_split[1]
        return patient_id, sample_type
    
def remove_spectrum_id(data_folder:str,
                               filename:str, 
                               spectra_id_to_remove:list):
        with open(os.path.join(data_folder, filename)) as f:
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
            print(f'{spectra_id_to_remove} removed from {filename}')
            print("Number of spectra after removing low quality spectra: ", len(spectrum_list))
            print(f"{len(spectrum_list)/1024} spectra left")
            return spectrum_list

def save_to_file(cleaned_spectra, save_file_to_path:str):
        with open(save_file_to_path, 'w') as f:
            for i in range(len(cleaned_spectra)):
                for j in range(len(cleaned_spectra[i][0])):
                    f.write(str(cleaned_spectra[i][0][j]) + "," + str(cleaned_spectra[i][1][j]) + "\n")
                if i != len(cleaned_spectra) - 1:
                    f.write("\n")

def main():
    current_directory = os.getcwd()
    data_folder = os.path.join(current_directory, "noodlepy","data", "head_and_neck_cancer", "plasma_cleaned")
    annotation_for_low_quality_spectra = os.path.join(current_directory, "low_quality_spectra.json")
    clean_files(data_folder, annotation_for_low_quality_spectra)

if __name__ == "__main__":
    main()