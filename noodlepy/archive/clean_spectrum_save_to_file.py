from noodlepy.utils.spectrum import Spectrum
from noodlepy.utils.spectrumpreprocessor import SpectrumPreprocessor
import os
import pandas as pd
import numpy as np
import copy


def load_files(data_folder = None,
                 annotation_file_path = None):
        """
        Load the database of spectra from the txt file 

        Args:
        data_folder (str): The folder where the spectra are stored

        Returns:
        list[Spectrum]: A list of Spectrum objects
        """
        print("Loading the Raman dataset")

        list_of_spectrum_objects = []
        list_of_file_names = sorted([f for f in os.listdir(data_folder) if f.endswith(('.txt'))])

        annotation_all = pd.read_excel(annotation_file_path)

        for filename in list_of_file_names:
            patient_annotations = extract_patient_labels(filename, annotation_all)
            spectrum_objects = load_files_to_spectrum_objects(data_folder, filename, patient_annotations)
            list_of_spectrum_objects+=spectrum_objects

        print(f"Loaded {len(list_of_spectrum_objects)} spectra")
        return list_of_spectrum_objects

def extract_patient_labels(spectrum_file_name:str, 
                        all_patient_labels:pd.DataFrame):

        # Patient metatdata extraction
        patient_labels = {}
        f_split = spectrum_file_name.split('_')
        patient_id = int(f_split[0])
        sample_type = f_split[1]
        patient_labels['patient_id'] = patient_id
        patient_labels['sample_type'] = sample_type
        # Extract the metadata for the given patient_id
        if patient_id in all_patient_labels['OD Number'].values:
            patient_metadata_row = all_patient_labels[all_patient_labels['OD Number'] == patient_id]

            if len(patient_metadata_row) > 1:
                patient_metadata_row = patient_metadata_row.iloc[[0]]
                print(f"Patient ID {patient_id} has multiple entries in the metadata file")
                print("Only the first entry will be used")
                
            patient_labels['staging'] = patient_metadata_row['Staging'].values[0]

        else:
            # If the patient_id is not found in the metadata file, set the every metadata to empty string and number
            patient_labels['staging'] = np.nan

            print(f"Patient ID {patient_id} not found in the metadata file")
            print("Metadata set to empty strings and numbers")
        return patient_labels
    
def load_files_to_spectrum_objects(data_folder:str,
                               filename:str, 
                               patient_annotations:dict):
        with open(os.path.join(data_folder, filename)) as f:
            data = pd.read_csv(f, sep=",", header=None)

            repeated_wavelengths = data.iloc[:,0].value_counts()
            first_repeated_wavelength = repeated_wavelengths.idxmax()
            start_indexes = data[data.iloc[:,0] == first_repeated_wavelength].index.tolist()

            spectrum_objects = []
            # split the repeated measurements into individual spectra
            for i in range(len(start_indexes)):
                spectrum_id = i + 1
                if i == len(start_indexes) - 1:
                    wavelength_nm = data.iloc[start_indexes[i]:, 0].values.round(3)
                    intensity = data.iloc[start_indexes[i]:, 1].values.round(3)
                else:
                    wavelength_nm = data.iloc[start_indexes[i]:start_indexes[i + 1], 0].values.round(3)
                    intensity = data.iloc[start_indexes[i]:start_indexes[i + 1], 1].values.round(3)
                
                metadata = copy.deepcopy(patient_annotations)
                metadata['spectrum_id'] = spectrum_id
                spectrum = Spectrum(wavelength_nm=wavelength_nm,
                                    intensity=intensity,
                                    metadata=metadata,)
                spectrum_objects.append(spectrum)
        return spectrum_objects

def preprocess_and_save_to_file(db, save_file_to_path:str):
        preprocessor = SpectrumPreprocessor(cropping=True, baseline_correction=True, remove_cosmic_rays=True, normalization=True, smoothing=True)
        with open(save_file_to_path, 'w') as f:
            for spectrum in db:
                spectrum = preprocessor.pre_process(spectrum)
                # save wavelength and intensity to file with name as patient_id_sample_type_spectrum_id_staging.txt
                file_name = f"{spectrum.metadata['patient_id']}_{spectrum.metadata['sample_type']}_{spectrum.metadata['spectrum_id']}_{spectrum.metadata['staging']}.txt"
                f.write(f"{file_name}\n")
                f.write(f"{spectrum.wavelength_nm}\n")
                f.write(f"{spectrum.intensity}\n")
                f.write("\n")

def main():
    current_directory = os.getcwd()
    data_file_path = os.path.join(current_directory, "NoodlePy","noodlepy","data","Raman_DB","plasma_saliva_mixed","all")
    annotation_file_path = os.path.join(current_directory, "NoodlePy","noodlepy","data","Biofluid_list_annotated_v4.xlsx")
    file_path_to_save = os.path.join(current_directory,"data","raman_data","cleaned_spectra")
    db = load_files(data_file_path, annotation_file_path)
    preprocess_and_save_to_file(db, file_path_to_save)

if __name__ == "__main__":
    main()