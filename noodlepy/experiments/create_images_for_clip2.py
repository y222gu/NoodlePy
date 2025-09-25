import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch.utils.data import Dataset
import torch
from transformers import CLIPProcessor
from noodlepy.utils.spectrum import Spectrum



class HyperspectralDataset(Dataset):
    def __init__(self, data_folder = None,
                 annotation_file_path = None,
                 r_filter = None, 
                 clip_model_name="openai/clip-vit-base-patch32"):
        """
        images: numpy array of shape (N, 50, 8, 745)
        labels: numpy array of shape (N,) with 0 or 1
        """
        print("Loading the Raman dataset")
        if r_filter is None:
            r_filter = np.arange(1, 51)
        self.r_filter = r_filter


        list_of_file_paths = []
        for root, dirs, files in os.walk(data_folder):
            if root == data_folder or root.count(os.sep) == data_folder.count(os.sep) + 1:
                list_of_file_paths += [os.path.relpath(os.path.join(root, f), data_folder) for f in files if f.endswith('.txt')]
        list_of_file_paths = sorted(list_of_file_paths)

        annotation_all = pd.read_excel(annotation_file_path)

        spectrum_objects_list = []

        for file_path in list_of_file_paths:
            patient_annotations = HyperspectralDataset._extract_patient_labels(file_path, self.r_filter, annotation_all)
            spectrum_objects = HyperspectralDataset._load_files_to_spectrum_objects(data_folder, file_path, patient_annotations)
            spectrum_objects_list += spectrum_objects

        # get the list of unique (patient_id, date) pairs
        unique_patient_dates = set()
        for spectrum in spectrum_objects_list:
            patient_id = spectrum.metadata['patient_id']
            date = spectrum.metadata['date']
            unique_patient_dates.add((patient_id, date))

        self.images = []
        self.labels = []
        self.metadata = []

        # Form each unique (patient_id, date), stack the spectrum intensity with the same line into an image
        # and assign the label to the first spectrum in the list
        for i, (patient_id, date) in enumerate(unique_patient_dates):
            patient_spectra = [spectrum for spectrum in spectrum_objects_list if spectrum.metadata['patient_id'] == patient_id and spectrum.metadata['date'] == date]
            # find the spectrum with the same line and stack them into an image from ring 1 to 50
            for line in range(1, 5):
                line_spectra = [spectrum for spectrum in patient_spectra if spectrum.metadata['line'] == line]
                if len(line_spectra) > len(r_filter)-1:
                    # stack the spectra into an image
                    image = np.zeros((len(self.r_filter), 750), dtype=np.float32)
                    min_ring = min(r_filter)
                    for spectrum in line_spectra:
                        ring = spectrum.metadata['ring']
                        # crop the spectrum to 745 wavelengths
                        cropped_spectrum = copy.deepcopy(spectrum).crop_spectrum(615.879, 1784.104) #624.573
                        # nomalize the spectrum
                        # cropped_spectrum.normalize_spectrum()
                        raman_shift = cropped_spectrum.raman_shift_cm
                        image[ring - min_ring, :] = cropped_spectrum.intensity
                        
                    updated_metadata = copy.deepcopy(line_spectra[0].metadata)    
                    updated_metadata['raman_shift'] = raman_shift
                    self.images.append(image)
                    self.labels.append(updated_metadata['staging'])
                    self.metadata.append(updated_metadata)

                    print(f"Patient ID {patient_id}, line {line}:")
                    print('image shape:', image.shape)
                else:
                    print(f"Patient ID {patient_id} has {len(line_spectra)} spectra for date {date} and line {line}.")
                    print("Skipping this line.")

        print(f"Total number of images: {len(self.images)}")
        # check the number of unique patient_id, date, line pairs
        unique_patient_dates_lines = set()
        for metadata in self.metadata:
            patient_id = metadata['patient_id']
            date = metadata['date']
            line = metadata['line']
            unique_patient_dates_lines.add((patient_id, date, line))
        print(f"Total number of unique patient_id, date, line pairs: {len(unique_patient_dates_lines)}")



    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1) reduce to 3 channels
        image = self.images[idx]                        # (50, 745)
        label = self.labels[idx]                      # (1,)
        return image, label
    
    def _extract_patient_labels(spectrum_file_path:str, 
                                r_filter: np.array,
                          all_patient_labels:pd.DataFrame):

        # Patient metatdata extraction
        patient_labels = {}
        file_name = os.path.basename(spectrum_file_path)
        f_split = file_name.split('_')
        date = f_split[0]
        patient_id = int(f_split[1])
        sample_type = f_split[2]
        line = int(f_split[9])
        ring = int(f_split[10].split('.')[0])

        if patient_id in all_patient_labels['OD Number'].values:
            if r_filter is None or ring in r_filter:
                patient_labels['date'] = date
                patient_labels['patient_id'] = patient_id
                patient_labels['sample_type'] = sample_type
                patient_labels['ring'] = ring
                patient_labels['line'] = line
                patient_metadata_row = all_patient_labels[all_patient_labels['OD Number'] == patient_id]

                if len(patient_metadata_row) > 1:
                    patient_metadata_row = patient_metadata_row.iloc[[0]]
                    print(f"Patient ID {patient_id} has multiple entries in the metadata file")
                    print("Only the first entry will be used")
                    
                if patient_metadata_row['Staging'].values[0] == 0:
                    patient_labels['staging'] = int(0)
                elif patient_metadata_row['Staging'].values[0] == 1 or patient_metadata_row['Staging'].values[0] == 2:
                    patient_labels['staging'] = int(1)
                elif patient_metadata_row['Staging'].values[0] == 3 or patient_metadata_row['Staging'].values[0] == 4:
                    patient_labels['staging'] = int(1)

                patient_labels['gender'] = patient_metadata_row['Gender'].values[0]
                patient_labels['race'] = patient_metadata_row['Race'].values[0]
                return patient_labels
            else:
                return {}
        else:
            print(f"Patient ID {patient_id} not found in the metadata file")
            print("Metadata set to empty strings and numbers")
        return {}
    
    def _load_files_to_spectrum_objects(data_folder:str,
                               filename:str, 
                               patient_annotations:dict):
        if patient_annotations == {}: # skip the spectrum if the patient_id is not found in the metadata file
            return []

        with open(os.path.join(data_folder, filename)) as f:
            # if the file is empty, skip the file
            if os.stat(os.path.join(data_folder, filename)).st_size == 0:
                return []
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


    def plot_spectrum_as_heatmap(self, idx):
        """
        Plot the spectrum.intentsity as a heatmap for a given patient_id and date.
        The output should be 25x8x745 heatmap with the x-axis ring (50), y-axis line (4), and z-axis intensity (1024).
        """
        image = self.images[idx]                        # (50, 745)
        label = self.labels[idx]                      # (1,)
        metadata = self.metadata[idx]                  # (1,)
        raman_shift = metadata['raman_shift']          # (745,)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image, aspect='auto', cmap='hot', interpolation='nearest')
        # set x axis tick positions and labels based on raman shift
        num_ticks = 10
        tick_positions = np.linspace(0, len(raman_shift) - 1, num_ticks).astype(int)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(np.round(raman_shift[tick_positions], 3), rotation=45)
        ax.set_title(f"Patient: {metadata['patient_id']}, line: {metadata['line']}, Staging: {label}")
        ax.set_xlabel("wavenumber (cm-1)")
        ax.set_ylabel("Intensity")
        plt.colorbar(ax.imshow(image, aspect='auto', cmap='hot', interpolation='nearest'))
        # plt.show()
        # save to file
        if label == 0:
            # check if the directory exists, if not create it
            if not os.path.exists(os.path.join("noodlepy", "data", "images", 'control_without_normalization')):
                os.makedirs(os.path.join("noodlepy", "data", "images", 'control_without_normalization'))
            plt.savefig(os.path.join("noodlepy", "data", "images", 'control_without_normalization',f"patient_{metadata['patient_id']}_line_{metadata['line']}_staging_0.png"))
        elif label == 1:
            # check if the directory exists, if not create it
            if not os.path.exists(os.path.join("noodlepy", "data", "images", 'cancer_without_normalization')):
                os.makedirs(os.path.join("noodlepy", "data", "images", 'cancer_without_normalization'))
            plt.savefig(os.path.join("noodlepy", "data", "images", 'cancer_without_normalization', f"patient_{metadata['patient_id']}_line_{metadata['line']}_staging_1.png"))
        return
    

if __name__ == "__main__":
    # point this to your data folder
    data_dir = os.path.join(os.getcwd(), "noodlepy", "data", "cosmic_ray_removed", "train")
    annotation_file_path = os.path.join(os.getcwd(), "noodlepy", "data", "bec_hnc_patient_annotations.xlsx")
    dataset = HyperspectralDataset(data_dir, annotation_file_path, r_filter=range(3, 49))
    for i in range(len(dataset)):
        dataset.plot_spectrum_as_heatmap(i)