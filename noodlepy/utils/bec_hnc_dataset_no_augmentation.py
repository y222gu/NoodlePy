# https://pytorch.org/get-started/locally/
from torch.utils.data import Dataset
import pandas as pd
import os
from noodlepy.utils.spectrum import Spectrum
import torch
import copy
import numpy as np
import matplotlib.pyplot as plt
import random
from combat.pycombat import pycombat
from sklearn.decomposition import PCA

class Bec_HNC_Dataset(Dataset):
    def __init__(self, data_folder = None,
                 annotation_file_path = None,
                 r_filter = None):
        """
        Load the database of spectra from the txt file 

        Args:
        data_folder (str): The folder where the spectra are stored

        Returns:
        list[Spectrum]: A list of Spectrum objects
        """
        print("Loading the Raman dataset")
        self.r_filter = r_filter

        list_of_file_paths = []
        for root, dirs, files in os.walk(data_folder):
            if root == data_folder or root.count(os.sep) == data_folder.count(os.sep) + 1:
                list_of_file_paths += [os.path.relpath(os.path.join(root, f), data_folder) for f in files if f.endswith('.txt')]
        list_of_file_paths = sorted(list_of_file_paths)

        annotation_all = pd.read_excel(annotation_file_path)

        self.db_1 = []
        self.db_2 = []

        for file_path in list_of_file_paths:
            patient_annotations = Bec_HNC_Dataset._extract_patient_labels(file_path, self.r_filter, annotation_all)
            spectrum_objects = Bec_HNC_Dataset._load_files_to_spectrum_objects(data_folder, file_path, patient_annotations)
            # split the spectrum objects into 2 lists based on spectrum_id
            # first check if the number of spectra in the file, onlu continue if there are more 2 spectra in the file
            if len(spectrum_objects) < 2:
                print(f"File {file_path} has less than 2 spectra, skipping")
            else:
                # put the last one in db_2 and the rest in db_1
                self.db_1 += spectrum_objects[:-1]
                self.db_2 += spectrum_objects[-1:]

        print(f"Loaded {len(self.db_1)} spectra in db_1")
        print(f"Loaded {len(self.db_2)} spectra in db_2")

    def __len__(self):
        return len(self.db_1)
    
    def __getitem__(self, 
                    idx:int
                    )-> tuple[Spectrum, Spectrum]:
        """
        Return 2 augmented spectra from the chosen spectrum

        Args:
        idx (int): The index of the spectrum to augment
        preprocessing_flag (bool): Whether to apply preprocessing to the chosen_spectrum
        augmentation_step_option_list (list[str]): The list of augmentation steps to choose and apply randomly

        returns:
        augmented_spectrum_list (list[Spectrum]): a tuple of 2 augmented Spectrum objects
        """
        # Crop, augment, preprocess, return 2 spectra
        chosen_spectrum:Spectrum = copy.deepcopy(self.db_1[idx])
        chosen_spectrum_2:Spectrum= None

        # find the spectrum with the same patient_id, date, ring, line in db_2
        patient_id = chosen_spectrum.metadata['patient_id']
        date = chosen_spectrum.metadata['date']
        ring = chosen_spectrum.metadata['ring']
        line = chosen_spectrum.metadata['line']

        for spectrum in self.db_2:
            if spectrum.metadata['patient_id'] == patient_id and spectrum.metadata['date'] == date and spectrum.metadata['ring'] == ring and spectrum.metadata['line'] == line:
                chosen_spectrum_2 = copy.deepcopy(spectrum)
                break
        if chosen_spectrum_2 is None:
            print(f"Could not find spectrum with patient_id {patient_id}, date {date}, ring {ring}, line {line} in db_2")
            return None, None
        
        cropped_spectrum_1 = chosen_spectrum.crop_spectrum(624.573, 1784.104)
        cropped_spectrum_2 = chosen_spectrum_2.crop_spectrum(624.573, 1784.104)

        cropped_spectrum_intensity_1 = torch.tensor(cropped_spectrum_1.intensity, dtype=torch.float32).unsqueeze(0)
        cropped_spectrum_intensity_2 = torch.tensor(cropped_spectrum_2.intensity, dtype=torch.float32).unsqueeze(0)
        
        return cropped_spectrum_intensity_1, cropped_spectrum_intensity_2, chosen_spectrum.metadata, chosen_spectrum_2.raman_shift_cm
    
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
    
    def combat_batch_correction(self):
        """
        Applies ComBat batch correction to the intensity values of all spectra in the dataset.
        This function modifies the dataset's intensities while keeping Raman shift and metadata intact.
        """
        if not self.db:
            print("Dataset is empty. No batch correction applied.")
            return

        # Extract intensity values and batch labels (using 'date' as batch label).
        intensity_matrix = np.array([spectrum.intensity for spectrum in self.db])
        batch_labels = pd.Series([spectrum.metadata['date'] for spectrum in self.db])  # Convert to Pandas Series

        # Convert batch labels to categorical numeric labels.
        batch_categories = pd.factorize(batch_labels)[0]

        # Transpose data so that features are rows.
        data_transposed = pd.DataFrame(intensity_matrix.T)  # Convert NumPy array to DataFrame

        print("Data transposed shape:", data_transposed.shape)
        print("Batch categories shape:", len(batch_categories))

        # Apply ComBat for batch effect correction.
        corrected_data_transposed = pycombat(data_transposed, batch_categories)  # Now using a DataFrame

        # Convert back to NumPy array and transpose to original shape.
        corrected_intensity_matrix = corrected_data_transposed.to_numpy().T  # Convert DataFrame to NumPy and transpose back

        # Replace original intensity values while keeping metadata and Raman shift intact.
        for i, spectrum in enumerate(self.db):
            self.db[i].intensity = corrected_intensity_matrix[i]  # Update intensity

        print("Batch effect correction using ComBat has been applied successfully.")

        # Plot PCA of corrected data color coded by batch. 
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(corrected_intensity_matrix)
        plt.scatter(pca_data[:, 0], pca_data[:, 1], c=batch_categories)
        plt.title("PCA of corrected data color coded by batch")
        plt.savefig("PCA_corrected_data.png")
        
        return self.db  # Return corrected dataset
    
    def plot_spectrum_as_heatmap(self, patient_id,date):
        """
        Plot the spectrum.intentsity as a heatmap for a given patient_id and date.
        The output should be 25x8x745 heatmap with the x-axis ring (50), y-axis line (4), and z-axis intensity (1024).
        """
        # Filter the dataset for the given patient_id and date
        filtered_spectra = [spectrum for spectrum in self.db_1 if spectrum.metadata['patient_id'] == patient_id and spectrum.metadata['date'] == date]
        if not filtered_spectra:
            print(f"No spectra found for patient_id {patient_id} and date {date}")
            return
        
        # loop through the intensity values of different rings and lines
        intensity = np.zeros((1024, 4, 50))
        for spectrum in filtered_spectra:
            ring = spectrum.metadata['ring'] - 1
            line = spectrum.metadata['line'] - 1
            # normalize the intensity values with under area to 1
            spectrum.intensity = spectrum.intensity / np.trapz(spectrum.intensity, spectrum.raman_shift_cm)
            intensity[:, line, ring] = spectrum.intensity

        # plot random 5 layers of 1024 layer as a heatmap of 4*50
        fig, ax = plt.subplots(5, 1, figsize=(10, 10), sharex=True)
        layers = random.sample(range(1024), 5)
        for i in range(5):
            ax[i].imshow(intensity[layers[i]], aspect='auto', cmap='hot', interpolation='nearest')
            ax[i].set_title(f"Wavenumber {layers[i]}")
            ax[i].set_ylabel("Line")
            ax[i].set_xlabel("Ring")
            ax[i].set_xticks(np.arange(0, 50, 5))
            ax[i].set_xticklabels(np.arange(0, 50, 5))
            # colorbar
            cbar = plt.colorbar(ax[i].imshow(intensity[i], aspect='auto', cmap='hot', interpolation='nearest'), ax=ax[i])
            cbar.set_label("Intensity (a.u.)")
        plt.tight_layout()
        plt.show()
        return



    
if __name__ == "__main__":
    data_folder = os.path.join(os.getcwd(), "noodlepy", "data", "explore", "train")
    metadata_file = os.path.join(os.getcwd(), "noodlepy", "data", "bec_hnc_patient_annotations.xlsx")

    seed = 4
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    r_filter = None
    dataset = Bec_HNC_Dataset(data_folder, metadata_file, r_filter)

    # for i in range(50):
    #     example_spectrum_1, example_spectrum_2, meta, raman_shift = dataset.__getitem__(i)
    #     print(f"Example spectrum 1: {example_spectrum_1.shape}, Example spectrum 2: {example_spectrum_2.shape}")
    #     print(f"Meta data: {meta}")
    #     # plot the spectrum
    #     plt.plot(example_spectrum_1.squeeze(0).numpy(), label="Example spectrum 1")
    #     plt.plot(example_spectrum_2.squeeze(0).numpy(), label="Example spectrum 2")
    #     plt.title(f"Example spectrum {i}")
    #     plt.xlabel("Wavenumber (cm-1)")
    #     plt.ylabel("Intensity (a.u.)")
    #     plt.legend(["Example spectrum 1", "Example spectrum 2"])
    #     plt.show()

    dataset.plot_spectrum_as_heatmap(309, '20240425')