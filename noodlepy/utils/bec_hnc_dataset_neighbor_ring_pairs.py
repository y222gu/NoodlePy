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
        if r_filter is None:
            r_filter = np.arange(1, 51)
        self.r_filter = r_filter

        list_of_file_paths = []
        for root, dirs, files in os.walk(data_folder):
            if root == data_folder or root.count(os.sep) == data_folder.count(os.sep) + 1:
                list_of_file_paths += [os.path.relpath(os.path.join(root, f), data_folder) for f in files if f.endswith('.txt')]
        list_of_file_paths = sorted(list_of_file_paths)

        annotation_all = pd.read_excel(annotation_file_path)

        db_1 = []
        db_2 = []

        # assign every other ring to db_1 and db_2
        # sort r_filter to split the rings into 2 lists alternatively
        r_filter = sorted(r_filter)
        db_1_rings = r_filter[::2]
        db_2_rings = r_filter[1::2]

        for file_path in list_of_file_paths:
            patient_annotations = Bec_HNC_Dataset._extract_patient_labels(file_path, self.r_filter, annotation_all)
            spectrum_objects = Bec_HNC_Dataset._load_files_to_spectrum_objects(data_folder, file_path, patient_annotations)
            # split the spectrum objects into 2 lists based on ring
            for spectrum in spectrum_objects:
                if spectrum.metadata['ring'] in db_1_rings:
                    db_1.append(spectrum)

                elif spectrum.metadata['ring'] in db_2_rings:
                    db_2.append(spectrum)

        self.db = []
        for spectrum in db_1:
            # check if it can find the spectrum with the same patient_id, date, ring + 1, line in db_2
            patient_id = spectrum.metadata['patient_id']
            date = spectrum.metadata['date']
            ring = spectrum.metadata['ring'] + 1
            line = spectrum.metadata['line']
            spectrum_id = spectrum.metadata['spectrum_id']

            for spectrum_2 in db_2:
                if spectrum_2.metadata['patient_id'] == patient_id and spectrum_2.metadata['date'] == date and spectrum_2.metadata['ring'] == ring and spectrum_2.metadata['line'] == line and spectrum_2.metadata['spectrum_id'] == spectrum_id:
                    self.db.append([spectrum, spectrum_2])
                    break


        print(f"Loaded {len(self.db)} spectra pairs")

    def __len__(self):
        return len(self.db)
    
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
        chosen_spectrum_1, chosen_spectrum_2 = copy.deepcopy(self.db[idx])
        
        cropped_spectrum_1 = chosen_spectrum_1.crop_spectrum(624.573, 1784.104)
        cropped_spectrum_2 = chosen_spectrum_2.crop_spectrum(624.573, 1784.104)

        cropped_spectrum_intensity_1 = torch.tensor(cropped_spectrum_1.intensity, dtype=torch.float32).unsqueeze(0)
        cropped_spectrum_intensity_2 = torch.tensor(cropped_spectrum_2.intensity, dtype=torch.float32).unsqueeze(0)
        
        return cropped_spectrum_intensity_1, cropped_spectrum_intensity_2, chosen_spectrum_1.metadata, chosen_spectrum_1.raman_shift_cm
    
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
    
    def plot_spectrum_pair(self, idx):
        """
        Plot the spectrum.intentsity 
        """
        chosen_spectrum_1, chosen_spectrum_2, metadata, raman_shift_cm = self.__getitem__(idx)
        plt.figure(figsize=(10, 5))
        plt.plot(raman_shift_cm, chosen_spectrum_1.squeeze().numpy(), label='Spectrum 1')
        plt.plot(raman_shift_cm, chosen_spectrum_2.squeeze().numpy(), label='Spectrum 2')
        plt.title(f"Patient ID: {metadata['patient_id']}, Date: {metadata['date']}, Ring: {metadata['ring'], metadata['ring']+1}, Line: {metadata['line']}")
        plt.xlabel("Raman Shift (cm^-1)")
        plt.ylabel("Intensity")
        plt.legend()
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

    r_filter = list(range(3, 49)) # all rings
    dataset = Bec_HNC_Dataset(data_folder, metadata_file, r_filter)

    for i in range(8):
        dataset.plot_spectrum_pair(i)