# https://pytorch.org/get-started/locally/
from torch.utils.data import Dataset
import pandas as pd
import os
from noodlepy.utils.spectrum import Spectrum
from noodlepy.utils.spectrumpreprocessor import SpectrumPreprocessor
from noodlepy.utils.spectrumaugmentor import SpectrumAugmentor
import torch
import copy
import numpy as np
import matplotlib.pyplot as plt
import random
from PyQt5.QtWidgets import QApplication
import sys
from noodlepy.archive import spectrum_inspection as si
from combat.pycombat import pycombat
from sklearn.decomposition import PCA

class Bec_HNC_Dataset(Dataset):
    def __init__(self, data_folder = None,
                 annotation_file_path = None,
                 r_filter = None,
                 preprocessor = None,
                 augmentor = None):
        """
        Load the database of spectra from the txt file 

        Args:
        data_folder (str): The folder where the spectra are stored

        Returns:
        list[Spectrum]: A list of Spectrum objects
        """
        print("Loading the Raman dataset")
        self.r_filter = r_filter
        self.preprocessor = preprocessor
        self.augmentor = augmentor

        list_of_spectrum_objects = []
        list_of_file_paths = []
        for root, dirs, files in os.walk(data_folder):
            if root == data_folder or root.count(os.sep) == data_folder.count(os.sep) + 2:
                list_of_file_paths += [os.path.relpath(os.path.join(root, f), data_folder) for f in files if f.endswith('.txt')]
        list_of_file_paths = sorted(list_of_file_paths)

        annotation_all = pd.read_excel(annotation_file_path)

        for file_path in list_of_file_paths:
            patient_annotations = Bec_HNC_Dataset._extract_patient_labels(file_path, self.r_filter, annotation_all)
            spectrum_objects = Bec_HNC_Dataset._load_files_to_spectrum_objects(data_folder, file_path, patient_annotations)
            list_of_spectrum_objects+=spectrum_objects

        self.db = list_of_spectrum_objects
        print(f"Loaded {len(self.db)} spectra")

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
        augmentor = self.augmentor
        preprocessor = self.preprocessor
        # Crop, augment, preprocess, return 2 spectra
        chosen_spectrum:Spectrum = copy.deepcopy(self.db[idx])
        cropped_spectrum = chosen_spectrum.crop_spectrum(624.573, 1784.104)
        original_preprocessed_spectrum = preprocessor.preprocess(cropped_spectrum)

        if self.augmentor is not None:
            augmented_spectrum_1, augmented_spectrum_2 = augmentor.augment(cropped_spectrum, 2) # REQ: Only need 2 children of the chosen_spectrum
        else:
            augmented_spectrum_1 = cropped_spectrum
            augmented_spectrum_2 = cropped_spectrum

        if self.preprocessor is not None:
            preprocessed_spectrum_1 = preprocessor.preprocess(augmented_spectrum_1)
            preprocessed_spectrum_2 = preprocessor.preprocess(augmented_spectrum_2)
        else:
            preprocessed_spectrum_1 = augmented_spectrum_1
            preprocessed_spectrum_2 = augmented_spectrum_2

        preprocessed_spectrum_intensity_1 = torch.tensor(preprocessed_spectrum_1.intensity, dtype=torch.float32).unsqueeze(0)
        preprocessed_spectrum_intensity_2 = torch.tensor(preprocessed_spectrum_2.intensity, dtype=torch.float32).unsqueeze(0)

        return preprocessed_spectrum_intensity_1, preprocessed_spectrum_1.raman_shift_cm, chosen_spectrum.metadata
    
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
    
if __name__ == "__main__":
    data_folder = os.path.join(os.getcwd(), "noodlepy", "data", "bec_hnc_train")
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

    preprocessor = SpectrumPreprocessor(cropping=False,
                                        baseline_correction=True,
                                        remove_cosmic_rays= True,
                                        normalization=True,
                                        smoothing=True)
    
    augmentor = SpectrumAugmentor(ramdom_augmentations=False,
                                  augmentation_step_list = None,
                                  config_path= None)

    r_filter = None
    dataset = Bec_HNC_Dataset(data_folder, metadata_file, r_filter, preprocessor, augmentor= None)

    # for i in range(50):
    #     ## get a random spectrum
    #     #idx = random.randint(0, dataset.__len__() - 1)
    #     example_spectrum = dataset.__getitem__(i)

    app = QApplication(sys.argv)
    window = si.SpectraViewer(dataset.db, preprocessor)
    window.resize(1000, 500)
    window.show()
    sys.exit(app.exec_())

    print("Done")