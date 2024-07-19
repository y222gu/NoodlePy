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
class HNC_Dataset(Dataset):
    def __init__(self, data_folder = None,
                 annotation_file_path = None,
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
        self.preprocessor = preprocessor
        self.augmentor = augmentor

        list_of_spectrum_objects = []
        list_of_file_names = sorted([f for f in os.listdir(data_folder) if f.endswith(('.txt'))])

        annotation_all = pd.read_excel(annotation_file_path)

        for filename in list_of_file_names:
            patient_annotations = HNC_Dataset._extract_patient_labels(filename, annotation_all)
            spectrum_objects = HNC_Dataset._load_files_to_spectrum_objects(data_folder, filename, patient_annotations)
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
        chosen_spectrum:Spectrum = self.db[idx]


        if self.preprocessor is not None:
            preprocessor = self.preprocessor
            preprocessed_spectrum = preprocessor.preprocess(chosen_spectrum)
        else:
            preprocessed_spectrum = chosen_spectrum
        
        # preprocessed_spectrum.display("preprocessed_spectrum")


        if self.augmentor is not None:
            augmentor = self.augmentor
            augmented_spectrum_1,augmented_spectrum_2 = augmentor.augment(preprocessed_spectrum, 2) # REQ: Only need 2 children of the chosen_spectrum
        else:
            augmented_spectrum_1 = preprocessed_spectrum
            augmented_spectrum_2 = preprocessed_spectrum

        augmented_spectrum_intensity_1 = torch.tensor(augmented_spectrum_1.intensity, dtype=torch.float32).unsqueeze(0)
        augmented_spectrum_intensity_2 = torch.tensor(augmented_spectrum_2.intensity, dtype=torch.float32).unsqueeze(0)

        # cropped_spectrum = copy.deepcopy(chosen_spectrum)        
        # cropped_spectrum = cropped_spectrum.crop_spectrum(624.573, 1784.104)
        # cropped_spectrum.display(f"cropped_spectrum_{idx}")
        # preprocessed_spectrum.display(f"preprocessed_spectrum_{idx}")
        # augmented_spectrum_1.display(f"augmented_spectrum_1_{idx}")
        # augmented_spectrum_2.display(f"augmented_spectrum_2_{idx}")

        # fig, ax = plt.subplots(4,1,figsize=(12, 10),sharex=True, gridspec_kw={'hspace': 0})
        # fig.suptitle("Spectra Augmentation Example", fontsize=25, color = "white")
        # fig.supylabel("Intensity (a.u.)", fontsize=25, color = "white")

        # ax[0].plot(cropped_spectrum.raman_shift_cm, cropped_spectrum.intensity, linewidth=2, color = "rebeccapurple", label = "Raw spectrum")
        # ax[1].plot(preprocessed_spectrum.raman_shift_cm, preprocessed_spectrum.intensity, linewidth=2, color = "mediumslateblue", label = "Preprocessed spectrum")
        # ax[2].plot(augmented_spectrum_1.raman_shift_cm, augmented_spectrum_1.intensity, linewidth=2, color = "lightskyblue", label = "Augmented spectrum 1")
        # ax[3].plot(augmented_spectrum_2.raman_shift_cm, augmented_spectrum_2.intensity, linewidth=2, color = "tab:blue", label = "Augmented spectrum 2")

        # ax[0].legend(loc='upper left', fontsize=20, facecolor='none', edgecolor='rebeccapurple', labelcolor='white')
        # ax[1].legend(loc='upper left', fontsize=20, facecolor='none', edgecolor='mediumslateblue', labelcolor='white')
        # ax[2].legend(loc='upper left', fontsize=20, facecolor='none', edgecolor='lightskyblue', labelcolor='white')
        # ax[3].legend(loc='upper left', fontsize=20, facecolor='none', edgecolor='tab:blue', labelcolor='white')

        # ax[0].set_ylim(2500, 6250)
        # ax[1].set_ylim(-0.1, 1.1)
        # ax[2].set_ylim(2500, 6250)
        # ax[3].set_ylim(2500, 6250)

        # for ax in fig.get_axes():
        #     ax.label_outer(remove_inner_ticks= True)
        #     ax.spines['top'].set_color('white')
        #     ax.spines['top'].set_linewidth(1.5)
        #     ax.spines['right'].set_color('white')
        #     ax.spines['right'].set_linewidth(1.5)
        #     ax.spines['bottom'].set_color('white')
        #     ax.spines['bottom'].set_linewidth(1.5)
        #     ax.spines['left'].set_color('white')
        #     ax.spines['left'].set_linewidth(1.5)
        #     ax.title.set_color('white')
        #     ax.xaxis.label.set_color('white')
        #     ax.yaxis.label.set_color('white')
        #     ax.tick_params(axis='x', which= 'major',colors='white', labelsize=25)
        #     ax.tick_params(axis='y', which= 'major',colors='white', labelsize=25)
        #     ax.yaxis.label.set_size(25)
        #     ax.xaxis.label.set_size(25)
        #     ax.set_xlim(624.573, 1782.711)

        # plt.xlabel("Raman Shift (cm^-1)", fontsize=25)
        # plt.subplots_adjust(hspace=0)
        # plt.tight_layout()
        # path_for_figure = os.path.join(os.getcwd(), "output_plots", "example_spectra_from_the_training_set.png")
        # plt.savefig(path_for_figure, transparent=True)

        return augmented_spectrum_intensity_1, augmented_spectrum_intensity_2, chosen_spectrum.metadata
    
    def _extract_patient_labels(spectrum_file_name:str, 
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
            patient_labels['gender'] = patient_metadata_row['Gender'].values[0]
            patient_labels['race'] = patient_metadata_row['Race'].values[0]

        else:
            # If the patient_id is not found in the metadata file, set the every metadata to empty string and number
            patient_labels['staging'] = np.nan
            patient_labels['gender'] = ''
            patient_labels['race'] = ''
        
            print(f"Patient ID {patient_id} not found in the metadata file")
            print("Metadata set to empty strings and numbers")
        return patient_labels
    
    def _load_files_to_spectrum_objects(data_folder:str,
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
    
    def get_peak_distribution(self):
        """
        Get the distribution of the highest intensity peaks
        """
        # get the highest intensity
        highest_intensity = []
        mean_intensity = []
        for spectrum in self.db:
            # smooth the spectrum
            spectrum.savgol_filter(window_length=5, polyorder=3)
            # get the highest intensity
            max_intensity = max(spectrum.intensity)
            highest_intensity.append(max_intensity)
            mean_intensity.append(np.mean(spectrum.intensity))

        # plt.figure()
        # plt.hist(highest_intensity, bins=1000)
        # plt.show()

        # plt.figure()
        # plt.hist(mean_intensity, bins=1000)
        # plt.show()
        mean_peak = np.mean(highest_intensity)
        std_peak = np.std(highest_intensity)
        return mean_peak, std_peak
    
    def get_cosmic_ray_counts_distribution(self):
        """
        Get the distribution of the cosmic ray counts
        """
        cosmic_ray_counts = []
        for spectrum in self.db:
            cosmic_ray_counts.append(spectrum.count_number_of_cosmic_rays())
        
        plt.figure()
        plt.hist(cosmic_ray_counts, bins=1000)
        plt.xlim(0, 50)
        mean_cosmic_ray_count = np.mean(cosmic_ray_counts)
        std_cosmic_ray_count = np.std(cosmic_ray_counts)
        return mean_cosmic_ray_count, std_cosmic_ray_count
    
if __name__ == "__main__":
    data_folder = os.path.join(os.getcwd(), "noodlepy", "data", "head_and_neck_cancer", "plasma_saliva_mixed","train")
    metadata_file = os.path.join(os.getcwd(), "noodlepy", "data", "Biofluid_list_annotated_v4.xlsx")

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

    preprocessor = SpectrumPreprocessor(cropping=True,
                                        baseline_correction=True,
                                        remove_cosmic_rays= True,
                                        normalization=True,
                                        smoothing=True)
    
    augmentor = SpectrumAugmentor(ramdom_augmentations=False,
                                  augmentation_step_list = None,
                                  config_path= None)

    dataset = HNC_Dataset(data_folder, metadata_file, preprocessor, augmentor)

    for i in range(5150):
        ## get a random spectrum
        #idx = random.randint(0, dataset.__len__() - 1)
        example_spectrum = dataset.__getitem__(i)

    # dataset.get_peak_distribution()
    # dataset.get_cosmic_ray_counts_distribution()

    print("Done")