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

# perform hierarchical clustering
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
from scipy.spatial.distance import pdist
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

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
            if spectrum_objects != []:
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


        if self.augmentor is not None:
            augmentor = self.augmentor
            augmented_spectrum_1,augmented_spectrum_2 = augmentor.augment(preprocessed_spectrum, 2) # REQ: Only need 2 children of the chosen_spectrum
        else:
            augmented_spectrum_1 = preprocessed_spectrum
            augmented_spectrum_2 = preprocessed_spectrum

        augmented_spectrum_intensity_1 = torch.tensor(augmented_spectrum_1.intensity, dtype=torch.float32).unsqueeze(0)
        augmented_spectrum_intensity_2 = torch.tensor(augmented_spectrum_2.intensity, dtype=torch.float32).unsqueeze(0)

        cropped_spectrum = copy.deepcopy(chosen_spectrum)  
        # cropped_spectrum.display(f"original_spectrum{idx}")
        cropped_spectrum = cropped_spectrum.crop_spectrum(624.573, 1784.104)
        normalized_spectrum = cropped_spectrum.normalize_spectrum()

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

        # ax[0].set_ylim(2800, 4100) #2500, 6250
        # ax[1].set_ylim(-0.1, 1.3)
        # ax[2].set_ylim(-0.1, 1.3)
        # ax[3].set_ylim(-0.1, 1.3)

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
        # path_for_figure = os.path.join(os.getcwd(), "output_plots", "example_spectra_from_the_training_set.svg")
        # plt.savefig(path_for_figure, transparent=True)

        return augmented_spectrum_intensity_1, augmented_spectrum_intensity_2, chosen_spectrum.metadata, preprocessed_spectrum.raman_shift_cm#, cropped_spectrum.intensity, preprocessed_spectrum.intensity, normalized_spectrum.intensity
    
    def _extract_patient_labels(spectrum_file_name:str, 
                          all_patient_labels:pd.DataFrame):

        # Patient metatdata extraction
        patient_labels = {}
        f_split = spectrum_file_name.split('_')
        patient_id = int(f_split[0])
        sample_type = f_split[1]

        # Extract the metadata for the given patient_id
        if patient_id in all_patient_labels['OD Number'].values:
            patient_labels['patient_id'] = patient_id
            patient_labels['sample_type'] = sample_type

            patient_metadata_row = all_patient_labels[all_patient_labels['OD Number'] == patient_id]

            if len(patient_metadata_row) > 1:
                patient_metadata_row = patient_metadata_row.iloc[[0]]
                print(f"Patient ID {patient_id} has multiple entries in the metadata file")
                print("Only the first entry will be used")
            
            # map the staging to a number 0, 1, 2 (healthy, early stage, late stage)
            if patient_metadata_row['Staging'].values[0] == 0:
                patient_labels['staging'] = int(0)
            elif patient_metadata_row['Staging'].values[0] == 1 or patient_metadata_row['Staging'].values[0] == 2:
                patient_labels['staging'] = int(1)
            elif patient_metadata_row['Staging'].values[0] == 3 or patient_metadata_row['Staging'].values[0] == 4:
                patient_labels['staging'] = int(2)

            # patient_labels['staging'] = patient_metadata_row['Staging'].values[0]
            patient_labels['gender'] = patient_metadata_row['Gender'].values[0]
            patient_labels['race'] = patient_metadata_row['Race'].values[0]
            return patient_labels
        else:
            # skip the spectrum if the patient_id is not found in the metadata file
            print(f"Patient ID {patient_id} not found in the metadata file")
            print("Metadata set to empty strings and numbers")
        return {}
    
    def _load_files_to_spectrum_objects(data_folder:str,
                               filename:str, 
                               patient_annotations:dict):
        if patient_annotations == {}: # skip the spectrum if the patient_id is not found in the metadata file
            return []
        
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
        mean_spectra_above_throshold = []
        mean_spectra_below_throshold = []
        max_spectra_above_throshold = []
        max_spectra_below_throshold = []

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
            if max_intensity > 1000:
                max_spectra_above_throshold.append(spectrum.intensity)
            else:
                max_spectra_below_throshold.append(spectrum.intensity)

            if np.mean(spectrum.intensity) > 1000:
                mean_spectra_above_throshold.append(spectrum.intensity)
            else:
                mean_spectra_below_throshold.append(spectrum.intensity)


        # plot each spectrum for above and below the threshold
        plt.figure()
        plt.title("Spectra with the highest intensity above 2500")
        for spectrum in max_spectra_above_throshold:
            plt.plot(spectrum)
        plt.show()
        plt.figure()
        plt.title("Spectra with the highest intensity below 2500")
        for spectrum in max_spectra_below_throshold:
            plt.plot(spectrum)

        plt.figure()
        plt.title("Spectra with the mean intensity above 2500")
        for spectrum in mean_spectra_above_throshold:
            plt.plot(spectrum)
        plt.show()
        plt.figure()
        plt.title("Spectra with the mean intensity below 2500")
        for spectrum in mean_spectra_below_throshold:
            plt.plot(spectrum)
        plt.show()

        plt.figure()
        plt.hist(highest_intensity, bins=2000)
        plt.title("Distribution of the highest intensity peaks of db")
        plt.show()

        plt.figure()
        plt.hist(mean_intensity, bins=2000)
        plt.title("Distribution of the mean intensity of db")
        plt.show()
        mean_peak = np.mean(highest_intensity)
        std_peak = np.std(highest_intensity)
        print(f"Mean peak intensity: {mean_peak}")
        print(f"Standard deviation of peak intensity: {std_peak}")
        return
    
    def hierarchical_clustering(self, threshold):
        """
        Perform hierarchical clustering on the dataset
        """
        # hierarchical clustering on cropped and normalized spectra
        # get the cropped and normalized spectra
        cropped_spectra = []
        for ori_spectrum in self.db:
            # copy the spectrum
            spectrum = copy.deepcopy(ori_spectrum)
            cropped_spectrum = spectrum.crop_spectrum(624.573, 1784.104)
            #cropped_spectrum = cropped_spectrum.normalize_spectrum()
            cropped_spectra.append(cropped_spectrum.intensity)
        # convert to numpy array
        cropped_spectra = np.array(cropped_spectra)

        # calculate the distance matrix
        distance_matrix = pdist(cropped_spectra, metric='euclidean')
        Z = linkage(distance_matrix, method='complete')
        clusters = cut_tree(Z, height=threshold)
        num_clusters = np.max(clusters) + 1
        # plot the dendrogram
        plt.figure(figsize=(10, 5))
        plt.title("Dendrogram of the dataset")
        dendrogram(Z, color_threshold= threshold,above_threshold_color="#808080")
        plt.savefig(f'Dendrogram clustering with threshold {threshold}.png')

        ## Example spectra for each cluster
        fig, axes = plt.subplots(num_clusters, 1, sharex=True, figsize=(10, num_clusters * 2))
        colors = plt.cm.get_cmap('tab20', num_clusters)
        raman_shift = spectrum.raman_shift_cm

        for i in range(num_clusters):
            cluster_spectra = cropped_spectra[clusters.flatten()== i]
            for j in range(min(300, len(cluster_spectra))):
                sns.lineplot(x=raman_shift, y=cluster_spectra[j], color=colors(i), alpha=0.5, ax=axes[i])
                # add text to show the total number of spectra in the cluster
                axes[i].text(0.05, 0.95, f"Cluster {i} Total spectra: {len(cluster_spectra)}", transform=axes[i].transAxes, fontsize=12, verticalalignment='top', color=colors(i))
        plt.xlabel("Raman shift (cm^-1)")
        plt.ylabel("Intensity (a.u.)")
        plt.title("Example Spectra clustering")
        plt.tight_layout()
        plt.savefig(f'Example Spectra clustering with threshold {threshold}.png')

        # perform T-SNE on the dataset
        tsne = TSNE(n_components=2, random_state=0)
        tsne_results = tsne.fit_transform(cropped_spectra)
        plt.figure(figsize=(10, 5))
        plt.title("T-SNE of the dataset")
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=clusters.flatten(), cmap='tab20')
        plt.colorbar()
        plt.savefig(f'T-SNE clustering with threshold {threshold}.png')

        # t-SNE of the dataset colored by staging in the metadata
        plt.figure(figsize=(10, 5))
        plt.title("T-SNE of the dataset colored by staging")
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=[s.metadata['staging'] for s in self.db], cmap='rainbow')
        plt.colorbar()
        plt.savefig(f'T-SNE clustering with threshold {threshold} colored by staging.png')

        # add the clusters to the metadata in the database
        for i in range(len(self.db)):
            self.db[i].metadata['cluster'] = clusters[i][0]
        return
    
    def remove_clusters_from_db(self, clusters_index_to_remove:list):
        """
        Remove the clusters from the database
        """
        # remove the clusters from the database
        self.db = [spectrum for spectrum in self.db if spectrum.metadata['cluster'] not in clusters_index_to_remove]
        print(f"Removed {len(clusters_index_to_remove)} clusters from the database")
        return
        
    
if __name__ == "__main__":
    data_folder = os.path.join(os.getcwd(), "noodlepy", "data", "head_and_neck_cancer","plasma_cleaned")
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
                                        baseline_correction=False,
                                        remove_cosmic_rays= False,
                                        normalization=True,
                                        smoothing=False)
    
    augmentor = SpectrumAugmentor(ramdom_augmentations=True,
                                  augmentation_step_list = None,
                                  config_path= None)

    dataset = HNC_Dataset(data_folder, metadata_file, preprocessor, augmentor)

    # augmented_spectra = []

    # for i in range(1):
    #     ## get a random spectrum
    #     #idx = random.randint(0, dataset.__len__() - 1)
    #     example_spectrum = dataset.__getitem__(1)
    #     augmented_spectra.append(example_spectrum[0])
    #     augmented_spectra.append(example_spectrum[1])


    ######################################################
    # plt.figure()
    # # plot all the augmented spectra
    # # for i in range(len(augmented_spectra)):
    # #     plt.plot(raman_shift, augmented_spectra[i].numpy().flatten(), alpha=0.5, color="blue")
    # # add label for one of the augmented spectra
    # # plt.plot(raman_shift, augmented_spectra[0].numpy().flatten(), label="Augmented spectra", alpha=0.5, color="blue")
    
    # # plot mean spectrum
    # raman_shift = example_spectrum[3]
    # plt.plot(raman_shift, np.mean([s.numpy().flatten() for s in augmented_spectra], axis=0), label="Mean of augmented spectra", color="red")
    
    # # plot original spectrum
    # plt.plot(raman_shift, example_spectrum[5], label="preprocessed preprocessed spectrum", color="black")
    # plt.plot(raman_shift, example_spectrum[6], label="original spectrum", color="y")
    
    # # plt the std band
    # std_band = np.std([s.numpy().flatten() for s in augmented_spectra], axis=0)
    # plt.fill_between(raman_shift, np.mean([s.numpy().flatten() for s in augmented_spectra], axis=0) - std_band, np.mean([s.numpy().flatten() for s in augmented_spectra], axis=0) + std_band, color="y", alpha=0.5, label="Standard deviation band of augmented spectra")
    # plt.xlabel("Raman shift (cm^-1)")
    # plt.ylabel("Normalized Intensity (a.u.)")
    # plt.legend()
    # plt.title("Augmented the same spectrum 1000 times")
    # plt.show()

    ######################################################

    # dataset.get_peak_distribution()

    ######################################################

    dataset.hierarchical_clustering(threshold=10000)
    dataset.remove_clusters_from_db([5,6])

    ######################################################


    print("Done")