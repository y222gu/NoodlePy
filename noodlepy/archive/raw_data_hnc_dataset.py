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
import matplotlib.colors as mcolors
from noodlepy.archive import spectrum_inspection as si
import sys
from PyQt5.QtWidgets import QApplication

# perform hierarchical clustering
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
from scipy.spatial.distance import pdist
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import json

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
        list_of_file_names = sorted([os.path.join(root, f) for root, _, files in os.walk(data_folder) for f in files if f.endswith('.txt')])

        annotation_all = pd.read_excel(annotation_file_path)

        for filename in list_of_file_names:
            patient_annotations = HNC_Dataset._extract_patient_labels(filename, annotation_all)
            spectrum_objects = HNC_Dataset._load_files_to_spectrum_objects(data_folder, filename, patient_annotations)
            if spectrum_objects != []:
                list_of_spectrum_objects+=spectrum_objects

        # preprocess the spectra
        # preprocessed_spectra = []
        # for spectrum in list_of_spectrum_objects:
        #     spectrum = self.preprocessor.preprocess(spectrum)
        #     preprocessed_spectra.append(spectrum)

        # self.db = preprocessed_spectra
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

        # cropped_spectrum = copy.deepcopy(chosen_spectrum)  
        # # cropped_spectrum.display(f"original_spectrum{idx}")
        # cropped_spectrum = cropped_spectrum.crop_spectrum(624.573, 1784.104)
        # normalized_spectrum = cropped_spectrum.normalize_spectrum()

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

        return preprocessed_spectrum_intensity_1, preprocessed_spectrum_intensity_2, chosen_spectrum.metadata, cropped_spectrum.raman_shift_cm#, augmented_spectrum_1.intensity, augmented_spectrum_2.intensity, original_preprocessed_spectrum.intensity #, cropped_spectrum.intensity, preprocessed_spectrum.intensity, normalized_spectrum.intensity
    
    def _extract_patient_labels(spectrum_file_path:str, 
                          all_patient_labels:pd.DataFrame):

        # Patient metatdata extraction
        patient_labels = {}
        spectrum_file_name = os.path.basename(spectrum_file_path)
        f_split = spectrum_file_name.split('_')
        date = f_split[0]
        patient_id = int(f_split[1])
        sample_type = f_split[2]
        position = f_split[-1].split('.')[0]

        # Extract the metadata for the given patient_id
        if patient_id in all_patient_labels['OD Number'].values:
            patient_labels['patient_id'] = patient_id
            patient_labels['sample_type'] = sample_type
            patient_labels['date'] = date
            patient_labels['position'] = position

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
    
    def get_peak_distribution(self):
        """
        Get the distribution of the highest intensity peaks
        """
        max_spectra_above_throshold = []
        max_spectra_below_throshold = []
        min_spectra_below_throshold = []

        # get the highest intensity
        highest_intensity = []
        min_intensity = []
        count = 0
        for spectrum in self.db:
            # smooth the spectrum
            spectrum.savgol_filter(window_length=5, polyorder=3)
            # get the highest intensity
            min_intensity.append(min(spectrum.intensity))
            max_intensity = max(spectrum.intensity)
            highest_intensity.append(max_intensity)
            if max_intensity > 20000:
                max_spectra_above_throshold.append(spectrum.intensity)
                print(spectrum.metadata)
                plt.figure()
                plt.plot(spectrum.raman_shift_cm, spectrum.intensity)
                count += 1
            else:
                max_spectra_below_throshold.append(spectrum.intensity)

            if min(spectrum.intensity) < 0:
                min_spectra_below_throshold.append(spectrum.intensity)
                print(spectrum.metadata)

        # # plot each spectrum for above and below the threshold
        # plt.figure()
        # plt.title("Spectra with the highest intensity above 2500")
        # for spectrum in max_spectra_above_throshold:
        #     plt.plot(spectrum)
        # plt.show()

        # plt.figure()
        # plt.title("Spectra with the highest intensity below 2500")
        # for spectrum in max_spectra_below_throshold:
        #     plt.plot(spectrum)

        # plt.figure()
        # plt.hist(highest_intensity, bins=2000)
        # plt.title("Distribution of the highest intensity peaks of db")
        # plt.show()

        plt.figure()
        plt.title("Spectra with the kowest intensity below 0")
        for spectrum in min_spectra_below_throshold:
            plt.plot(spectrum)
        plt.show()

        return
    
    def hierarchical_clustering(self, threshold, threshold_normalized, output_folder):
        """
        Perform hierarchical clustering on the dataset
        """
        # hierarchical clustering on cropped and normalized spectra
        # get the cropped and normalized spectra
        cropped_spectra = []
        preprocessed_spectra = []
        for ori_spectrum in self.db:
            # copy the spectrum
            spectrum = copy.deepcopy(ori_spectrum)

            cropped_spectrum = spectrum.crop_spectrum(624.573, 1784.104) #624.573, 1784.104
            cropped_spectra.append(cropped_spectrum)

        def plot_sample_spectra_and_tsne(spectra, clusters, num_clusters, threshold, title_prefix, color_map, output_folder):
            # Prep color map
            fig, axes = plt.subplots(num_clusters, 1, sharex=True, figsize=(10, num_clusters * 2))
            colors = plt.cm.get_cmap(color_map, num_clusters)
            raman_shift = spectrum.raman_shift_cm

            for i in range(num_clusters):
                cluster_spectra = [spectra[j] for j in np.where(clusters.flatten() == i)[0]]
                # Example spectra for each cluster
                for j in range(min(500, len(cluster_spectra))):
                    sns.lineplot(x=raman_shift, y=cluster_spectra[j].intensity, color=colors(i), alpha=0.5, ax=axes[i])
                    axes[i].text(0.05, 0.95, f"Cluster {i} Total spectra: {len(cluster_spectra)}", transform=axes[i].transAxes, fontsize=12, verticalalignment='top', color=colors(i))

                # if i == 1:
                #     # save the metadata of all the spectra in the clusters 1 into a json file
                #     cluster_1_spectra = [s.metadata for s in cluster_spectra]
                #     with open(os.path.join(output_folder, f'cluster_1_spectra_metadata.json'), 'w') as f:
                #         json.dump(cluster_1_spectra, f, default=lambda o: int(o) if isinstance(o, np.int64) else o)
            plt.xlabel("Raman shift (cm^-1)")
            plt.ylabel("Intensity (a.u.)")
            plt.title(f"Example Spectra clustering {title_prefix}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, f'Example Spectra {title_prefix} with threshold {threshold}.png'))



            # # T-SNE by cluster
            # spectra_intensities = [spectrum.intensity for spectrum in spectra]
            # spectra_intensities = np.array(spectra_intensities)
            # tsne_results = TSNE(n_components=2, random_state=0).fit_transform(spectra_intensities)
            
            # plt.figure(figsize=(10, 5))
            # plt.title(f"T-SNE of the {title_prefix}")
            # plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=clusters.flatten(), cmap=color_map)
            # plt.colorbar()
            # plt.savefig(os.path.join(output_folder, f'T-SNE {title_prefix} with threshold {threshold} by clustering.png'))

            # # T-SNE by staging
            # unique_stages = sorted(list(set([s.metadata['staging'] for s in spectra])))
            # cmap = mcolors.ListedColormap(plt.cm.gnuplot(np.linspace(1, 0, len(unique_stages))))
            # norm = mcolors.BoundaryNorm(boundaries=np.arange(len(unique_stages) + 1) - 0.5, ncolors=len(unique_stages))
            # stage_to_index = {stage: i for i, stage in enumerate(unique_stages)}
            # stage_indices = np.array([stage_to_index[s.metadata['staging']] for s in spectra])
            # plt.figure(figsize=(10, 5))
            # plt.title(f"T-SNE of the {title_prefix} by staging")
            # sc = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=stage_indices, cmap=cmap, norm=norm)
            # cbar = plt.colorbar(sc, ticks=np.arange(len(unique_stages)))
            # cbar.ax.set_yticklabels(unique_stages)  # Assign unique stages as labels
            # plt.savefig(os.path.join(output_folder, f'T-SNE clustering {title_prefix} with threshold {threshold} by stage.png'))

            # # T-SNE by date
            # unique_dates = sorted(list(set([s.metadata['date'] for s in spectra])))
            # cmap = mcolors.ListedColormap(plt.cm.cool(np.linspace(0, 1, len(unique_dates)))) #winter
            # norm = mcolors.BoundaryNorm(boundaries=np.arange(len(unique_dates) + 1) - 0.5, ncolors=len(unique_dates))
            # date_to_index = {date: i for i, date in enumerate(unique_dates)}
            # date_indices = np.array([date_to_index[s.metadata['date']] for s in spectra])
            # plt.figure(figsize=(10, 5))
            # plt.title(f"T-SNE of the {title_prefix} by date")
            # sc = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=date_indices, cmap=cmap, norm=norm)
            # cbar = plt.colorbar(sc, ticks=np.arange(len(unique_dates)))
            # cbar.ax.set_yticklabels(unique_dates)  # Assign unique dates as labels
            # plt.savefig(os.path.join(output_folder, f'T-SNE clustering {title_prefix} with threshold {threshold} by date.png'))

            # # T-SNE by position
            # unique_positions = sorted(list(set([s.metadata['position'] for s in spectra])))
            # cmap = mcolors.ListedColormap(plt.cm.cool(np.linspace(0, 1, len(unique_positions))))
            # norm = mcolors.BoundaryNorm(boundaries=np.arange(len(unique_positions) + 1) - 0.5, ncolors=len(unique_positions))
            # position_to_index = {position: i for i, position in enumerate(unique_positions)}
            # position_indices = np.array([position_to_index[s.metadata['position']] for s in spectra])
            # plt.figure(figsize=(10, 5))
            # plt.title(f"T-SNE of the {title_prefix} by position")
            # sc = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=position_indices, cmap=cmap, norm=norm)
            # cbar = plt.colorbar(sc, ticks=np.arange(len(unique_positions)))
            # cbar.ax.set_yticklabels(unique_positions)  # Assign unique positions as labels
            # plt.savefig(os.path.join(output_folder, f'T-SNE clustering {title_prefix} with threshold {threshold} by position.png'))

            # # T-SNE by patient_id
            # unique_patient_ids = sorted(list(set([s.metadata['patient_id'] for s in spectra])))
            # cmap = mcolors.ListedColormap(plt.cm.cool(np.linspace(0, 1, len(unique_patient_ids))))
            # norm = mcolors.BoundaryNorm(boundaries=np.arange(len(unique_patient_ids) + 1) - 0.5, ncolors=len(unique_patient_ids))
            # patient_id_to_index = {patient_id: i for i, patient_id in enumerate(unique_patient_ids)}
            # patient_id_indices = np.array([patient_id_to_index[s.metadata['patient_id']] for s in spectra])
            # plt.figure(figsize=(10, 5))
            # plt.title(f"T-SNE of the {title_prefix} by patient_id")
            # sc = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=patient_id_indices, cmap=cmap, norm=norm)
            # cbar = plt.colorbar(sc, ticks=np.arange(len(unique_patient_ids)))
            # cbar.ax.set_yticklabels(unique_patient_ids)  # Assign unique patient_ids as labels
            # plt.savefig(os.path.join(output_folder, f'T-SNE clustering {title_prefix} with threshold {threshold} by patient_id.png'))

        def match_with_original_spectra(spectra, clusters, num_clusters, threshold, title_prefix, color_map, output_folder):
            # Prep color map
            # fig1, axe_1 = plt.subplots(num_clusters, 1, sharex=True, figsize=(10, num_clusters * 2))
            # fig2, axe_2 = plt.subplots(num_clusters, 1, sharex=True, figsize=(10, num_clusters * 2))
            # fig1.tight_layout()
            # fig2.tight_layout()
            # axe_1[-1].set_xlabel("Raman shift (cm^-1)")
            # axe_1[-1].set_ylabel("Intensity (a.u.)")
            # axe_2[-1].set_xlabel("Raman shift (cm^-1)")
            # axe_2[-1].set_ylabel("Intensity (a.u.)")

            raman_shift = spectrum.raman_shift_cm

            # check how many cluster is in the spectra metadata
            cluster_set = set([s.metadata['cluster'] for s in spectra])
            print(f"Number of clusters in the spectra metadata: {len(cluster_set)}")
            colors = plt.cm.get_cmap("tab20", len(cluster_set))
            color_for_text = plt.cm.get_cmap("tab20b", num_clusters)


            # for i in range(num_clusters):
            #     cluster_spectra = [spectra[j] for j in np.where(clusters.flatten() == i)[0]]
            #     # Example spectra for each cluster
            #     num = min(500, len(cluster_spectra))
            #     # random indices
            #     random_indices = random.sample(range(len(cluster_spectra)), num)

            #     for j in random_indices:
            #         sns.lineplot(x=raman_shift, y=cluster_spectra[j].intensity, color=colors(cluster_spectra[j].metadata['cluster']), alpha=0.5, ax=axe_1[i])
            #         axe_1[i].text(0.05, 0.95, f"Cluster {i} Total spectra: {len(cluster_spectra)}", transform=axe_1[i].transAxes, fontsize=12, verticalalignment='top', color=color_for_text(i))
            #         sns.lineplot(x=raman_shift, y=cluster_spectra[j].intensity, color=color_for_text(i), alpha=0.5, ax=axe_2[i])
            #         axe_2[i].text(0.05, 0.95, f"Cluster {i} Total spectra: {len(cluster_spectra)}", transform=axe_2[i].transAxes, fontsize=12, verticalalignment='top', color=color_for_text(i))
                    
            # axe_2[0].set_title(f"new clusters of preprocessed data", fontsize=12)
            # axe_1[0].set_title(f"colored by each spectrum's cluster before preprocessing", fontsize=12)
            # fig1.savefig(os.path.join(output_folder, f'Example Spectra {title_prefix} with threshold {threshold} labeled by raw data clusters color.png'))
            # fig2.savefig(os.path.join(output_folder, f'Example Spectra {title_prefix} with threshold {threshold} labeled by new clusters color.png'))

            # T-SNE by cluster
            spectra_intensities = [spectrum.intensity for spectrum in spectra]
            spectra_intensities = np.array(spectra_intensities)
            tsne_results = TSNE(n_components=2, random_state=0).fit_transform(spectra_intensities)
            
            colors = [s.metadata['cluster'] for s in spectra]
            plt.figure(figsize=(10, 5))
            plt.title(f"T-SNE of the {title_prefix}")
            plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=colors, cmap=color_map)
            plt.colorbar()
            plt.savefig(os.path.join(output_folder, f'T-SNE {title_prefix} with threshold {threshold} by clustering.png'))


        def dendrogram_cluster(spectra, threshold, title_prefix, save_cluster_label):
            # Extract intensities from cropped spectra
            spectra_intensities = [spectrum.intensity for spectrum in spectra]

            # Calculate the distance matrix and perform hierarchical clustering
            distance_matrix = pdist(spectra_intensities, metric='euclidean')
            Z = linkage(distance_matrix, method='complete')
            clusters = cut_tree(Z, height=threshold)
            num_clusters = np.max(clusters) + 1

            # Add cluster numbers back into the dictionary
            if save_cluster_label:
                for i, spectrum in enumerate(spectra):
                    spectrum.metadata['cluster'] = clusters[i][0]

                        # plot the dendrogram
            plt.figure(figsize=(10, 5))
            plt.title(f"Dendrogram of the {title_prefix} dataset")
            dendrogram(Z, color_threshold=threshold, above_threshold_color="#808080")
            plt.savefig(os.path.join(output_folder,f'Dendrogram clustering {title_prefix} with threshold {threshold}.png'))
            return spectra, clusters, num_clusters

        # # Plot dendrogram and t-SNE
        original_spectra, original_clusters, original_num_clusters = dendrogram_cluster(cropped_spectra, threshold, "raw data", save_cluster_label=True)
        plot_sample_spectra_and_tsne(original_spectra, original_clusters, original_num_clusters, threshold, "raw data", 'tab20', output_folder)

        # preprocess the spectra
        # for spectrum in original_spectra:
        #     preprocessed_spectra.append(self.preprocessor.preprocess(spectrum))

        # # calculate the distance matrix and clusters for normalized spectra
        # preprocessed_spectra, preprocessed_clusters, preprocessed_num_clusters = dendrogram_cluster(preprocessed_spectra, threshold_normalized, "preprocessed data", save_cluster_label=False)
        # plot_sample_spectra_and_tsne(preprocessed_spectra, preprocessed_clusters, preprocessed_num_clusters, threshold_normalized, "preprocessed data", 'tab20b', output_folder)
        
        # match_with_original_spectra(preprocessed_spectra, preprocessed_clusters, preprocessed_num_clusters, threshold_normalized, "preprocessed data", 'tab20', output_folder)
        # print("Done with hierarchical clustering")


    def remove_clusters_from_db(self, clusters_index_to_remove:list):
        """
        Remove the clusters from the database
        """
        # remove the clusters from the database
        self.db = [spectrum for spectrum in self.db if spectrum.metadata['cluster'] not in clusters_index_to_remove]
        print(f"Removed {len(clusters_index_to_remove)} clusters from the database")
        return

    def check_specific_spectra(self, json_file):
        # load json file 
        with open(json_file) as f:
            spectra_to_check = json.load(f)

        # get the spectra
        for s in spectra_to_check:
            # find the spectrum with the matching patient_id, date, position, and spectrum_id
            for spectrum in self.db:
                if spectrum.metadata['patient_id'] == s['patient_id'] and spectrum.metadata['date'] == s['date'] and spectrum.metadata['position'] == s['position'] and spectrum.metadata['spectrum_id'] == s['spectrum_id']:
                    preprocessed_spectrum = self.preprocessor.preprocess(spectrum)

    
if __name__ == "__main__":
    data_folder = os.path.join(os.getcwd(), "noodlepy", "data", "bec_hnc_train")
    metadata_file = os.path.join(os.getcwd(), "noodlepy", "data", "Biofluid_list_annotated_v4.xlsx")
    # output_folder = os.path.join(os.getcwd(), "quality_control")
    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder)

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
                                        normalization= True,
                                        smoothing=True)
    
    augmentor = SpectrumAugmentor(ramdom_augmentations=True,
                                  augmentation_step_list = None,
                                  config_path= None)

    dataset = HNC_Dataset(data_folder, metadata_file, preprocessor, augmentor)

    # preprocessed_augmented_spectra = []
    # augmented_spectra = []

    # dataset.hierarchical_clustering(threshold=20000, threshold_normalized=10, output_folder=output_folder)


    ######################################################

    # json_file = r"C:\Users\Yifei\Documents\NoodlePy\quality_control\weird_spectra.json"
    # dataset.check_specific_spectra(json_file)

    app = QApplication(sys.argv)
    window = si.SpectraViewer(dataset.db, preprocessor)
    window.resize(1000, 500)
    window.show()
    sys.exit(app.exec_())

    print("Done")