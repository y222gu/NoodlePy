import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from transformers import CLIPModel, CLIPProcessor
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import pandas as pd
import copy
from noodlepy.utils.spectrum import Spectrum

########## 1) HYPERSPECTRAL → 3-CHANNEL PREPROCESSING via PCA ##########

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
        self.r_filter = r_filter

        list_of_file_paths = []
        for root, dirs, files in os.walk(data_folder):
            if root == data_folder or root.count(os.sep) == data_folder.count(os.sep) + 1:
                list_of_file_paths += [os.path.relpath(os.path.join(root, f), data_folder) for f in files if f.endswith('.txt')]
        list_of_file_paths = sorted(list_of_file_paths)

        annotation_all = pd.read_excel(annotation_file_path)

        self.db =[]
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

        self.images = np.zeros((len(unique_patient_dates), 50, 4, 745), dtype=np.float32)
        self.labels = np.zeros((len(unique_patient_dates),), dtype=np.int64)

        # Form each unique (patient_id, date) pair into a single spectrum object
        for i, (patient_id, date) in enumerate(unique_patient_dates):
            patient_spectra = [spectrum for spectrum in spectrum_objects_list if spectrum.metadata['patient_id'] == patient_id and spectrum.metadata['date'] == date]
            if len(patient_spectra) > 1:
                # form an image of 25x8x745 from the spectra
                image = np.zeros((50, 4, 745), dtype=np.float32)
                for spectrum in patient_spectra:
                    ring = spectrum.metadata['ring'] - 1
                    line = spectrum.metadata['line'] - 1
                    #crop the spectrum to the range of 624.573 to 1784.104
                    cropped_spectrum = copy.deepcopy(spectrum).crop_spectrum(624.573, 1784.104)
                    # normalize the maximum intensity to 1
                    cropped_spectrum.intensity = cropped_spectrum.intensity / np.max(cropped_spectrum.intensity)
                    image[ring, line, :] = cropped_spectrum.intensity
                # add the image to the list of images
                self.images[i] = image
                # add the label to the list of labels
                self.labels[i] = patient_spectra[0].metadata['staging']

        N, H, W, C = self.images.shape

        # Fit PCA on all pixels to reduce 745 bands → 3 components
        flat = self.images.reshape(-1, C)            # (N*H*W, 745)
        self.pca = PCA(n_components=3)
        self.pca.fit(flat)

        # Compute global min/max of PCA outputs for scaling
        flat_pca = self.pca.transform(flat)     # (N*H*W, 3)
        self.pca_min = flat_pca.min(axis=0)     # (3,)
        self.pca_max = flat_pca.max(axis=0)     # (3,)

        # CLIP processor to do the resize → (224,224) and normalization
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1) reduce to 3 channels
        img = self.images[idx]                        # (25, 8, 745)
        flat = img.reshape(-1, img.shape[-1])         # (25*8, 745)
        rgb_flat = self.pca.transform(flat)           # (25*8, 3)


        # plot the PCA transformed image
        rgb_flat = rgb_flat.reshape(50, 4, 3).astype(np.float32)
        plt.imshow(rgb_flat)
        plt.show()


        # 2) scale into [0,1]
        rgb_flat = (rgb_flat - self.pca_min) / (self.pca_max - self.pca_min)
        rgb_flat = np.clip(rgb_flat, 0.0, 1.0)

        rgb = rgb_flat.reshape(50, 4, 3).astype(np.float32)

        # 3) feed through CLIPProcessor: will resize to 224×224 and normalize
        proc = self.processor(images=rgb, return_tensors="pt")
        pixel_values = proc["pixel_values"].squeeze(0)  # (3, 224, 224)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return pixel_values, label
    
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


    def plot_spectrum_as_heatmap(self, patient_id,date):
        """
        Plot the spectrum.intentsity as a heatmap for a given patient_id and date.
        The output should be 25x8x745 heatmap with the x-axis ring (50), y-axis line (4), and z-axis intensity (1024).
        """
        # Filter the dataset for the given patient_id and date
        filtered_spectra = [spectrum for spectrum in self.db if spectrum.metadata['patient_id'] == patient_id and spectrum.metadata['date'] == date]
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



# class HyperspectralDataset(Dataset):
#     def __init__(self, images: np.ndarray, labels: np.ndarray, clip_model_name="openai/clip-vit-base-patch32"):
#         """
#         images: numpy array of shape (N, 25, 8, 745)
#         labels: numpy array of shape (N,) with 0 or 1
#         """
#         self.images = images
#         self.labels = labels
#         N, H, W, C = images.shape

#         # Fit PCA on all pixels to reduce 745 bands → 3 components
#         flat = images.reshape(-1, C)            # (N*H*W, 745)
#         self.pca = PCA(n_components=3)
#         self.pca.fit(flat)

#         # Compute global min/max of PCA outputs for scaling
#         flat_pca = self.pca.transform(flat)     # (N*H*W, 3)
#         self.pca_min = flat_pca.min(axis=0)     # (3,)
#         self.pca_max = flat_pca.max(axis=0)     # (3,)

#         # CLIP processor to do the resize → (224,224) and normalization
#         self.processor = CLIPProcessor.from_pretrained(clip_model_name)

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         # 1) reduce to 3 channels
#         img = self.images[idx]                        # (25, 8, 745)
#         flat = img.reshape(-1, img.shape[-1])         # (25*8, 745)
#         rgb_flat = self.pca.transform(flat)           # (25*8, 3)

#         # 2) scale into [0,1]
#         rgb_flat = (rgb_flat - self.pca_min) / (self.pca_max - self.pca_min)
#         rgb_flat = np.clip(rgb_flat, 0.0, 1.0)

#         rgb = rgb_flat.reshape(25, 8, 3).astype(np.float32)

#         # 3) feed through CLIPProcessor: will resize to 224×224 and normalize
#         proc = self.processor(images=rgb, return_tensors="pt")
#         pixel_values = proc["pixel_values"].squeeze(0)  # (3, 224, 224)

#         label = torch.tensor(self.labels[idx], dtype=torch.long)
#         return pixel_values, label
    

# ########## 2) BUILD MODEL & FREEZE LAYERS ##########

# # Load full CLIP (we only use the vision tower)
# clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# vision = clip.vision_model        # CLIPVisionTransformer
# proj_dim = clip.config.projection_dim  # typically 512

# # Freeze everything except the last 2 transformer blocks
# # and the visual_projection layer (if you want to fine-tune that too)
# for name, param in vision.named_parameters():
#     # enable training only for blocks 10 & 11 and layer norm + projection
#     if ("encoder.layers.10" in name or 
#         "encoder.layers.11" in name or 
#         "post_layernorm" in name or 
#         "visual_projection" in name):
#         param.requires_grad = True
#     else:
#         param.requires_grad = False

# # New classification head on top of CLIP’s pooled output
# class HSClipClassifier(nn.Module):
#     def __init__(self, vision: nn.Module, num_classes: int = 2):
#         super().__init__()
#         self.vision = vision
#         hidden_size = vision.config.hidden_size  # 768
#         self.classifier = nn.Linear(hidden_size, num_classes)

#     def forward(self, pixel_values):
#         vision_outputs = self.vision(pixel_values=pixel_values)
#         pooled = vision_outputs.pooler_output    # (B, 768)
#         logits = self.classifier(pooled)         # (B, num_classes)
#         return logits

# model = HSClipClassifier(vision=clip.vision_model, num_classes=2)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# ########## 3) TRAINING LOOP ##########

# # dummy numpy arrays: replace with your actual loaded data
# # hs_images = np.load("your_images.npy")   # shape (80,25,8,745)
# # labels    = np.load("your_labels.npy")   # shape (80,)
# # For illustration only:
# hs_images = np.random.rand(80,25,8,745).astype(np.float32)
# labels    = np.random.randint(0,2,size=(80,),dtype=np.int64)

# dataset = HyperspectralDataset(hs_images, labels)
# loader  = DataLoader(dataset, batch_size=8, shuffle=True)

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
#                         lr=1e-4, weight_decay=1e-2)

# model.train()
# for epoch in range(100):
#     total_loss = 0.0
#     for pixel_values, batch_labels in loader:
#         pixel_values = pixel_values.to(device)
#         batch_labels = batch_labels.to(device)

#         optimizer.zero_grad()
#         logits = model(pixel_values)
#         loss = criterion(logits, batch_labels)
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#     avg_loss = total_loss / len(loader)
#     print(f"Epoch {epoch+1:02d} — loss: {avg_loss:.4f}")

# print("Fine-tuning complete.")


# test the dataset class
data_folder = os.path.join(os.getcwd(), "noodlepy", "data", "explore", "train")
annotation_file_path = os.path.join(os.getcwd(), "noodlepy", "data", "bec_hnc_patient_annotations.xlsx")
r_filter = None
dataset = HyperspectralDataset(data_folder, annotation_file_path, r_filter)
dataset.__getitem__(0)
dataset.plot_spectrum_as_heatmap(1, "2023-01-01")
