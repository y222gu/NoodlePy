import os
from PIL import Image
from torch.optim import AdamW
from transformers import (
    CLIPProcessor,
    CLIPVisionModelWithProjection,
    get_linear_schedule_with_warmup
)
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
from noodlepy.utils.spectrum import Spectrum
from sklearn.metrics import classification_report

class HyperspectralDataset(Dataset):
    def __init__(self, data_folder = None,
                 annotation_file_path = None,
                 r_filter = None, 
                 processor = None):
        """
        images: numpy array of shape (N, 50, 8, 745)
        labels: numpy array of shape (N,) with 0 or 1
        """
        print("Loading the Raman dataset")
        if r_filter is None:
            r_filter = np.arange(1, 51)
        self.r_filter = r_filter

        self.processor = processor

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
                    image = np.zeros((len(self.r_filter), 50), dtype=np.float32)
                    min_ring = min(r_filter)
                    for spectrum in line_spectra:
                        ring = spectrum.metadata['ring']
                        # crop the spectrum to 745 wavelengths
                        cropped_spectrum = copy.deepcopy(spectrum).crop_spectrum(615.879, 1784.104) #624.573
                        # bin the spectrum to 50 wavelengths
                        binned_spectrum_intensity = np.mean(cropped_spectrum.intensity.reshape(-1, 15), axis=1)
                        image[ring - min_ring, :] = binned_spectrum_intensity
                        
                    self.images.append(image)
                    self.labels.append(patient_spectra[0].metadata['staging'])
                    self.metadata.append(patient_spectra[0].metadata)

                    print(f"Patient ID {patient_id}, line {line}:")
                    print('image shape:', image.shape)
                else:
                    print(f"Patient ID {patient_id} has {len(line_spectra)} spectra for date {date} and line {line}.")
                    print("Skipping this line.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1) reduce to 3 channels
        image = self.images[idx]                        # (50, 745)
        label = self.labels[idx]                      # (1,)

        # Map your float or int matrix into [0,255] uint8 if needed:
        arr = np.interp(image, (image.min(), image.max()), (0,255)).astype(np.uint8)

        img = Image.fromarray(arr.astype(np.uint8))       # L mode, 1-channel
        img = img.convert("RGB")                          # 3 channels
        # Processor will do resize to 224×224, normalization, torch.tensor
        inputs = self.processor(images=img, return_tensors="pt")
        pixel_values = inputs.pixel_values.squeeze(0)      # (3,224,224)
        
        return pixel_values, torch.tensor(label, dtype=torch.long)
    
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

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image, aspect='auto', cmap='hot', interpolation='nearest')
        ax.set_title(f"Patient: {metadata['patient_id']}, line: {metadata['line']}, Staging: {label}")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Intensity")
        plt.colorbar(ax.imshow(image, aspect='auto', cmap='hot', interpolation='nearest'))
        plt.show()
        return


# 2) Build the model: CLIP vision + projection + new head
class CLIPPartialFineTuner(nn.Module):
    def __init__(self,
                 pretrained_model_name="openai/clip-vit-base-patch32",
                 num_unfreeze_blocks=2,
                 num_classes=2,
                 dropout_p=0.5):
        super().__init__()
        self.clip_vision = CLIPVisionModelWithProjection.from_pretrained(
            pretrained_model_name
        )
        proj_dim = self.clip_vision.config.projection_dim

        # New head with dropout
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(proj_dim, num_classes)
        )

        # Freeze all CLIP layers
        for p in self.clip_vision.parameters():
            p.requires_grad = False

        # Unfreeze last N transformer blocks
        encoder_layers = self.clip_vision.vision_model.encoder.layers
        for layer in encoder_layers[-num_unfreeze_blocks:]:
            for p in layer.parameters():
                p.requires_grad = True

        # Unfreeze final layernorm & projection
        for p in self.clip_vision.vision_model.post_layernorm.parameters():
            p.requires_grad = True
        for p in self.clip_vision.visual_projection.parameters():
            p.requires_grad = True

    def forward(self, pixel_values):
        outputs = self.clip_vision(pixel_values=pixel_values)
        image_embeds = outputs.image_embeds
        logits = self.classifier(image_embeds)
        return logits
    
class CLIPHeadOnlyFineTuner(nn.Module):
    def __init__(self,
                 pretrained_model_name="openai/clip-vit-base-patch32",
                 num_classes=2,
                 dropout_p=0.5):
        super().__init__()
        # load full CLIP vision+projection
        self.clip_vision = CLIPVisionModelWithProjection.from_pretrained(
            pretrained_model_name
        )
        proj_dim = self.clip_vision.config.projection_dim

        # head: dropout + linear
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(proj_dim, num_classes)
        )

        # freeze entire CLIP vision trunk + projection
        for p in self.clip_vision.parameters():
            p.requires_grad = False

    def forward(self, pixel_values):
        outputs = self.clip_vision(pixel_values=pixel_values)
        image_embeds = outputs.image_embeds   # (batch_size, proj_dim)
        return self.classifier(image_embeds)

# --- 1) Prepare data loaders ---
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

train_dir = os.path.join(os.getcwd(), "noodlepy", "data", "cosmic_ray_removed", "train")
val_dir = os.path.join(os.getcwd(), "noodlepy", "data", "cosmic_ray_removed", "val")
test_dir = os.path.join(os.getcwd(), "noodlepy", "data", "cosmic_ray_removed", "test")
annotation_file_path = os.path.join(os.getcwd(), "noodlepy", "data", "bec_hnc_patient_annotations.xlsx")

train_dataset = HyperspectralDataset(train_dir, annotation_file_path, r_filter=range(3, 49), processor=processor)
val_dataset = HyperspectralDataset(val_dir, annotation_file_path, r_filter=range(3, 49), processor=processor)
test_dataset = HyperspectralDataset(test_dir, annotation_file_path, r_filter=range(3, 49), processor=processor)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=8, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=8, shuffle=False)

# --- 2) Model, criterion, optimizer ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPPartialFineTuner(
    pretrained_model_name="openai/clip-vit-base-patch32",
    num_classes=2,
    num_unfreeze_blocks=1,
    dropout_p=0.7
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-5,
    weight_decay=1e-5
)

num_epochs = 80
total_steps = num_epochs * len(train_loader)
warmup_steps = int(0.1 * total_steps)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# --- 4) Early stopping & training loop ---
patience = 20
best_val_loss = float('inf')
epochs_no_improve = 0
best_model_wts = copy.deepcopy(model.state_dict())

for epoch in range(1, num_epochs + 1):
    # ---- Train ----
    model.train()
    running_train_loss = 0.0
    for px, labels in train_loader:
        px, labels = px.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(px)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_train_loss += loss.item() * px.size(0)

    train_loss = running_train_loss / len(train_loader.dataset)

    # ---- Validate ----
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for px, labels in val_loader:
            px, labels = px.to(device), labels.to(device)
            logits = model(px)
            loss = criterion(logits, labels)
            running_val_loss += loss.item() * px.size(0)

    val_loss = running_val_loss / len(val_loader.dataset)
    print(f"Epoch {epoch}/{num_epochs} — train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")

    # ---- Checkpointing & early stop ----
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(best_model_wts, "best_clip_model.pth")
        epochs_no_improve = 0
        print(f" New best model (val_loss={val_loss:.4f}) saved.")
    else:
        epochs_no_improve += 1
        print(f" No improvement for {epochs_no_improve} epoch(s).")

    if epochs_no_improve >= patience:
        print(f" Early stopping after {epoch} epochs without improvement.")
        break

# --- 5) Load best model & test evaluation ---
model.load_state_dict(best_model_wts)
model.eval()

test_loss   = 0.0
correct     = 0
total       = 0
all_preds   = []
all_labels  = []

with torch.no_grad():
    for px, labels in test_loader:
        px, labels = px.to(device), labels.to(device)
        logits = model(px)
        loss = criterion(logits, labels)
        test_loss += loss.item() * px.size(0)

        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

test_loss /= len(test_loader.dataset)
test_acc = correct / total

print(f"\n Test Loss: {test_loss:.4f}")
print(f" Test Accuracy: {test_acc:.4f} ({correct}/{total})")
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, digits=4))