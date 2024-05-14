# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.

import pytorch_lightning as pl
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
import os
from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import SimSiamPredictionHead, SimSiamProjectionHead
from noodlepy.utils.class_SpectrumDataset import SpectrumDataset
import wandb
import math
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from noodlepy.utils.class_SpectrumAugmentor import SpectrumAugmentor
from noodlepy.utils.class_SpectrumPreprocessor import SpectrumPreprocessor
import plotly.express as px
import plotly.graph_objects as go

class cnn_backbone(nn.Module):
    def __init__(self, layer_channel_sizes):
        super(cnn_backbone, self).__init__()
        layers = []
        in_channels = layer_channel_sizes[0] #intialize the input channel size with the first element of the list

        for out_channels in layer_channel_sizes[1:]:
            conv_layer = nn.Conv1d(in_channels, out_channels, kernel_size=4)
            torch.nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='relu') # weights initialization using kaiming uniform
            layers.append(conv_layer)
            layers.append(nn.ReLU())
            layers.append(nn.AvgPool1d(kernel_size=3)) # layers.append(nn.MaxPool1d(kernel_size=2))
            in_channels = out_channels
        #layers.append(nn.Linear(512 * block.expansion, num_classes))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class resnet_backbone(nn.Module):
    def __init__(self):
        super(resnet_backbone, self).__init__()
        resnet = torchvision.models.resnet18()
        self.layers = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        x = self.layers(x)
        return x

class SimSiam(pl.LightningModule):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SimSiamProjectionHead(128, 128, 64)
        self.prediction_head = SimSiamPredictionHead(64, 32, 64)
        self.criterion = NegativeCosineSimilarity()

    def forward(self, x):
        f = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(f)
        p = self.prediction_head(z)
        z = z.detach()
        return z, p

    def training_step(self, batch, batch_idx):
        (x0, x1) = batch
        z0, p0 = self.forward(x0)
        z1, p1 = self.forward(x1)
        loss = 0.5 * (self.criterion(z0, p1) + self.criterion(z1, p0))
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.06)
        return optim

def visualize_embeddings(embeddings2d, labels_for_colors, labels_for_markers, plot_name="test",  map=None):
        colors = map_label(labels_for_colors, 'to_color', map)
        markers = map_label(labels_for_markers, 'to_marker', map)
        # Create DF
        embeddingsdf = pd.DataFrame()
        # Add x coordinate
        embeddingsdf['x'] = embeddings2d[:, 0]
        # Add y coordinate
        embeddingsdf['y'] = embeddings2d[:, 1]
        # Loop through different plot groups
        fig, ax = plt.subplots(figsize=(10, 8))

        # Scatter points, set alpha low to make points translucent
        for i in range(len(embeddingsdf.x)):
            ax.scatter(embeddingsdf.x[i], embeddingsdf.y[i], c=colors[i], marker=markers[i] ,alpha=0.5)
        plt.title('Scatter plot of embeddings using t-SNE')
        save_path = os.path.join(os.getcwd(), "output_plots", "tsne_plot_" + plot_name + ".png")
        plt.savefig(save_path)


def map_label(labels, type, map=None):
    if map is None:
        map = {0:'green', 1:'gold', 2:'orangered', 3:'red', 4:'purple', # staging
                    'Male':'xkcd:blue', 'Female':'xkcd:golden brown', # gender
                    'White':'xkcd:salmon', # race
                    'plasma':"P", 'saliva':">"} #  'plasma':"circle", 'saliva':"cross"
    if type == 'to_color':
        default_label = 'teal'
    elif type == 'to_marker':
        default_label = '*'
    else:
        raise ValueError("Invalid type. Choose 'to_color' or 'to_marker'")
    
    mapped_labels = []
    for label in labels:
        if label in map:
            mapped_labels.append(map[label])
        else:
            mapped_labels.append(default_label)
    return mapped_labels

def visualize_embeddings_in_construction(embeddings2d, labels_for_colors, labels_for_markers, plot_name="test", map=None):
    # Define a default color and marker mapping

    # Map colors and markers based on labels
    colors = map_label(labels_for_colors, 'to_color', map)
    markers = map_label(labels_for_markers, 'to_marker', map)
    # To store unique combinations of color, marker, and labels
    unique_combinations = {}

    # Create a DataFrame with embeddings and labels
    x = embeddings2d[:, 0]
    y = embeddings2d[:, 1]

    # Create the scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot scatter points with specified color and marker
    for i in range(len(x)):
        # Create a unique key for the legend (using both color and marker labels)
        combination_key = (labels_for_colors[i], labels_for_markers[i])

        # Check if this combination is already in the unique_combinations dict
        if combination_key not in unique_combinations:
            # Add to unique combinations with a meaningful label for legend
            unique_combinations[combination_key] = f"{labels_for_colors[i]} & {labels_for_markers[i]}"
            ax.scatter(x, y, 
                   c=colors[i], 
                   marker=markers[i], 
                   alpha=0.6,
                   label=unique_combinations[combination_key])
        else:
            # Plot without additional legend entry
            ax.scatter(x, y, 
                   c=colors[i], 
                   marker=markers[i], 
                   alpha=0.6)

    plt.legend(loc='upper right', fontsize='small')  # Adjust legend size
    ax.set_title('Scatter plot of embeddings using t-SNE')
    ax.set_xlabel('TSNE Component 1')
    ax.set_ylabel('TSNE Component 2')
    plt.show()

    # Save the plot
    save_path = os.path.join(os.getcwd(), "output_plots", f"tsne_plot_{plot_name}.png")
    plt.savefig(save_path)


def visualize_embeddings_3d(embeddings_3d, colors, markers, title='t-SNE 3D Visualization'):

    fig = go.Scatter3d(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2], mode='markers', marker=dict(color=colors, size=5, symbol=markers))

            
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
            zaxis=dict(showgrid=False),
            bgcolor='rgba(0,0,0,0)'
        ))
    fig.write_html(os.path.join(os.getcwd(), "output_plots", "tsne_plot_3d.html"))


def compute_tsne_3d(embeddings, n_components=3, **kwargs):
    tsne = TSNE(n_components=n_components, **kwargs)
    embeddings_3d = tsne.fit_transform(embeddings)
    return embeddings_3d

if __name__ == "__main__":

    wandb.login()
    wandb.init(
        # Set the project where this run will be logged
        project="SimSiam", 
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=f"plasma_saliva_mixed", 
        # Track hyperparameters and run metadata
        config={
            "learning_rate": 0.02,
            "epochs": 1,
            "batch_size": 10,
            "backbone_dim": [1, 8, 16, 32, 64, 128]
            })
    training_cfg = wandb.config
    
    cnn_backbone_1d = cnn_backbone(training_cfg.backbone_dim) # 1D spectral data start with 1 channel, RGB 2D image start with 3 channels
    model = SimSiam(cnn_backbone_1d)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    criterion = NegativeCosineSimilarity()
    optimizer = torch.optim.SGD(model.parameters(), lr=training_cfg.learning_rate)

    train_dataset_path = os.path.join(os.getcwd(), "noodlepy", "data", "Raman_DB", "plasma_saliva_mixed","train")
    test_dataset_path = os.path.join(os.getcwd(), "noodlepy", "data", "Raman_DB", "plasma_saliva_mixed","test" )

    annotation_file_path = os.path.join(os.getcwd(), "noodlepy", "data", "Biofluid_list_annotated_v4.xlsx")

    train_preprocessor = SpectrumPreprocessor(cropping=True,
                                        baseline_correction=True,
                                        remove_cosmic_rays= True,
                                        normalization=True,
                                        smoothing=True)
    train_augmentor = SpectrumAugmentor(ramdom_augmentations=True,
                                  augmentation_step_list = None,
                                  config_path= None)
    train_dataset = SpectrumDataset(train_dataset_path, annotation_file_path, train_preprocessor, train_augmentor)

    train_dataloaders = DataLoader(
        train_dataset,
        batch_size=training_cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8,
    )

    # training
    print("Starting Training")
    for epoch in range(training_cfg.epochs):
        avg_loss = 0.0
        avg_output_std = 0.0
        #total_loss = 0.0
        for i, (x0, x1, labels) in enumerate(train_dataloaders):
            x0 = x0.to(device)
            x1 = x1.to(device)
            z0, p0 = model(x0)
            z1, p1 = model(x1)
            loss = 0.5 * (criterion(z0, p1) + criterion(z1, p0))
            #total_loss += loss.detach()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # calculate the per-dimension standard deviation of the outputs
            # we can use this later to check whether the embeddings are collapsing
            output = p0.detach()
            output = torch.nn.functional.normalize(output, dim=1)
            output_std = torch.std(output, 0)
            output_std = output_std.mean()
            # use moving averages to track the loss and standard deviation
            w = 0.9
            avg_loss = w * avg_loss + (1 - w) * loss.item()
            avg_output_std = w * avg_output_std + (1 - w) * output_std.item()

        # the level of collapse is large if the standard deviation of the l2
        # normalized output is much smaller than 1 / sqrt(dim)
        collapse_level = max(0.0, 1 - math.sqrt(64) * avg_output_std)
        wandb.log({'epoch': epoch+1, 'loss': avg_loss, 'collapse_level': collapse_level})
        print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}, collapse_level: {collapse_level:.5f}")

    print("Finished Training")

    test_preprocessor = SpectrumPreprocessor(cropping=True,
                                        baseline_correction=True,
                                        remove_cosmic_rays= True,
                                        normalization=True,
                                        smoothing=True)
    
    test_augmentor = SpectrumAugmentor(ramdom_augmentations=True)

    test_dataset = SpectrumDataset(test_dataset_path, annotation_file_path, preprocessor=test_preprocessor, augmentor=test_augmentor)

    test_dataloaders = DataLoader(
    test_dataset,
    batch_size=training_cfg.batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=8
    )

    embeddings = []
    labels_staging = []
    labels_sample_type = []
    labels_gender = []
    labels_race = []

    # test the model on the test set
    model.eval()
    with torch.no_grad():
        for i, (x, _, labels) in enumerate(test_dataloaders):
            x = x.to(device)
            y = model.backbone(x).flatten(start_dim=1)
            embeddings.append(y)
            labels_staging.extend(labels['staging'].cpu().numpy())
            labels_sample_type.extend(labels['sample_type'])
            labels_gender.extend(labels['gender'])
            labels_race.extend(labels['race'])    

    embeddings = torch.cat(embeddings, dim=0)
    embeddings = embeddings.cpu().numpy()

    # Save embeddings to file
    with open('embeddings_augment_plasma_saliva_mixed.tsv', 'w') as f:
        for embedding in embeddings:
            embedding_str = '\t'.join(map(str, embedding))
            f.write(embedding_str + '\n')

    combined_labels = [str(labels_staging[i]) + '\t' + labels_sample_type[i] + '\t' + labels_gender[i] + '\t' + labels_race[i] for i in range(len(labels_staging))]
    with open('metadata_augment_plasma_saliva_mixed.tsv', 'w') as f:
        f.write('Index\tstaging\tsample_type\tgender\trace\n')
        for i, label in enumerate(combined_labels):
            f.write('{}\t{}\n'.format(i, label))

    tsne = TSNE(random_state=0, n_iter=1000, metric='cosine')
    embeddings2d = tsne.fit_transform(embeddings)

    visualize_embeddings(embeddings2d, labels_staging, labels_sample_type, 'staging_augment_plasma_saliva_mixed')
    visualize_embeddings(embeddings2d, labels_gender, labels_sample_type,'gender_augment_plasma_saliva_mixed')
    visualize_embeddings(embeddings2d, labels_race, labels_sample_type,'race_augment_plasma_saliva_mixed')
    visualize_embeddings(embeddings2d, labels_staging, labels_sample_type, 'staging_with_three_classes_augment_plasma_saliva_mixed', map=
                         {0: 'green', 1: 'gold', 2: 'gold', 3: 'red', 4: 'red', 'plasma': 'P', 'saliva': '>'})



    test_preprocessor_crop_only = SpectrumPreprocessor(cropping=True)
    test_dataset_crop_only = SpectrumDataset(test_dataset_path, annotation_file_path, preprocessor=test_preprocessor_crop_only, augmentor=None)

    test_dataloaders_crop_only = DataLoader(
    test_dataset_crop_only,
    batch_size=training_cfg.batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=8,
    )

    embeddings = []
    labels_staging = []
    labels_sample_type = []
    labels_gender = []
    labels_race = []

    # test the model on the test set
    model.eval()
    with torch.no_grad():
        for i, (x, _, labels) in enumerate(test_dataloaders_crop_only):
            x = x.to(device)
            y = model.backbone(x).flatten(start_dim=1)
            embeddings.append(y)
            labels_staging.extend(labels['staging'].cpu().numpy())
            labels_sample_type.extend(labels['sample_type'])
            labels_gender.extend(labels['gender'])
            labels_race.extend(labels['race'])    

    # Save embeddings to file
    with open('embeddings_crop_only_plasma_saliva_mixed.tsv', 'w') as f:
        for embedding in embeddings:
            embedding_str = '\t'.join(map(str, embedding))
            f.write(embedding_str + '\n')

    combined_labels = [str(labels_staging[i]) + '\t' + labels_sample_type[i] + '\t' + labels_gender[i] + '\t' + labels_race[i] for i in range(len(labels_staging))]
    with open('metadata_crop_only_plasma_saliva_mixed.tsv', 'w') as f:
        f.write('Index\tstaging\tsample_type\tgender\trace\n')
        for i, label in enumerate(combined_labels):
            f.write('{}\t{}\n'.format(i, label))

    embeddings = torch.cat(embeddings, dim=0)
    embeddings = embeddings.cpu().numpy()
    tsne = TSNE(random_state=0, n_iter=1000, metric='cosine')
    embeddings2d = tsne.fit_transform(embeddings)

    visualize_embeddings(embeddings2d, labels_staging, labels_sample_type, 'staging_crop_only_plasma_saliva_mixed')
    visualize_embeddings(embeddings2d, labels_gender, labels_sample_type,'gender_crop_only_plasma_saliva_mixed')
    visualize_embeddings(embeddings2d, labels_race, labels_sample_type,'race_crop_only_plasma_saliva_mixed')
    visualize_embeddings(embeddings2d, labels_staging, labels_sample_type, 'staging_with_three_classes_crop_only_plasma_saliva_mixed', map=
                         {0: 'green', 1: 'gold', 2: 'gold', 3: 'red', 4: 'red', 'plasma': 'P', 'saliva': '>'})

    print("Finished Visualizing")
