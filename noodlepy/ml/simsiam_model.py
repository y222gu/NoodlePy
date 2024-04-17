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
from noodlepy.ml.raman_dataset import RamanDataset
import wandb
import math
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt

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
        x = self.layers(x)
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


if __name__ == "__main__":

    wandb.login()
    wandb.init(
        # Set the project where this run will be logged
        project="test", 
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=f"experiment_{1}", 
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

    train_dataset_path = os.path.join(os.getcwd(), "noodlepy", "data", "Raman_DB", "train")
    test_dataset_path = os.path.join(os.getcwd(), "noodlepy", "data", "Raman_DB", "test")

    train_dataset = RamanDataset(train_dataset_path)
    test_dataset = RamanDataset(test_dataset_path)

    train_dataloaders = DataLoader(
        train_dataset,
        batch_size=training_cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8,
    )
    test_dataloaders = DataLoader(
        test_dataset,
        batch_size=training_cfg.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=8,
    )

    print("Starting Training")
    for epoch in range(10):
        avg_loss = 0.0
        avg_output_std = 0.0
        #total_loss = 0.0
        for i, batch in enumerate(train_dataloaders, 0):
            x0, x1 = batch
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

    # Extract embedding of the test set
    embeddings = []
    filenames = []
    # disable gradients for faster calculations
    model.eval()
    with torch.no_grad():
        for i,(x,_) in enumerate(test_dataloaders):
            # embed the images with the pre-trained backbone
            x = x.to(device)
            y = model.backbone(x).flatten(start_dim=1)
            # store the embeddings in a list
            embeddings.append(y)

    # concatenate the embeddings and convert to numpy
    embeddings = torch.cat(embeddings, dim=0)
    embeddings = embeddings.cpu().numpy()

    # visualize the embeddings with t-SNE
    tsne = TSNE(random_state = 0, n_iter = 1000, metric = 'cosine')
    # Fit and transform
    embeddings2d = tsne.fit_transform(embeddings)
    # Create DF
    embeddingsdf = pd.DataFrame()
    # Add x coordinate
    embeddingsdf['x'] = embeddings2d[:,0]
    # Add y coordinate
    embeddingsdf['y'] = embeddings2d[:,1]
    # Check
    embeddingsdf.head()
    # Set figsize
    fig, ax = plt.subplots(figsize=(10,8))
    # Scatter points, set alpha low to make points translucent
    ax.scatter(embeddingsdf.x, embeddingsdf.y, alpha=.1)
    plt.title('Scatter plot of games using t-SNE')
    plt.show()
    save_path = os.path.join(os.getcwd(), "output_plots", "tsne_plot.png")
    plt.savefig(save_path)
