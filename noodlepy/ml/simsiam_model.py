# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.

import pytorch_lightning as pl
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import SimSiamPredictionHead, SimSiamProjectionHead

from noodlepy.ml.raman_dataset import RamanDataset

class cnn_backbone(nn.Module):
    def __init__(self, layer_channel_sizes):
        super(cnn_backbone, self).__init__()
        layers = []
        in_channels = layer_channel_sizes[0] #intialize the input channel size with the first element of the list

        for out_channels in layer_channel_sizes[1:]:
            conv_layer = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
            torch.nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='relu') # weights initialization using kaiming uniform
            layers.append(conv_layer)
            layers.append(nn.ReLU())
            layers.append(nn.AvgPool1d(kernel_size=2)) # layers.append(nn.MaxPool1d(kernel_size=2))
            in_channels = out_channels

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x

class resnet_backbone():
    resnet = torchvision.models.resnet18()
    backbone = nn.Sequential(*list(resnet.children())[:-1])


class SimSiam(pl.LightningModule):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SimSiamProjectionHead(128, 128, 64)
        self.prediction_head = SimSiamPredictionHead(64, 32, 64)
        self.criterion = NegativeCosineSimilarity()

    def forward(self, x):
        f = self.backbone(x).flatten(start_dim=1)
        print(f.shape)
        z = self.projection_head(f)
        p = self.prediction_head(z)
        z = z.detach()
        return z, p

    def training_step(self, batch, batch_idx):
        (x0, x1) = batch[0]
        z0, p0 = self.forward(x0)
        z1, p1 = self.forward(x1)
        loss = 0.5 * (self.criterion(z0, p1) + self.criterion(z1, p0))
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.06)
        return optim

if __name__ == "__main__":
    cnn_backbone_1d = cnn_backbone([1, 8, 16, 32, 64, 128]) # 1D spectral data start with 1 channel, RGB 2D image start with 3 channels
    model = SimSiam(cnn_backbone_1d)

    dataset = RamanDataset()
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        drop_last=True,
        num_workers=8,
    )
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    trainer = pl.Trainer(max_epochs=1, devices=1, accelerator=accelerator)
    trainer.fit(model=model, train_dataloaders=dataloader)