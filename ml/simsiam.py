# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.
import os
import pytorch_lightning as pl
import torch
import torchvision
from torch import nn
from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import SimSiamPredictionHead, SimSiamProjectionHead
import torch.nn.functional as F

class cnn_backbone(nn.Module):
    def __init__(self, layer_channel_sizes):
        super(cnn_backbone, self).__init__()
        layers = []
        in_channels = layer_channel_sizes[0]  # input channel size

        for out_channels in layer_channel_sizes[1:]:
            # 1) Conv layer
            conv_layer = nn.Conv1d(in_channels, out_channels, kernel_size=4)

            layers.append(conv_layer)

            # 2) Batch Normalization
            layers.append(nn.BatchNorm1d(out_channels))

            # 3) Activation
            layers.append(nn.ReLU())

            # 4) Pooling
            layers.append(nn.AvgPool1d(kernel_size=3))

            in_channels = out_channels

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

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