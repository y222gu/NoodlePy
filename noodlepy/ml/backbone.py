import torch
import torch.nn as nn
import torchvision

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