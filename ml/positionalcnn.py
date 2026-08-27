import torch
from torch import nn

class PositionalCNN(nn.Module):
    def __init__(self, layer_channel_sizes, num_classes=5):
        super(PositionalCNN, self).__init__()

        # spectrum_encoding branch
        layers = []
        in_channels = layer_channel_sizes[0] #intialize the input channel size with the first element of the list
        for out_channels in layer_channel_sizes[1:]:
            conv_layer = nn.Conv1d(in_channels, out_channels, kernel_size=4)
            torch.nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='relu') # weights initialization using kaiming uniform
            layers.append(conv_layer)
            layers.append(nn.ReLU())
            layers.append(nn.AvgPool1d(kernel_size=3)) # layers.append(nn.MaxPool1d(kernel_size=2))
            in_channels = out_channels
        self.convs = nn.Sequential(*layers)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Positional Encoding branch (simple feedforward network)
        self.pos_fc1 = nn.Linear(2, 32)  # Input: 2D (x, y) or (r, θ)
        self.pos_fc2 = nn.Linear(32, 64)  # Output: 64-dimensional feature vector
        
        # Fully connected layers after concatenation
        self.fc1 = nn.Linear(out_channels + 64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, num_classes)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, spectrum, pos):
        # Spectrum branch forward pass (1D CNN layers)
        x = self.convs(spectrum)
        x = self.global_avg_pool(x)  # Global average pooling over the channel dimension
        x = x.view(x.size(0), -1)  # Flatten the tensor (batch_size, out_channels)

        # Positional encoding branch
        p = nn.ReLU()(self.pos_fc1(pos))
        p = nn.ReLU()(self.pos_fc2(p))

        # Concatenate the outputs from both branches
        combined = torch.cat((x, p), dim=1)

        # Fully connected layers for classification
        z = nn.ReLU()(self.fc1(combined))
        z = self.dropout(z)
        z = nn.ReLU()(self.fc2(z))
        z = self.fc_out(z)  # No activation here; CrossEntropyLoss expects raw logits

        return z
    
