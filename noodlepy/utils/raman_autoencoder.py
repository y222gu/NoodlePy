import torch
import torch.nn as nn


class RamanAutoencoder(nn.Module):
    """
    Autoencoder for 1D Raman spectra anomaly detection.
    
    Architecture:
    - Input: 724 features (wavenumber intensities)
    - Encoder: 724 -> 512 -> 256 -> 128 -> 64 (bottleneck)
    - Decoder: 64 -> 128 -> 256 -> 512 -> 724
    
    Uses LeakyReLU activations and batch normalization for stable training.
    """
    
    def __init__(self, input_dim=724, latent_dim=64):
        super(RamanAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            
            nn.Linear(128, latent_dim),
            nn.LeakyReLU(0.2)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            
            nn.Linear(512, input_dim)
            # No activation on output to allow negative values if needed
        )
    
    def forward(self, x):
        # Flatten input if needed
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # Encode
        latent = self.encoder(x)
        
        # Decode
        reconstruction = self.decoder(latent)
        
        return reconstruction
    
    def encode(self, x):
        """Get latent representation"""
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        return self.encoder(x)
    
    def decode(self, latent):
        """Reconstruct from latent representation"""
        return self.decoder(latent)


class ConvRamanAutoencoder(nn.Module):
    """
    1D Convolutional Autoencoder for Raman spectra.
    
    Uses 1D convolutions to capture local spectral patterns.
    Better for preserving spatial relationships in the spectra.
    """
    
    def __init__(self, input_length=724):
        super(ConvRamanAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            # Input: (batch, 1, 724)
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),  # -> (batch, 32, 362)
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),  # -> (batch, 64, 181)
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),  # -> (batch, 128, 91)
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),  # -> (batch, 256, 46)
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            # Input: (batch, 256, 46)
            nn.ConvTranspose1d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> (batch, 128, 92)
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),  # -> (batch, 64, 183)
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),  # -> (batch, 32, 365)
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose1d(32, 1, kernel_size=7, stride=2, padding=3, output_padding=1),  # -> (batch, 1, 729)
        )
        
        # Final adjustment layer to match exact input size
        self.final_adjust = nn.Linear(729, input_length)
    
    def forward(self, x):
        # Ensure input has channel dimension
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dimension
        
        # Encode
        latent = self.encoder(x)
        
        # Decode
        reconstruction = self.decoder(latent)
        
        # Adjust to exact input size
        reconstruction = reconstruction.squeeze(1)  # Remove channel dimension
        reconstruction = self.final_adjust(reconstruction)
        
        return reconstruction