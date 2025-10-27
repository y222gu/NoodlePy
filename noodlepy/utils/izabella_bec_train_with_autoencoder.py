import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from datetime import datetime
from noodlepy.utils.raman_autoencoder import RamanAutoencoder, ConvRamanAutoencoder
from noodlepy.utils.izabelladataset import OC_Dataset  # Adjust import based on your file structure
from noodlepy.utils.spectrumpreprocessor import SpectrumPreprocessor

class AutoencoderTrainer:
    """
    Trainer class for Raman spectra autoencoder with anomaly detection capabilities.
    """
    
    def __init__(self, model, device='cuda', learning_rate=1e-3, model_type='linear'):
        self.model = model.to(device)
        self.device = device
        self.model_type = model_type
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        self.criterion = nn.MSELoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        
        for batch_idx, (intensity, raman_shift, metadata) in enumerate(train_loader):
            # Move data to device
            intensity = intensity.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            reconstruction = self.model(intensity)
            
            # Compute loss
            if self.model_type == 'conv':
                # For conv model, ensure shapes match
                loss = self.criterion(reconstruction, intensity.squeeze(1))
            else:
                # For linear model
                loss = self.criterion(reconstruction, intensity.view(intensity.size(0), -1))
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
        
        return epoch_loss / len(train_loader)
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for intensity, raman_shift, metadata in val_loader:
                intensity = intensity.to(self.device)
                
                reconstruction = self.model(intensity)
                
                if self.model_type == 'conv':
                    loss = self.criterion(reconstruction, intensity.squeeze(1))
                else:
                    loss = self.criterion(reconstruction, intensity.view(intensity.size(0), -1))
                
                val_loss += loss.item()
        
        return val_loss / len(val_loader)
    
    def train(self, train_loader, val_loader, num_epochs, save_dir='checkpoints'):
        """Complete training loop"""
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Starting training for {num_epochs} epochs")
        print(f"Training on {len(train_loader.dataset)} samples")
        print(f"Validating on {len(val_loader.dataset)} samples")
        print(f"Device: {self.device}\n")
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Print progress
            print(f"Epoch [{epoch+1}/{num_epochs}] - "
                  f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(os.path.join(save_dir, 'best_model.pth'), epoch, val_loss)
                print(f"  → New best model saved! (Val Loss: {val_loss:.6f})")
            
            # # Save periodic checkpoint
            # if (epoch + 1) % 10 == 0:
            #     self.save_checkpoint(
            #         os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'),
            #         epoch, val_loss
            #     )
        
        print("\nTraining completed!")
        return self.train_losses, self.val_losses
    
    def save_checkpoint(self, path, epoch, val_loss):
        """Save model checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, path)
    
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']} with val_loss {checkpoint['val_loss']:.6f}")

class AnomalyDetector:
    """
    Anomaly detection using trained autoencoder.
    Calculates reconstruction error as anomaly score.
    """
    
    def __init__(self, model, device='cuda', model_type='linear'):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.model_type = model_type
        self.reconstruction_errors = None
        self.threshold = None
    
    def calculate_reconstruction_errors(self, dataloader):
        """Calculate reconstruction errors for all samples"""
        errors = []
        
        with torch.no_grad():
            for intensity, raman_shift, metadata in dataloader:
                intensity = intensity.to(self.device)
                
                reconstruction = self.model(intensity)
                
                # Calculate per-sample MSE
                if self.model_type == 'conv':
                    error = torch.mean((reconstruction - intensity.squeeze(1)) ** 2, dim=1)
                else:
                    intensity_flat = intensity.view(intensity.size(0), -1)
                    error = torch.mean((reconstruction - intensity_flat) ** 2, dim=1)
                
                errors.extend(error.cpu().numpy())
        
        self.reconstruction_errors = np.array(errors)
        return self.reconstruction_errors
    
    def set_threshold(self, percentile=95):
        """Set anomaly threshold based on percentile of reconstruction errors"""
        if self.reconstruction_errors is None:
            raise ValueError("Calculate reconstruction errors first")
        
        self.threshold = np.percentile(self.reconstruction_errors, percentile)
        print(f"Anomaly threshold set at {percentile}th percentile: {self.threshold:.6f}")
        return self.threshold
    
    def detect_anomalies(self, dataloader):
        """Detect anomalies in dataset"""
        if self.threshold is None:
            raise ValueError("Set threshold first using set_threshold()")
        
        errors = self.calculate_reconstruction_errors(dataloader)
        anomalies = errors > self.threshold
        
        print(f"Found {np.sum(anomalies)} anomalies out of {len(errors)} samples")
        print(f"Anomaly rate: {100 * np.sum(anomalies) / len(errors):.2f}%")
        
        return anomalies, errors
    
    def visualize_reconstruction(self, spectrum_intensity, save_path=None):
        """Visualize original vs reconstructed spectrum"""
        self.model.eval()
        
        with torch.no_grad():
            spectrum_tensor = torch.tensor(spectrum_intensity, dtype=torch.float32).unsqueeze(0).to(self.device)
            if self.model_type == 'linear':
                spectrum_tensor = spectrum_tensor.unsqueeze(0)
            
            reconstruction = self.model(spectrum_tensor)
            reconstruction = reconstruction.cpu().numpy().flatten()
        
        original = spectrum_intensity.flatten()
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(original, label='Original', linewidth=1.5)
        plt.plot(reconstruction, label='Reconstructed', linewidth=1.5, alpha=0.7)
        plt.xlabel('Wavenumber Index')
        plt.ylabel('Intensity')
        plt.title('Spectrum Reconstruction')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(original - reconstruction, linewidth=1.5, color='red')
        plt.xlabel('Wavenumber Index')
        plt.ylabel('Reconstruction Error')
        plt.title('Reconstruction Error')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def plot_training_history(train_losses, val_losses, save_path=None):
    """Plot training and validation losses"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_anomaly_scores(reconstruction_errors, threshold, save_path=None):
    """Plot reconstruction error distribution with threshold"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(reconstruction_errors, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.6f}')
    plt.xlabel('Reconstruction Error (MSE)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Reconstruction Errors')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    sorted_errors = np.sort(reconstruction_errors)
    plt.plot(sorted_errors, linewidth=2)
    plt.axhline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.6f}')
    plt.xlabel('Sample Index (sorted)')
    plt.ylabel('Reconstruction Error (MSE)')
    plt.title('Sorted Reconstruction Errors')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """
    Main training script
    """
    # Configuration
    TRAIN_DATA_FOLDER = r'D:\for_autoencoder\train'  # UPDATE THIS
    TEST_DATA_FOLDER = r'D:\for_autoencoder\test'  # UPDATE THIS
    OUTPUT_FOLDER = r'D:\for_autoencoder\results'  # UPDATE THIS
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-3
    VAL_SPLIT = 0.2
    INPUT_DIM = 724
    LATENT_DIM = 64
    MODEL_TYPE = 'linear'  # 'linear' or 'conv'
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    preprocessor = SpectrumPreprocessor(cropping=True,
                                        baseline_correction=True,
                                        remove_cosmic_rays=False,
                                        normalization=True,
                                        smoothing=True)

    # You'll need to define your preprocessor and augmentor
    # For now, using None as placeholders
    dataset = OC_Dataset(
        data_folder=TRAIN_DATA_FOLDER,
        preprocessor=preprocessor,
        augmentor=None      # Add your augmentor here
    )
    
    # Split dataset
    val_size = int(VAL_SPLIT * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Dataset split: {train_size} training, {val_size} validation\n")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Initialize model
    if MODEL_TYPE == 'linear':
        model = RamanAutoencoder(input_dim=INPUT_DIM, latent_dim=LATENT_DIM)
    else:
        model = ConvRamanAutoencoder(input_length=INPUT_DIM)
    
    print(f"Model architecture: {MODEL_TYPE}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Initialize trainer
    trainer = AutoencoderTrainer(model, device=device, learning_rate=LEARNING_RATE, model_type=MODEL_TYPE)
    
    # Train model
    train_losses, val_losses = trainer.train(train_loader, val_loader, NUM_EPOCHS, OUTPUT_FOLDER)
    
    # Plot training history
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(OUTPUT_FOLDER, f'training_history_{timestamp}.png')
    plot_training_history(train_losses, val_losses, file_path)
    
    # Anomaly detection on validation set
    print("\n" + "="*50)
    print("ANOMALY DETECTION")
    print("="*50 + "\n")
    
    detector = AnomalyDetector(model, device=device, model_type=MODEL_TYPE)
    
    # Calculate reconstruction errors on validation set
    print("Calculating reconstruction errors...")
    errors = detector.calculate_reconstruction_errors(val_loader)
    
    # Set threshold (95th percentile)
    threshold = detector.set_threshold(percentile=95)
    
    # Detect anomalies
    anomalies, errors = detector.detect_anomalies(val_loader)
    
    # Plot anomaly scores
    file_path = os.path.join(OUTPUT_FOLDER, f'anomaly_scores_{timestamp}.png')
    plot_anomaly_scores(errors, threshold, file_path)

    # Visualize a few reconstructions
    print("\nVisualizing sample reconstructions...")
    for i in range(min(3, len(val_dataset))):
        file_path = os.path.join(OUTPUT_FOLDER, f'reconstruction_sample_{i}_{timestamp}.png')
        intensity, _, metadata = val_dataset[i]
        detector.visualize_reconstruction(
            intensity.numpy(),
            save_path=file_path
        )
    
    print("\nTraining and evaluation complete!")
    print(f"Best validation loss: {trainer.best_val_loss:.6f}")
    print(f"Anomaly threshold: {threshold:.6f}")


if __name__ == '__main__':
    main()