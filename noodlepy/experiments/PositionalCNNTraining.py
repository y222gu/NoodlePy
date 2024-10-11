from torch.utils.data import DataLoader
import torch
from torch import nn
from noodlepy.utils.positionaldataset import PositionalDataset
from noodlepy.ml.positionalcnn import PositionalCNN

# Assuming you have a model, loss function, and optimizer defined
model = PositionalCNN(layer_channel_sizes=[1, 8, 16, 32, 64], num_classes=5)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10

# Instantiate the dataset
dataset = PositionalDataset(spectra, positions, labels)

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    for batch in dataloader:
        spectra_batch = batch['spectrum']
        positions_batch = batch['position']
        labels_batch = batch['label']

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(spectra_batch.unsqueeze(1), positions_batch)  # Add channel dimension to spectra
        loss = criterion(outputs, labels_batch)

        # Backward pass and optimization
        loss.backward()

        # average loss
        optimizer.step()
 
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the model
torch.save(model.state_dict(), 'positional_cnn.pth')

# test the model
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for batch in dataloader:
        spectra_batch = batch['spectrum']
        positions_batch = batch['position']
        labels_batch = batch['label']

        outputs = model(spectra_batch.unsqueeze(1), positions_batch)
        _, predicted = torch.max(outputs, 1)
        total += labels_batch.size(0)
        correct += (predicted == labels_batch).sum().item()

print(f'Accuracy of the network on the test images: {100 * correct / total}%')





