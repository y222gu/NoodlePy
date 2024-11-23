from sklearn.model_selection import KFold
import torch
from torch.utils.data import DataLoader
from noodlepy.ml.classifier import ClassifierWithPretrainedModel
from noodlepy.ml.backbone import cnn_backbone
import torch.nn as nn
import torch.optim as optim
import numpy as np
from noodlepy.utils.hncdataset import HNC_Dataset
from torch.utils.data import DataLoader
from noodlepy.utils.spectrumpreprocessor import SpectrumPreprocessor
from noodlepy.utils.spectrumaugmentor import SpectrumAugmentor
import os
import matplotlib.pyplot as plt

if __name__ == "__main__":

    dataset_path = os.path.join(os.getcwd(), "noodlepy", "data", "head_and_neck_cancer","plasma_saliva_mixed","all")
    annotation_file_path = os.path.join(os.getcwd(), "noodlepy", "data", "Biofluid_list_annotated_v4.xlsx")

    preprocessor = SpectrumPreprocessor(cropping=True,
                                        baseline_correction=True,
                                        remove_cosmic_rays= True,
                                        normalization=True,
                                        smoothing=True)
    
    augmentor = SpectrumAugmentor(ramdom_augmentations=True,
                                    augmentation_step_list = None,
                                    config_path= None)

    dataset = HNC_Dataset(dataset_path, annotation_file_path, preprocessor=preprocessor, augmentor=augmentor)

    # Set a fixed random seed for reproducibility
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the KFold object
    k_folds = 5
    num_epochs = 10
    results = {}
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    # Start k-fold cross-validation
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f'FOLD {fold + 1}')
        print('--------------------------------')

        # Split dataset into train and validation sets for this fold
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=50, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=50)

        # Initialize a fresh model instance for each fold
        backbone = cnn_backbone(layer_channel_sizes=[1, 8, 16, 32, 64, 128])
        backbone.load_state_dict(torch.load(os.path.join(os.getcwd(), "output_plots","HNC_pretrained_backbone_HNC.pth")))
        num_classes = 3
        classifier = ClassifierWithPretrainedModel(backbone, num_classes=num_classes)
        classifier.to(device)

        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9)


        # Training loop
        for epoch in range(num_epochs):
            classifier.train()
            for i, (x0, x1, labels) in enumerate(train_loader):
                x0, labels = x0.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = classifier(x0)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # Validation loop
            classifier.eval()
            val_loss = 0.0
            correct, total = 0, 0
            with torch.no_grad():
                for i, (x0, x1, labels) in enumerate(val_loader):
                    x0, labels = x0.to(device), labels.to(device)
                    outputs = classifier(x0)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * x0.size(0)
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)
            
            val_loss /= total
            val_accuracy = correct / total
            print(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # Store the validation accuracy for this fold
        results[fold] = val_accuracy

    # Print results for each fold
    print('--------------------------------')
    print("K-Fold Cross-Validation Results:")
    for fold, accuracy in results.items():
        print(f"Fold {fold + 1}: Validation Accuracy = {accuracy:.4f}")

    # Calculate and print the mean accuracy across folds
    mean_accuracy = np.mean(list(results.values()))
    print(f"Mean Cross-Validation Accuracy: {mean_accuracy:.4f}")

    # plot the results
    plt.plot(list(results.keys()), list(results.values()), marker='o')
    plt.xlabel('Fold')
    plt.ylabel('Validation Accuracy')
    plt.title('K-Fold Cross-Validation Results')
    plt.show()