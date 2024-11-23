import os
import torch
from torch.utils.data import DataLoader
from noodlepy.utils.bec_hnc_dataset import Bec_HNC_Dataset
from noodlepy.utils.spectrumpreprocessor import SpectrumPreprocessor
from noodlepy.ml.classifier import Classifier
from noodlepy.ml.backbone import cnn_backbone
import wandb
from noodlepy.utils.seed import seed_all_random_process, seed_worker
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def train_model(model, train_dataloaders, num_epochs, criterion, optimizer):
    early_stopping = 0
    all_losses = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for x0, x1, labels in train_dataloaders:
            optimizer.zero_grad()
            outputs = model(x0)
            label = labels['staging']
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x0.size(0)

        # Calculate average loss over epoch
        epoch_loss = running_loss / len(train_dataloaders.dataset)
        wandb.log({"epoch": epoch+1, "loss": epoch_loss})
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        # early stopping if the loss is not decreasing for more than 3 epochs
        if epoch == 0:
            all_losses.append(epoch_loss)
        else:
            if epoch_loss < all_losses[-1]:
                all_losses.append(epoch_loss)
                early_stopping = 0
            else:
                early_stopping += 1
                all_losses.append(epoch_loss)
                if early_stopping > 3:
                    print("Early stopping")
                    break
    return model

def evaluate_model(model, test_dataloaders):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x0, x1, labels in test_dataloaders:
            outputs = model(x0)
            _, predicted = torch.max(outputs, 1)
            label = labels['staging']
            total += label.size(0)
            correct += (predicted == label).sum().item()

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    return accuracy

def make_r_filters(num_groups):
    num_rings = int(25/num_groups)
    r_filter = []
    for group in range(num_groups):
        first_half = np.arange(group*num_rings+1, (group+1)*num_rings+1)
        second_half = np.arange(51-(group+1)*num_rings, 51-group*num_rings)
        r_filter_for_this_group = np.concatenate((
            first_half,
            second_half
        ))
        print(r_filter_for_this_group)
        r_filter.append(r_filter_for_this_group)

    print(r_filter)
    return r_filter

def prepare_data(train_dataset_path,test_dataset_path, annotation_file_path, r_filter, generator):
    preprocessor = SpectrumPreprocessor(cropping=True,
                                        baseline_correction=True,
                                        remove_cosmic_rays= True,
                                        normalization=True,
                                        smoothing=True)
    train_dataset = Bec_HNC_Dataset(train_dataset_path, annotation_file_path, r_filter = r_filter, preprocessor=preprocessor, augmentor=None)
    test_dataset = Bec_HNC_Dataset(test_dataset_path, annotation_file_path, r_filter = r_filter, preprocessor=preprocessor, augmentor=None)
    train_dataloaders = DataLoader(
        train_dataset,
        batch_size=50,
        shuffle=True,
        drop_last=False,
        num_workers=8,
        worker_init_fn=seed_worker,
        generator=generator
    )
    test_dataloaders = DataLoader(
    test_dataset,
    batch_size=50,
    shuffle=False,
    drop_last=False,
    num_workers=8,
    worker_init_fn=seed_worker,
    generator=generator
    )
    return train_dataloaders, test_dataloaders


if __name__ == "__main__":
    # track experiment
    wandb.login()
    wandb.init(project="bec_hnc_cnn_trained_by_group_of_rings")

    track_accuracy = []
    # prepare data
    number_of_groups = 5
    r_filters = make_r_filters(number_of_groups)
    generator = seed_all_random_process(0)
    train_dataset_path = os.path.join(os.getcwd(), "noodlepy", "data", "bec_hnc","train")
    test_dataset_path = os.path.join(os.getcwd(), "noodlepy", "data", "bec_hnc","test" )
    annotation_file_path = os.path.join(os.getcwd(), "noodlepy", "data", "bec_hnc_patient_annotations.xlsx")

    for group_i in range(number_of_groups):
        model = None
        train_dataloaders, test_dataloaders = prepare_data(train_dataset_path, test_dataset_path, annotation_file_path, r_filters[group_i], generator)
        num_classes = 3
        # start a new model and make sure the previous model is not used
        cnn_backbone_1d = cnn_backbone([1, 8, 16, 32, 64, 128]) # 1D spectral data start with 1 channel, RGB 2D image start with 3 channels
        model = Classifier(cnn_backbone_1d, num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train model
        num_epochs = 100
        model = train_model(model, train_dataloaders, num_epochs, criterion, optimizer)

        # save model
        torch.save(model.state_dict(), f"model_trained_on_ring_group_{group_i+1}.pt")

        # Evaluate model
        accuracy = evaluate_model(model, test_dataloaders)
        print(f"Accuracy: {accuracy * 100:.2f}%")
        track_accuracy.append(accuracy)

    plt.plot([1,2,3,4,5], track_accuracy)
    plt.xlabel("Group (5 rings per group)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy of model trained on each group")
    plt.show()

    print("Done!")