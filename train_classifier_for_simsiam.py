import torch
from torch.utils.data import DataLoader
from noodlepy.utils.spectrumpreprocessor import SpectrumPreprocessor
from noodlepy.ml.classifier import ClassifierWithPretrainedModel
from noodlepy.ml.backbone import cnn_backbone
import torch.nn as nn
import torch.optim as optim
import numpy as np
from noodlepy.utils.bec_hnc_dataset import Bec_HNC_Dataset
import os
import random
import wandb
from noodlepy.utils.seed import seed_all_random_process, seed_worker

if __name__ == "__main__":
    seed = 4
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    wandb.login()
    wandb.init(
        project="SimSiam", 
        name=f"2023_03_03_Bec_HNC_finetune_classifier", 
        config={
            "learning_rate": 0.001,
            "epochs": 150,
            "batch_size": 64,
            "backbone_dim": [1, 8, 16, 32, 64, 128],
            "random_seed": 42,
            "number_of_workers": 8,
            "num_classes": 3,
            "patience": 4  # Early stopping patience
        })
    
    training_cfg = wandb.config
    generator = seed_all_random_process(training_cfg.random_seed)

    train_dataset_path = os.path.join(os.getcwd(), "noodlepy", "data","bec_hnc", "train_classifier")
    test_dataset_path = os.path.join(os.getcwd(), "noodlepy", "data", "bec_hnc", "test")
    metadata_file = os.path.join(os.getcwd(), "noodlepy", "data", "bec_hnc_patient_annotations.xlsx")

    preprocessor = SpectrumPreprocessor(cropping=False,
                                        baseline_correction=True,
                                        remove_cosmic_rays=True,
                                        normalization=True,
                                        smoothing=True)

    train_dataset = Bec_HNC_Dataset(train_dataset_path, metadata_file, preprocessor=preprocessor, augmentor=None)
    test_dataset = Bec_HNC_Dataset(test_dataset_path, metadata_file, preprocessor=preprocessor, augmentor=None)
    
    train_dataset.combat_batch_correction()
    test_dataset.combat_batch_correction() 

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=training_cfg.number_of_workers,
        worker_init_fn=seed_worker,
        generator=generator
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=training_cfg.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=training_cfg.number_of_workers,
        worker_init_fn=seed_worker,
        generator=generator
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained CNN backbone
    backbone = cnn_backbone(training_cfg.backbone_dim)
    backbone.load_state_dict(torch.load(os.path.join(os.getcwd(), "output_plots","2023_03_03_Bec_HNC_batch_corrected","model_HNC.pth")), strict=False)
    classifier = ClassifierWithPretrainedModel(backbone, num_classes=training_cfg.num_classes)
    classifier.to(device)

    # Define loss, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(classifier.parameters(), lr=training_cfg.learning_rate, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    # Early Stopping Setup
    best_val_loss = float("inf")
    patience = training_cfg.patience
    patience_counter = 0

    # Training loop with early stopping
    for epoch in range(training_cfg.epochs):
        classifier.train()
        for i, (x0, x1, labels, raman_shift) in enumerate(train_dataloader):
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
            for i, (x0, x1, labels, raman_shift) in enumerate(test_dataloader):
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

        # Log results in wandb
        wandb.log({"epoch": epoch + 1, "val_loss": val_loss, "val_accuracy": val_accuracy})

        # Step scheduler
        scheduler.step()

        # **Early Stopping Check**
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save the best model
            torch.save(classifier.state_dict(), "best_finetuned_model.pth")
            print(f"Model saved at epoch {epoch + 1} with val_loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. Early stopping patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered. Training stopped.")
            break  # Stop training
