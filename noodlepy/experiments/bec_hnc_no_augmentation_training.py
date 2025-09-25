import os
import torch
from torch.utils.data import DataLoader
from noodlepy.utils.bec_hnc_dataset_neighbor_ring_pairs import Bec_HNC_Dataset
from noodlepy.ml.simsiam import SimSiam
from noodlepy.ml.backbone import cnn_backbone
import wandb
from noodlepy.utils.seed import seed_all_random_process, seed_worker
from noodlepy.utils.traintestmodel import train_model, test_model


if __name__ == "__main__":

    wandb.login()
    wandb.init(
        project="SimSiam", 
        name=f"Bec_HNC_SimSiam_neighbor_ring_pairs", 
        # Track hyperparameters and run metadata
        config={
            "learning_rate": 0.001,
            "epochs": 200,
            "batch_size": 256,
            "backbone_dim": [1, 8, 16, 32, 64, 128],
            "random_seed" : 0,
            "number_of_workers": 8
            })
    training_cfg = wandb.config
    generator = seed_all_random_process(training_cfg.random_seed)

    cnn_backbone_1d = cnn_backbone(training_cfg.backbone_dim) # 1D spectral data start with 1 channel, RGB 2D image start with 3 channels
    model = SimSiam(cnn_backbone_1d)

    train_dataset_path = os.path.join(os.getcwd(), "noodlepy", "data", "explore","train")
    val_dataset_path = os.path.join(os.getcwd(), "noodlepy", "data", "explore","val")
    test_dataset_path = os.path.join(os.getcwd(), "noodlepy", "data", "explore","test" )

    annotation_file_path = os.path.join(os.getcwd(), "noodlepy", "data", "bec_hnc_patient_annotations.xlsx")
    
    train_dataset = Bec_HNC_Dataset(train_dataset_path, annotation_file_path)

    train_dataloaders = DataLoader(
        train_dataset,
        batch_size=training_cfg.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=training_cfg.number_of_workers,
        worker_init_fn=seed_worker,
        generator=generator
    )

    val_dataset = Bec_HNC_Dataset(val_dataset_path, annotation_file_path)
    val_dataloaders = DataLoader(
        val_dataset,
        batch_size=training_cfg.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=training_cfg.number_of_workers,
        worker_init_fn=seed_worker,
        generator=generator
    )

    test_dataset = Bec_HNC_Dataset(test_dataset_path, annotation_file_path)
    test_dataloaders = DataLoader(
    test_dataset,
    batch_size=training_cfg.batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=training_cfg.number_of_workers,
    worker_init_fn=seed_worker,
    generator=generator
    )

    ##### NAME OF THE MODEL
    exp_name = "Bec_HNC_SimSiam_neighbor_ring_pairs"
    model_path = os.path.join(os.getcwd(), "output_plots", exp_name)


    # ##### LOAD A SAVED MODEL
    model.load_state_dict(torch.load(os.path.join(os.getcwd(), "output_plots", exp_name, "best_1.pth")))

    #### TRAIN THE MODEL
    model = train_model(model, train_dataloaders, val_dataloaders, training_cfg, patience=10, save_path=model_path)

    #### SAVE THE MODEL
    # torch.save(model.state_dict(), os.path.join(os.getcwd(), "output_plots", exp_name + "_model_HNC.pth"))

    # ##### LOAD A SAVED MODEL
    # model.load_state_dict(torch.load(os.path.join(os.getcwd(), "output_plots", exp_name, "best.pth")))

    ##### TEST A MODEL
    test_model(model, train_dataloaders,  label_name_for_color='staging',map_for_color = None, exp_name = exp_name)
   
