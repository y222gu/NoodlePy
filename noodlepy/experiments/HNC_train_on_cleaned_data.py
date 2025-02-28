import os
import torch
from torch.utils.data import DataLoader
from noodlepy.archive.raw_data_hnc_dataset import HNC_Dataset
from noodlepy.utils.spectrumpreprocessor import SpectrumPreprocessor
from noodlepy.utils.spectrumaugmentor import SpectrumAugmentor
from noodlepy.ml.simsiam import SimSiam
from noodlepy.ml.backbone import cnn_backbone
import wandb
from noodlepy.utils.seed import seed_all_random_process, seed_worker
from noodlepy.utils.traintestmodel import train_model, test_model
import matplotlib.pyplot as plt


if __name__ == "__main__":

    wandb.login()
    wandb.init(
        project="SimSiam", 
        name=f"2025_2_23_HNC_plasma_high_intensity_cleaned", 
        # Track hyperparameters and run metadata
        config={
            "learning_rate": 0.001,
            "epochs": 200,
            "batch_size": 32,
            "backbone_dim": [1, 8, 16, 32, 64, 128],
            "random_seed" : 0,
            "number_of_workers": 8
            })
    training_cfg = wandb.config
    generator = seed_all_random_process(training_cfg.random_seed)

    cnn_backbone_1d = cnn_backbone(training_cfg.backbone_dim) # 1D spectral data start with 1 channel, RGB 2D image start with 3 channels
    model = SimSiam(cnn_backbone_1d)

    plasma_train_dataset_path = r"C:\Users\Yifei\Documents\NoodlePy\noodlepy\data\hnc_raw_data_high_intensity_cleaned"

    annotation_file_path = os.path.join(os.getcwd(), "noodlepy", "data", "Biofluid_list_annotated_v4.xlsx")

    train_preprocessor = SpectrumPreprocessor(cropping=False,
                                        baseline_correction=True,
                                        remove_cosmic_rays= True,
                                        normalization=True,
                                        smoothing=True)
    train_augmentor = SpectrumAugmentor(ramdom_augmentations=True,
                                  augmentation_step_list = None,
                                  config_path= None)


    plasma_train_dataset = HNC_Dataset(plasma_train_dataset_path, annotation_file_path, train_preprocessor, train_augmentor)
 
    plasma_train_dataloaders = DataLoader(
        plasma_train_dataset,
        batch_size=training_cfg.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=training_cfg.number_of_workers,
        worker_init_fn=seed_worker,
        generator=generator
    )

    plasma_test_dataloaders = DataLoader(
        plasma_train_dataset,
        batch_size=training_cfg.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=training_cfg.number_of_workers,
        worker_init_fn=seed_worker,
        generator=generator
    )

    ##### NAME OF THE MODEL
    exp_name = "2025_2_23_HNC_plasma_high_intensity_cleaned"
    experiment_folder = os.path.join(os.getcwd(), "output_plots")
    wandb.run.name = exp_name

    # #### LOAD A SAVED MODEL
    # model.load_state_dict(torch.load(os.path.join(os.getcwd(), "output_plots", "2025_2_23_HNC_plasma_high_intensity_cleaned.pth")))

    ### TRAIN THE MODEL
    model = train_model(model, plasma_train_dataloaders, training_cfg)

    #### SAVE THE MODEL
    experiment_folder = os.path.join(os.getcwd(), "output_plots")
    os.makedirs(experiment_folder, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(experiment_folder, exp_name + "model_HNC.pth"))

    # #### LOAD A SAVED MODEL
    # model.load_state_dict(torch.load(os.path.join(os.getcwd(), "output_plots", "model_HNC.pth")))

    ##### TEST A MODEL
    colors = plt.cm.get_cmap('tab20', 2)
    map_for_color = {0: colors(0), 1: colors(1)}
    test_model(model, plasma_test_dataloaders,  label_name_for_color='cluster', map_for_color = map_for_color, exp_name = exp_name)
    test_model(model, plasma_test_dataloaders,  label_name_for_color='staging', map_for_color = None, exp_name = exp_name)
   
