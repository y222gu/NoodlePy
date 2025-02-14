import os
import torch
from torch.utils.data import DataLoader
from noodlepy.utils.hncdataset import HNC_Dataset
from noodlepy.utils.spectrumpreprocessor import SpectrumPreprocessor
from noodlepy.utils.spectrumaugmentor import SpectrumAugmentor
from noodlepy.ml.simsiam import SimSiam
from noodlepy.ml.backbone import cnn_backbone
import wandb
from noodlepy.utils.seed import seed_all_random_process, seed_worker
from noodlepy.utils.traintestmodel import train_model, test_model


if __name__ == "__main__":

    wandb.login()
    wandb.init(
        project="SimSiam", 
        name=f"2025_2_13_HNC_train_on_plasma_after_cleaning_hierarchical_clustering", 
        # Track hyperparameters and run metadata
        config={
            "learning_rate": 0.001,
            "epochs": 50,
            "batch_size": 32,
            "backbone_dim": [1, 8, 16, 32, 64, 128],
            "random_seed" : 0,
            "number_of_workers": 8
            })
    training_cfg = wandb.config
    generator = seed_all_random_process(training_cfg.random_seed)

    cnn_backbone_1d = cnn_backbone(training_cfg.backbone_dim) # 1D spectral data start with 1 channel, RGB 2D image start with 3 channels
    model = SimSiam(cnn_backbone_1d)

    plasma_train_dataset_path = os.path.join(os.getcwd(), "noodlepy", "data", "head_and_neck_cancer","plasma_cleaned")

    annotation_file_path = os.path.join(os.getcwd(), "noodlepy", "data", "Biofluid_list_annotated_v4.xlsx")

    train_preprocessor = SpectrumPreprocessor(cropping=True,
                                        baseline_correction=False,
                                        remove_cosmic_rays= False,
                                        normalization=True,
                                        smoothing=False)
    train_augmentor = SpectrumAugmentor(ramdom_augmentations=True,
                                  augmentation_step_list = None,
                                  config_path= None)


    plasma_train_dataset = HNC_Dataset(plasma_train_dataset_path, annotation_file_path, train_preprocessor, train_augmentor)
    plasma_train_dataset.hierarchical_clustering(threshold=8)
    plasma_train_dataset.remove_clusters_from_db([1, 2, 3, 4,5,6])
 
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
    exp_name = "2025_2_13_HNC_train_on_plasma_after_cleaning_hierarchical_clustering"
    wandb.run.name = exp_name

    #### TRAIN THE MODEL
    model = train_model(model, plasma_train_dataloaders, training_cfg)

    ##### SAVE THE MODEL
    torch.save(model.state_dict(), os.path.join(os.getcwd(), "output_plots", exp_name + "_model_HNC.pth"))

    # #### LOAD A SAVED MODEL
    # model.load_state_dict(torch.load(os.path.join(os.getcwd(), "output_plots", exp_name + "_model_HNC.pth")))

    # ##### TEST A MODEL
    # test_model(model, plasma_test_dataloaders,  label_name_for_color='staging', label_name_for_marker='sample_type', map_for_color_and_marker = None, exp_name = exp_name)
   
