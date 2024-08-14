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
        name=f"mixed_plasma_saliva_HNC", 
        # Track hyperparameters and run metadata
        config={
            "learning_rate": 0.02,
            "epochs": 10,
            "batch_size": 10,
            "backbone_dim": [1, 8, 16, 32, 64, 128],
            "random_seed" : 0,
            "number_of_workers": 8
            })
    training_cfg = wandb.config
    generator = seed_all_random_process(training_cfg.random_seed)

    cnn_backbone_1d = cnn_backbone(training_cfg.backbone_dim) # 1D spectral data start with 1 channel, RGB 2D image start with 3 channels
    model = SimSiam(cnn_backbone_1d)

    train_dataset_path = os.path.join(os.getcwd(), "noodlepy", "data", "","","")
    test_dataset_path = os.path.join(os.getcwd(), "noodlepy", "data", "","","" )

    annotation_file_path = os.path.join(os.getcwd(), "noodlepy", "data", "Biofluid_list_annotated_v4.xlsx")

    train_preprocessor = SpectrumPreprocessor(cropping=True,
                                        baseline_correction=True,
                                        remove_cosmic_rays= True,
                                        normalization=True,
                                        smoothing=True)
    train_augmentor = SpectrumAugmentor(ramdom_augmentations=True,
                                  augmentation_step_list = None,
                                  config_path= None)
    
    train_dataset = HNC_Dataset(train_dataset_path, annotation_file_path, preprocessor=train_preprocessor, augmentor=train_augmentor)

    train_dataloaders = DataLoader(
        train_dataset,
        batch_size=training_cfg.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=training_cfg.number_of_workers,
        worker_init_fn=seed_worker,
        generator=generator
    )
 
    test_preprocessor = SpectrumPreprocessor(cropping=True,
                                        baseline_correction=True,
                                        remove_cosmic_rays= True,
                                        normalization=True,
                                        smoothing=True)
    test_augmentor = SpectrumAugmentor(ramdom_augmentations=True)
    test_dataset = HNC_Dataset(test_dataset_path, annotation_file_path, preprocessor=test_preprocessor, augmentor=test_augmentor)
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
    exp_name = "augment_without_amplication"

    ##### TRAIN THE MODEL
    # model = train_model(model, train_dataloaders, training_cfg)

    ##### SAVE THE MODEL
    torch.save(model.state_dict(), os.path.join(os.getcwd(), "output_plots", exp_name + "_model_OC.pth"))

    ##### LOAD A SAVED MODEL
    model.load_state_dict(torch.load(os.path.join(os.getcwd(), "output_plots", exp_name + "_model_OC_.pth")))

    ##### RESET THE COLOR MAP FOR DATASET
    map_for_color_and_marker = {'C': 'b', 'OC': 'r', 'EVs': 'x', 'serum': 'o', 'desalted1x': 'p', 'original': 'x', 'none': 'o', 'default_color':'teal', 'default_marker':'*'}

    ##### TEST A MODEL
    test_model(model, test_dataloaders,  label_name_for_color='staging', label_name_for_marker='sample_type', map_for_color_and_marker = map_for_color_and_marker, exp_name = exp_name)
   
