import os
import torch
import random
from torch.utils.data import DataLoader
from noodlepy.utils.bec_hnc_dataset import Bec_HNC_Dataset
from noodlepy.utils.spectrumpreprocessor import SpectrumPreprocessor
from noodlepy.utils.spectrumaugmentor import SpectrumAugmentor
from noodlepy.ml.simsiam import SimSiam
from noodlepy.ml.backbone import cnn_backbone
import wandb
from noodlepy.utils.seed import seed_all_random_process, seed_worker
from noodlepy.utils.traintestmodel import train_model, test_model
import numpy as np


if __name__ == "__main__":
    seed = 4
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


    wandb.login()
    wandb.init(
        project="SimSiam", 
        name=f"2023_03_03_Bec_HNC_batch_corrected", 
        # Track hyperparameters and run metadata
        config={
            "learning_rate": 0.001,
            "epochs": 80,
            "batch_size": 128,
            "backbone_dim": [1, 8, 16, 32, 64, 128],
            "random_seed" : 0,
            "number_of_workers": 8
            })
    training_cfg = wandb.config
    generator = seed_all_random_process(training_cfg.random_seed)

    cnn_backbone_1d = cnn_backbone(training_cfg.backbone_dim) # 1D spectral data start with 1 channel, RGB 2D image start with 3 channels
    model = SimSiam(cnn_backbone_1d)

    train_dataset_path = os.path.join(os.getcwd(), "noodlepy", "data", "bec_hnc", "train_encoder")

    metadata_file = os.path.join(os.getcwd(), "noodlepy", "data", "bec_hnc_patient_annotations.xlsx")

    preprocessor = SpectrumPreprocessor(cropping=False,
                                        baseline_correction=True,
                                        remove_cosmic_rays= True,
                                        normalization=True,
                                        smoothing=True)
    augmentor = SpectrumAugmentor(ramdom_augmentations=True,
                                  augmentation_step_list = None,
                                  config_path= None)
    
    train_dataset = Bec_HNC_Dataset(train_dataset_path, metadata_file, preprocessor=preprocessor, augmentor=augmentor)

    train_dataset.combat_batch_correction()

    train_dataloaders = DataLoader(
        train_dataset,
        batch_size=training_cfg.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=training_cfg.number_of_workers,
        worker_init_fn=seed_worker,
        generator=generator
    )

    ##### NAME OF THE MODEL
    exp_name = "2023_03_03_Bec_HNC_batch_corrected"

    #### TRAIN THE MODEL
    model = train_model(model, train_dataloaders, training_cfg)

    #### SAVE THE MODEL
    if not os.path.exists(os.path.join(os.getcwd(), "output_plots", exp_name)):
        os.makedirs(os.path.join(os.getcwd(), "output_plots", exp_name))
    torch.save(model.state_dict(), os.path.join(os.getcwd(), "output_plots", exp_name , "model_HNC.pth"))

    # ##### LOAD A SAVED MODEL
    # model.load_state_dict(torch.load(os.path.join(os.getcwd(), "output_plots", exp_name, "model_HNC.pth")))

    ##### TEST A MODEL
    test_model(model, train_dataloaders,  label_name_for_color='staging',  map_for_color = None, exp_name = exp_name)
   
