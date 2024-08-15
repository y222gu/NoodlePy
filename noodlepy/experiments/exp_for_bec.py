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

    generator = seed_all_random_process(0)

    cnn_backbone_1d = cnn_backbone([1, 8, 16, 32, 64, 128]) # 1D spectral data start with 1 channel, RGB 2D image start with 3 channels
    model = SimSiam(cnn_backbone_1d)


    test_dataset_path = os.path.join(os.getcwd(), "noodlepy", "data", "Bec_data","unprocessed","all")

    annotation_file_path = os.path.join(os.getcwd(), "noodlepy", "data", "python_test_patient_staging.xlsx")

    test_preprocessor = SpectrumPreprocessor(cropping=True,
                                        baseline_correction=True,
                                        remove_cosmic_rays= True,
                                        normalization=True,
                                        smoothing=True)
    test_augmentor = SpectrumAugmentor(ramdom_augmentations=True)
    test_dataset = HNC_Dataset(test_dataset_path, annotation_file_path, preprocessor=test_preprocessor, augmentor=test_augmentor)
    test_dataloaders = DataLoader(
    test_dataset,
    batch_size=10,
    shuffle=False,
    drop_last=False,
    num_workers=8,
    worker_init_fn=seed_worker,
    generator=generator
    )

    ##### NAME OF THE MODEL
    exp_name = 'current'

    ##### LOAD A SAVED MODEL
    model.load_state_dict(torch.load(os.path.join(os.getcwd(), "output_plots", exp_name + "_model_HNC.pth")))

    ##### TEST A MODEL
    test_model(model, test_dataloaders,  label_name_for_color='location', label_name_for_marker='sample_type', map_for_color_and_marker = None, exp_name = exp_name)
   
