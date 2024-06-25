# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import pytorch_lightning as pl
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import SimSiamPredictionHead, SimSiamProjectionHead
from noodlepy.utils.class_SpectrumDataset import SpectrumDataset
import wandb
import math
from noodlepy.utils.class_SpectrumAugmentor import SpectrumAugmentor
from noodlepy.utils.class_SpectrumPreprocessor import SpectrumPreprocessor
from noodlepy.utils.class_EmbeddingViewer import EmbeddingViewer
import random
import numpy as np

class cnn_backbone(nn.Module):
    def __init__(self, layer_channel_sizes):
        super(cnn_backbone, self).__init__()
        layers = []
        in_channels = layer_channel_sizes[0] #intialize the input channel size with the first element of the list

        for out_channels in layer_channel_sizes[1:]:
            conv_layer = nn.Conv1d(in_channels, out_channels, kernel_size=4)
            torch.nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='relu') # weights initialization using kaiming uniform
            layers.append(conv_layer)
            layers.append(nn.ReLU())
            layers.append(nn.AvgPool1d(kernel_size=3)) # layers.append(nn.MaxPool1d(kernel_size=2))
            in_channels = out_channels
        #layers.append(nn.Linear(512 * block.expansion, num_classes))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class resnet_backbone(nn.Module):
    def __init__(self):
        super(resnet_backbone, self).__init__()
        resnet = torchvision.models.resnet18()
        self.layers = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        x = self.layers(x)
        return x

class SimSiam(pl.LightningModule):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SimSiamProjectionHead(128, 128, 64)
        self.prediction_head = SimSiamPredictionHead(64, 32, 64)
        self.criterion = NegativeCosineSimilarity()

    def forward(self, x):
        f = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(f)
        p = self.prediction_head(z)
        z = z.detach()
        return z, p

    def training_step(self, batch, batch_idx):
        (x0, x1) = batch
        z0, p0 = self.forward(x0)
        z1, p1 = self.forward(x1)
        loss = 0.5 * (self.criterion(z0, p1) + self.criterion(z1, p0))
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.06)
        return optim
    
def train_model(model, train_dataloader, training_cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    criterion = NegativeCosineSimilarity()
    optimizer = torch.optim.SGD(model.parameters(), lr=training_cfg.learning_rate)

        # training
    print("Starting Training")
    for epoch in range(training_cfg.epochs):
        avg_loss = 0.0
        avg_output_std = 0.0
        #total_loss = 0.0
        for i, (x0, x1, labels) in enumerate(train_dataloader):
            x0 = x0.to(device)
            x1 = x1.to(device)
            z0, p0 = model(x0)
            z1, p1 = model(x1)
            loss = 0.5 * (criterion(z0, p1) + criterion(z1, p0))
            #total_loss += loss.detach()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # calculate the per-dimension standard deviation of the outputs
            # we can use this later to check whether the embeddings are collapsing
            output = p0.detach()
            output = torch.nn.functional.normalize(output, dim=1)
            output_std = torch.std(output, 0)
            output_std = output_std.mean()
            # use moving averages to track the loss and standard deviation
            w = 0.9
            avg_loss = w * avg_loss + (1 - w) * loss.item()
            avg_output_std = w * avg_output_std + (1 - w) * output_std.item()

        # the level of collapse is large if the standard deviation of the l2
        # normalized output is much smaller than 1 / sqrt(dim)
        collapse_level = max(0.0, 1 - math.sqrt(64) * avg_output_std)
        wandb.log({'epoch': epoch+1, 'loss': avg_loss, 'collapse_level': collapse_level})
        print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}, collapse_level: {collapse_level:.5f}")
    print("Finished Training")
    return model

def test_model(model, test_dataloader):
    print("Visualizing test set embeddings")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = []
    label_dict_list = []
    model.eval()
    with torch.no_grad():
        for i, (x, _, labels) in enumerate(test_dataloader):
            x = x.to(device)
            y = model.backbone(x).flatten(start_dim=1)
            embeddings.append(y)
            label_dict_list.append(labels)

    embeddings = torch.cat(embeddings, dim=0)
    embeddings = embeddings.cpu().numpy()

    viewer = EmbeddingViewer(embeddings, label_dict_list)
    viewer.tsne2d(label_name_for_color='staging', label_name_for_marker='sample_type', title="2d_tsne_plot")
    viewer.save_files_for_tf_embedding_projector(embedding_file_name='embeddings', metadata_file_name='metadata')


def seed_all_random_process(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    np.random.seed(seed)
    pl.seed_everything(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    

if __name__ == "__main__":

    wandb.login()
    wandb.init(
        # Set the project where this run will be logged
        project="SimSiam", 
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=f"plasma_only", 
        # Track hyperparameters and run metadata
        config={
            "learning_rate": 0.02,
            "epochs": 3,
            "batch_size": 10,
            "backbone_dim": [1, 8, 16, 32, 64, 128],
            "random_seed" : 0,
            "number_of_workers": 8
            })
    training_cfg = wandb.config
    generator = seed_all_random_process(training_cfg.random_seed)

    cnn_backbone_1d = cnn_backbone(training_cfg.backbone_dim) # 1D spectral data start with 1 channel, RGB 2D image start with 3 channels
    model = SimSiam(cnn_backbone_1d)

    train_dataset_path = os.path.join(os.getcwd(), "noodlepy", "data", "head_and_neck_cancer", "plasma_saliva_mixed","train")
    test_dataset_path = os.path.join(os.getcwd(), "noodlepy", "data", "head_and_neck_cancer", "plasma_saliva_mixed","test" )
    annotation_file_path = os.path.join(os.getcwd(), "noodlepy", "data", "Biofluid_list_annotated_v4.xlsx")

    train_preprocessor = SpectrumPreprocessor(cropping=True,
                                        baseline_correction=True,
                                        remove_cosmic_rays= True,
                                        normalization=True,
                                        smoothing=True)
    train_augmentor = SpectrumAugmentor(ramdom_augmentations=True,
                                  augmentation_step_list = None,
                                  config_path= None)
    train_dataset = SpectrumDataset(train_dataset_path, annotation_file_path, train_preprocessor, train_augmentor)

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
    test_dataset = SpectrumDataset(test_dataset_path, annotation_file_path, preprocessor=test_preprocessor, augmentor=test_augmentor)
    test_dataloaders = DataLoader(
    test_dataset,
    batch_size=training_cfg.batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=training_cfg.number_of_workers,
    worker_init_fn=seed_worker,
    generator=generator
    )

    model = train_model(model, train_dataloaders, training_cfg)
    test_model(model, test_dataloaders)
   