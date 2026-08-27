import torch
from torch import nn
import pytorch_lightning as pl

class ClassifierWithPretrainedModel(pl.LightningModule):
    def __init__(self, backbone, num_classes):
        super(ClassifierWithPretrainedModel, self).__init__()
        self.backbone = backbone  # Reuse the pretrained backbone
        self.fc1 = nn.Linear(128, 64)  # Adjust the size based on input features
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)  # New classification head (adjust input dim if needed)
        self.criterion = nn.CrossEntropyLoss()  # Classification loss function

    def forward(self, x):
        with torch.no_grad():  # Freeze backbone during forward pass
            features = self.backbone(x).flatten(start_dim=1)
        features = self.fc1(features)  # Output layer for classification
        features = self.dropout(features)
        logits = self.fc2(features)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch  # x: input data, y: labels
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)


class Classifier(pl.LightningModule):
    def __init__(self, backbone, num_classes):
        super(Classifier, self).__init__()
        self.backbone = backbone
        self.fc1 = nn.Linear(128, 64)  # Adjust the size based on input features
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
# Example usage:
# backbone = cnn_backbone([1, 32, 64, 128])
# model = TimeSeriesClassifier(backbone, num_classes=10)
# trainer = pl.Trainer(max_epochs=10)
# trainer.fit(model, dataloader)
# model.eval()
# logits = model(x)
# preds = torch.argmax(logits, dim=1)
# print(preds)
# print("Done!")