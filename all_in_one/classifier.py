import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import pytorch_lightning as pl
import torchmetrics as tm
import torchvision.transforms as transforms
import torchvision.models as models


import numpy as np
import matplotlib.pyplot as plt

from .datasets import classes, Data


class Net(pl.LightningModule):

    def __init__(self, classes):
        super().__init__()
        self.n_classes = len(classes)

        self.net = models.convnext_tiny(pretrained=True)
        in_features = self.net.classifier[2].in_features
        self.net.classifier[2] = nn.Linear(in_features, len(classes))

        self.accuracy = tm.Accuracy()


    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(self(x), y)

        self.log("train_loss", loss, prog_bar=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.accuracy(preds, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.accuracy, prog_bar=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.accuracy(preds, y)

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.accuracy, prog_bar=True)

        return loss
 
    def configure_optimizers(self):
        return optim.Adam(self.parameters())