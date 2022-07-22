import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import pytorch_lightning as pl
import torchmetrics as tm
import torchvision.models as models

import numpy as np
import matplotlib.pyplot as plt

from datasets import TrainSet, EvalSet, get_train_sampler, classes

import argparse

class Net(pl.LightningModule):

    def __init__(self, classes):
        super().__init__()
        self.n_classes = len(classes)

        self.net = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', nargs='?', type=str, default='train')
    args = parser.parse_args()

    if args.task == 'train':
        name = 'all_in_one'
        version = 1

        model = Net(classes)
        trainloader = data.DataLoader(TrainSet(), batch_size=64, sampler=get_train_sampler(), num_workers=6)
        valloader = data.DataLoader(EvalSet(n_toilet_samples=2000, n_sink_samples=2000), batch_size=64, num_workers=6)

        logger = pl.loggers.TensorBoardLogger('lightning_logs', name=name, version=version)
        trainer = pl.Trainer(
            accelerator='gpu',
            devices=1,
            max_epochs=-1,
            logger=logger,
            log_every_n_steps=20,
            callbacks=[
                pl.callbacks.progress.TQDMProgressBar(refresh_rate=20),
                pl.callbacks.ModelCheckpoint(monitor="val_loss", save_last=True, save_top_k=3, filename='{epoch}-{val_loss:.3f}-{val_acc:.3f}')
            ],
        )

        trainer.fit(model, trainloader, valloader)
    elif args.task == 'test':
        pass
