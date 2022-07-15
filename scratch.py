import numpy as np
import pandas as pd
import torchmetrics
import sklearn
import torch
import pytorch_lightning as pl
from typing import Any
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import random_split, DataLoader
import os

from pytorch_lightning import LightningDataModule as ldm


class LitDataSet(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.dataset = MNIST
        self.train_size = 0.8
        self.data_dir = os.path.join(os.getcwd(), "data")

    def prepare_data(self):
        self.full_dataset = self.dataset(
            download=True,
            train=True,
            root=os.path.join(os.getcwd(), "data"),
            transform=ToTensor(),
        )

    def setup(self, stage: str):
        if stage in ["train", "val"]:
            self.train, self.val = random_split(
                self.full_dataset, [self.train_size, 1 - self.train_size]
            )
        else:
            self.test = self.full_dataset(
                self.data_dir, train=False, transform=self.transforms
            )

    def train_dataloader(self):
        return DataLoader(self.train)

    def val_dataloader(self):
        return DataLoader(self.val)

    def test_dataloader(self):
        return DataLoader(self.test)


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 64), torch.nn.ReLU(), torch.nn.Linear(64, 3)
        )

    def forward(self, x: Any) -> torch.TensorType:
        return self.l1(x)


class Decoder(torch.nn.Module):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__()
        self.l1 = torch.nn.Sequential(
            torch.nn.Linear(3, 64), torch.nn.ReLU(), torch.nn.Linear(64, 28 * 28)
        )

    def forward(self, x: Any) -> torch.TensorType:
        return self.l1(x)


class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x_hat = self.decoder(x)
        return x_hat

    def training_step(self, batch, *args, **kwargs):
        x, y = batch
        x = self.encoder(x)
        x_hat = self.decoder(x)
        loss = torchmetrics.MeanSquaredError(x_hat, x)
        self.log("loss", loss)
        return loss

    def test_step(self, batch):
        self._common_eval(batch, step="test")

    def validation_step(self, batch):
        self._common_eval(batch, "val")

    def _common_eval(self, batch, *args, step: str = None):
        x, y = batch
        x = self.encoder(x)
        x_hat = self.decoder(x)
        loss = torchmetrics.MeanSquaredError(x_hat, x)
        self.log(f"{step}-loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), eps=1e-4)
        return optimizer


from pytorch_lightning import Trainer

model = LitModel()
data = LitDataSet()
trainer = Trainer(fast_dev_run=True)
trainer.fit(model=model, datamodule=data)
