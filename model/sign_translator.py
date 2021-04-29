# Created by Patrick Kao
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

class SignTranslator(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # self.video_encoder = nn.Conv2d()
        # self.transformer_encoder = nn.TransformerEncoder()
        # self.transformer_decoder = nn.TransformerDecoder()
        self.linear = nn.Linear(5,5)


    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        frames, labels, lengths = batch
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.config.lr)
        return optimizer