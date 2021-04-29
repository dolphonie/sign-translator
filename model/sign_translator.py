# Created by Patrick Kao
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

from model.encoder import Encoder
from model.pretrain_videocnn import get_pretrained_cnn


class SignTranslator(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.video_encoder = get_pretrained_cnn()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.linear = nn.Linear(5,5)


    def forward(self, x):
        frames, labels, lengths = x


    def training_step(self, batch, batch_idx):

        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.config.lr)
        return optimizer