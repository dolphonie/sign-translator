# Created by Patrick Kao
import argparse
import os

import torch
from torch import nn, Tensor
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model.sign_translator_no_lightning import SignTranslatorNoLightning


def remove_slurm_vars():
    for k, v in os.environ.items():
        if "SLURM" in k:
            print(f"Deleting env variable {k}")
            del os.environ[k]


if __name__ == '__main__':
    remove_slurm_vars()

    from config import Config, LRS2Config, WholeConfig
    from data.lrs3 import LRSDataModule

    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--lrs2", action="store_true", help="Use the LRS2 Dataset params")
    parser.add_argument("-w", "--whole", action="store_true")
    args = parser.parse_args()

    config = Config
    if args.lrs2:
        config = LRS2Config
    if args.whole:
        config = WholeConfig

    data = LRSDataModule(config)
    data.setup()
    model = SignTranslatorNoLightning(config)
    train_loader = data.train_dataloader()
    optim = model.configure_optimizers()
    writer = SummaryWriter("runs")

    model = nn.DataParallel(model).to("cuda")
    for epoch in range(config.num_epochs):
        with tqdm(train_loader, unit="it") as tepoch:
            for i, batch in enumerate(tepoch):
                batch = [el.to("cuda") if isinstance(el, Tensor) else el for el in batch]
                optim.zero_grad()
                loss = model.module.training_step(batch)
                loss.backward()
                optim.step()
                writer.add_scalar("train_loss", loss.detach(), i)
                tepoch.set_postfix(loss=loss.detach().item())
        torch.save(model.state_dict(), f"model_{epoch}.pt")

    writer.close()
