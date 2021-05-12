# Created by Patrick Kao
import argparse
import os

import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model.pretrain_videocnn import transform_frames_for_pretrain
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
    val_loader = data.val_dataloader()
    optim = model.configure_optimizers()
    writer = SummaryWriter("runs")

    model = nn.DataParallel(model).to("cuda")
    for epoch in range(config.num_epochs):
        model.train()
        with tqdm(train_loader, unit="it") as tepoch:
            for i, batch in enumerate(tepoch):
                # batch = [el.to("cuda") if isinstance(el, Tensor) else el for el in batch]
                optim.zero_grad()
                frames, lengths, labels, labels_id = batch
                frames_tr = transform_frames_for_pretrain(frames)
                output_logits, output_mask, labels_tokenized, labels_mask = model(frames=frames_tr,
                                                               lengths=lengths,
                                                               labels=labels,
                                                               labels_id=labels_id)
                loss = model.module.masked_loss(output_logits, labels_tokenized, labels_mask)
                loss.backward()
                optim.step()
                writer.add_scalar("train_loss", loss.detach(), i)
                tepoch.set_postfix(loss=loss.detach().item())

        model.eval()
        # Validation
        with tqdm(train_loader, unit="it") as vepoch:
            for i, batch in enumerate(vepoch):
                # batch = [el.to("cuda") if isinstance(el, Tensor) else el for el in batch]
                frames, lengths, labels, labels_id = batch
                frames_tr = transform_frames_for_pretrain(frames)
                output_logits, output_mask, _, _ = model(frames=frames_tr,
                                                        lengths=lengths,
                                                        labels=None,
                                                        labels_id=None)
                if labels_id is not None:
                    labels = np.asarray(labels)
                    labels = labels[labels_id.detach().cpu()]
                    labels = list(labels)
                labels_tokenized, labels_mask = model.decoder.language_model.tokenize(labels)
                # no need to shift as in GPT2 objective, since each logit corresponds to the
                # prediction
                # for the corresponding word
                loss = model.module.masked_loss(output_logits, labels_tokenized, labels_mask)

                writer.add_scalar("val_loss", loss.detach(), i)
                vepoch.set_postfix(loss=loss.detach().item())

                greedy_ids = torch.argmax(output_logits, dim=2)
                output_mask[0] = 0
                _, mean_wer = model.module.decoder.language_model.get_wer(greedy_ids.T, output_mask.T, labels)
                writer.add_scalar("val_wer", mean_wer, i)

        torch.save(model.state_dict(), f"model_{epoch}.pt")

    writer.close()
