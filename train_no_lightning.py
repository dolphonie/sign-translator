# Created by Patrick Kao
import argparse
import os

import jiwer
from jiwer import wer
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
                batch = [el.to("cuda") if isinstance(el, Tensor) else el for el in batch]
                optim.zero_grad()
                frames, lengths, labels, labels_id = batch
                frames_tr = transform_frames_for_pretrain(frames)
                output_logits, labels_tokenized = model(frames=frames_tr,
                                                               lengths=lengths,
                                                               labels=labels,
                                                               labels_id=labels_id)
                # no need to shift as in GPT2 objective, since each logit corresponds to the
                # prediction
                # for the corresponding word
                logits_contig = output_logits.permute(1, 0, 2).contiguous()  # want batch first
                labels_contig = labels_tokenized.contiguous()
                # Flatten the tokens
                loss = model.module.loss_fn(logits_contig.view(-1, logits_contig.size(-1)),
                                    labels_contig.view(-1))
                loss.backward()
                optim.step()
                writer.add_scalar("train_loss", loss.detach(), i)
                tepoch.set_postfix(loss=loss.detach().item())

        model.eval()
        # Validation
        with tqdm(train_loader, unit="it") as vepoch:
            for i, batch in enumerate(vepoch):
                batch = [el.to("cuda") if isinstance(el, Tensor) else el for el in batch]
                frames, lengths, labels, labels_id = batch
                frames_tr = transform_frames_for_pretrain(frames)
                output_logits, labels_tokenized = model(frames=frames_tr,
                                                        lengths=lengths,
                                                        labels=labels,
                                                        labels_id=labels_id)
                # no need to shift as in GPT2 objective, since each logit corresponds to the
                # prediction
                # for the corresponding word
                logits_contig = output_logits.permute(1, 0, 2).contiguous()  # want batch first
                labels_contig = labels_tokenized.contiguous()
                # Flatten the tokens
                loss = model.loss_fn(logits_contig.view(-1, logits_contig.size(-1)),
                                            labels_contig.view(-1))
                writer.add_scalar("val_loss", loss.detach(), i)
                vepoch.set_postfix(loss=loss.detach().item())

                _, mean_wer = model.decoder.language_model.get_wer(labels_tokenized, labels)
                writer.add_scalar("val_wer", mean_wer, i)

        torch.save(model.state_dict(), f"model_{epoch}.pt")

    writer.close()
