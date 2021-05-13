# Created by Patrick Kao

from typing import List, Union

import numpy as np
import pytorch_lightning as pl
import torch
from numpy import ndarray
from torch import nn, Tensor

from model.decoders.greedy_decoder import GreedyDecoder
from model.encoder import Encoder
from model.pretrain_videocnn import get_pretrained_cnn, transform_frames_for_pretrain


class SignTranslator(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.lr = self.config.lr
        self.video_encoder = get_pretrained_cnn()
        self.encoder = Encoder(config)
        self.decoder = GreedyDecoder(config)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, frames: Tensor, lengths: List[int], labels: Union[List[str], None],
                labels_id: Tensor = None):
        """

        :param frames: batch x time x channels x height x width
        :param lengths: length batch
        :param labels: length batch
        :return:
        """
        if labels_id is not None:
            labels = np.asarray(labels)
            labels = labels[labels_id.detach().cpu()]
            labels = list(labels) if isinstance(labels, ndarray) else [labels]
            # remove nunpy dtype
            labels = [str(label) for label in labels]

        frame_embed = self.video_encoder(frames)  # batch x time x out_dim
        encoder_output, encoder_padding = self.encoder(frame_embeddings=frame_embed,
                                                       lengths=lengths)
        output_logits, output_mask, labels_tokenized, labels_mask = self.decoder(
            encoder_output=encoder_output,
            encoder_padding=encoder_padding,
            target_sequence=labels)
        return output_logits, output_mask, labels_tokenized, labels_mask

    def training_step(self, batch, batch_idx):
        frames, lengths, labels, labels_id = batch
        frames_tr = transform_frames_for_pretrain(frames)
        output_logits, output_mask, labels_tokenized, labels_mask = self.forward(frames=frames_tr,
                                                                                 lengths=lengths,
                                                                                 labels=labels,
                                                                                 labels_id=labels_id)
        loss = self.masked_loss(output_logits, labels_tokenized, labels_mask)
        self.log("train_loss", loss.detach().item())
        return loss

    def masked_loss(self, output_logits, labels_tokenized, labels_mask):
        device = output_logits.device
        # trim everything to min seq len
        min_seq_len = min(output_logits.shape[0], labels_tokenized.shape[1])
        output_logits = output_logits[:min_seq_len]
        labels_tokenized = labels_tokenized[:, :min_seq_len]
        labels_mask = labels_mask[:, :min_seq_len].to(device)
        # no need to shift as in GPT2 objective, since each logit corresponds to the prediction
        # for the corresponding word
        # output_logits shape: (seq_len, batch, vocab)
        logits_contig = output_logits.permute(1, 2, 0).contiguous()  # want (batch, vocab, seq_len)
        labels_contig = labels_tokenized.contiguous().to(device)  # (batch, seq_len)
        # Flatten the tokens
        loss = self.loss_fn(logits_contig, labels_contig)  # (batch, seq_len)
        loss = torch.mean(loss * labels_mask)
        return loss

    def validation_step(self, batch, batch_idx):
        self._compute_metrics(batch, prefix="val")

    def test_step(self, batch, batch_idx):
        self._compute_metrics(batch, prefix="test")

    def _compute_metrics(self, batch, prefix="val"):
        frames, lengths, labels, labels_id = batch
        frames_tr = transform_frames_for_pretrain(frames)
        output_logits, output_mask, _, _ = self.forward(frames=frames_tr,
                                                        lengths=lengths,
                                                        labels=None,
                                                        labels_id=None)
        if labels_id is not None:
            labels = np.asarray(labels)
            labels = labels[labels_id.detach().cpu()]
            labels = list(labels) if isinstance(labels, ndarray) else [labels]
            # remove nunpy dtype
            labels = [str(label) for label in labels]

        labels_tokenized, labels_mask = self.decoder.language_model.tokenize(labels)
        # no need to shift as in GPT2 objective, since each logit corresponds to the
        # prediction
        # for the corresponding word
        loss = self.masked_loss(output_logits, labels_tokenized, labels_mask)

        sync_dist = torch.cuda.device_count() > 1
        self.log(f"{prefix}_loss", loss.detach().item(), sync_dist=sync_dist)

        greedy_ids = torch.argmax(output_logits, dim=2)
        output_mask[0] = 0
        _, mean_wer = self.decoder.language_model.get_wer(greedy_ids.T, output_mask.T,
                                                          labels)
        self.log(f"{prefix}_wer", mean_wer, sync_dist=sync_dist)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.lr)
        return optimizer
